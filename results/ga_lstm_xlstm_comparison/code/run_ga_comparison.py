"""Validation-driven GA -> rerank -> lock -> three-seed train -> test.

Reuses the exact data/tokenizer/model/training code of the fixed baseline.
--resume replays the deterministic GA and reuses complete runs after checking
the locked configuration, data, tokenizer and source hashes. Partial runs are
preserved under interrupted/ before a fresh attempt of that same trial.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import traceback
import warnings

from compare_lstm_xlstm import (ROOT, XLSTM_SOURCE, Classifier, TextDataset, collate,
    dump, metrics, normalize, predict, read_split, sha, spm, torch, pd, train_one, DataLoader)
from ga_search import search, rank


def now():
    return datetime.now(timezone.utc).isoformat()


def fit(kind, seed, cfg, parent, data, device):
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / f'{kind}_seed{seed}'
    config_file = parent / f'{kind}_seed{seed}_config.json'
    if config_file.exists():
        if json.loads(config_file.read_text(encoding='utf-8')) != cfg:
            raise ValueError(f'Cached configuration mismatch: {config_file}')
    else:
        dump(config_file, cfg)
    done = folder / 'run_metadata.json'
    if done.exists():
        if not (folder / 'best.pt').exists():
            raise FileNotFoundError(f'Missing cached checkpoint: {folder}')
        return json.loads(done.read_text(encoding='utf-8'))
    if folder.exists():
        interrupted = parent / 'interrupted'
        interrupted.mkdir(exist_ok=True)
        target = interrupted / f'{folder.name}_{time.time_ns()}'
        # Both paths are explicitly constrained to this experiment's parent.
        if not folder.resolve().is_relative_to(parent.resolve()) or not target.resolve().is_relative_to(parent.resolve()):
            raise ValueError('Invalid interrupted-run path')
        shutil.move(str(folder), str(target))
    with (parent / f'{kind}_seed{seed}_train.log').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='cumsum_cuda_kernel does not have a deterministic implementation.*')
            result = train_one(kind, seed, *data, cfg, parent, device)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ga_lstm_xlstm_comparison.json')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'results/ga_lstm_xlstm_comparison')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    spec = json.loads(args.config.read_text(encoding='utf-8'))
    baseline = ROOT / spec['baseline_directory']
    base = json.loads((baseline / 'locked_config.json').read_text(encoding='utf-8'))
    original_env = json.loads((baseline / 'environment.json').read_text(encoding='utf-8'))
    if sha(ROOT / 'scripts/compare_lstm_xlstm.py') != original_env['script_sha256']:
        raise ValueError('The fixed-baseline training implementation changed')
    out = args.outdir.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f'Use a fresh directory or --resume: {out}')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(base['num_threads'])
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    hashes = {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path) for path in
              [ROOT / 'scripts/compare_lstm_xlstm.py', ROOT / 'scripts/ga_search.py', Path(__file__),
               ROOT / base['train'], ROOT / base['validation'], ROOT / 'label_mapping.json']}
    hashes.update({str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in (XLSTM_SOURCE / 'xlstm').rglob('*.py')})
    for split in ['train', 'validation']:
        if sha(ROOT / base[split]) != original_env['hashes'][base[split]]:
            raise ValueError('Baseline data changed')
    if sha(baseline / 'tokenizer.model') != original_env['tokenizer_sha256']:
        raise ValueError('Baseline tokenizer changed')
    lock = {'spec': spec, 'base_training_config': base, 'hashes': hashes,
            'tokenizer_sha256': original_env['tokenizer_sha256'], 'device': str(device),
            'packages': {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'pandas', 'scikit-learn', 'sentencepiece', 'einops']}}
    if (out / 'locked_protocol.json').exists():
        if json.loads((out / 'locked_protocol.json').read_text(encoding='utf-8')) != lock:
            raise ValueError('Cannot resume: protocol, environment, data or code changed')
    else:
        dump(out / 'locked_protocol.json', lock)
        dump(out / 'environment.json', {'started_utc': now(), 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
             'precision': 'float32', 'deterministic_algorithms': 'warn_only',
             'determinism_limit': original_env['determinism_limit'], 'test_evaluated': False})
        for name in ['tokenizer.model', 'tokenizer.vocab']:
            shutil.copyfile(baseline / name, out / name)
        code = out / 'code'
        code.mkdir()
        for name in ['compare_lstm_xlstm.py', 'ga_search.py', 'run_ga_comparison.py']:
            shutil.copyfile(ROOT / 'scripts' / name, code / name)
    labels = json.loads((ROOT / 'label_mapping.json').read_text(encoding='utf-8'))['label2id']
    names = [k for k, v in sorted(labels.items(), key=lambda pair: pair[1])]
    train, val = (read_split(ROOT / base[s], labels) for s in ['train', 'validation'])
    matching = lambda text: ' '.join(normalize(text).lower().split())
    if set(train.text.map(matching)) & set(val.text.map(matching)):
        raise ValueError('Train/validation text overlap')
    tokenizer = spm.SentencePieceProcessor(model_file=str(out / 'tokenizer.model'))
    train_ds = TextDataset(train, tokenizer, base['max_length'])
    val_ds = TextDataset(val, tokenizer, base['max_length'])
    data = (train_ds, val_ds, val, names, tokenizer.get_piece_size())
    selected = {}
    for kind in spec['models']:
        root = out / 'search' / kind
        root.mkdir(parents=True, exist_ok=True)
        def evaluate(genes, trial_id, generation):
            trial = root / f'trial_{trial_id:03d}'
            trial.mkdir(exist_ok=True)
            result_path = trial / 'result.json'
            if result_path.exists():
                cached = json.loads(result_path.read_text(encoding='utf-8'))
                if cached['genes'] != genes:
                    raise ValueError('GA replay produced different genes')
                return cached
            cfg = {**base, **genes, 'max_epochs': spec['ga']['proxy_epochs'], 'patience': spec['ga']['proxy_epochs'] + 1}
            print(f'SEARCH {kind} trial={trial_id+1}/{spec["ga"]["unique_trial_budget"]} generation={generation} genes={genes}', flush=True)
            begin = time.perf_counter()
            try:
                metadata = fit(kind, spec['ga']['training_seed'], cfg, trial, data, device)
                result = {'status': 'ok', 'fitness': metadata['validation_macro_f1'], 'run': metadata}
            except (torch.cuda.OutOfMemoryError, FloatingPointError) as exc:
                (trial / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
                result = {'status': 'failed', 'fitness': None, 'error': str(exc)}
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            result.update({'genes': genes, 'elapsed_seconds': time.perf_counter() - begin})
            dump(result_path, result)
            print(f'RESULT {kind} trial={trial_id+1} fitness={result["fitness"]}', flush=True)
            return result
        ranked = search(spec['search_space'], spec['ga'], {k: base[k] for k in spec['search_space']},
                        evaluate, lambda state: dump(root / 'search_state.json', state))
        successful = [r for r in ranked if r['fitness'] is not None]
        if len(successful) < spec['rerank']['top_k']:
            raise RuntimeError('Not enough successful GA candidates to rerank')
        reranked = []
        for candidate in successful[:spec['rerank']['top_k']]:
            cfg = {**base, **candidate['genes'], 'max_epochs': spec['rerank']['max_epochs'], 'patience': spec['rerank']['patience']}
            parent = out / 'rerank' / kind / f'trial_{candidate["trial_id"]:03d}'
            runs = []
            for seed in spec['rerank']['seeds']:
                print(f'RERANK {kind} trial={candidate["trial_id"]} seed={seed}', flush=True)
                runs.append(fit(kind, seed, cfg, parent, data, device))
            row = {'trial_id': candidate['trial_id'], 'genes': candidate['genes'],
                   'mean_validation_macro_f1': sum(r['validation_macro_f1'] for r in runs) / len(runs), 'runs': runs}
            reranked.append(row)
            dump(out / f'reranking_{kind}.json', reranked)
        chosen = min(reranked, key=lambda row: (-row['mean_validation_macro_f1'], row['trial_id']))
        selected[kind] = {**chosen, 'config': {**base, **chosen['genes'],
                           'max_epochs': spec['final']['max_epochs'], 'patience': spec['final']['patience']}}
        print(f'SELECTED {kind}: {chosen["genes"]} rerank_F1={chosen["mean_validation_macro_f1"]:.4f}', flush=True)
    if not (out / 'selected_configs.json').exists():
        dump(out / 'selected_configs.json', {'locked_utc': now(), 'selected': selected})
    elif json.loads((out / 'selected_configs.json').read_text(encoding='utf-8'))['selected'] != selected:
        raise ValueError('Selection changed on resume')
    runs = []
    for seed in spec['final']['seeds']:
        for kind in spec['models']:
            print(f'FINAL TRAIN {kind} seed={seed}', flush=True)
            run = fit(kind, seed, selected[kind]['config'], out / 'final', data, device)
            runs.append(run)
    if not (out / 'training_complete.json').exists():
        dump(out / 'training_complete.json', {'completed_utc': now(), 'runs': runs})
    # FIRST access to test rows and labels happens only after every final run.
    test = read_split(ROOT / base['test'], labels)
    test_ds = TextDataset(test, tokenizer, base['max_length'])
    loader = DataLoader(test_ds, batch_size=base['batch_size'], collate_fn=collate)
    for run in runs:
        folder = out / 'final' / f'{run["model"]}_seed{run["seed"]}'
        checkpoint = torch.load(folder / 'best.pt', map_location=device, weights_only=True)
        model = Classifier(run['model'], tokenizer.get_piece_size(), selected[run['model']]['config']).to(device)
        model.load_state_dict(checkpoint['model_state'])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='cumsum_cuda_kernel does not have a deterministic implementation.*')
            pred = predict(model, loader, device)
        scores = metrics(test.label_id.tolist(), pred, names)
        dump(folder / 'metrics_test.json', scores)
        pd.DataFrame({'sample_id': test.sample_id, 'pred_label': [names[p] for p in pred]}).to_csv(folder / 'predictions_test.csv', index=False)
        run.update({f'test_{k}': scores[k] for k in ['accuracy', 'macro_f1', 'weighted_f1']})
        del model, checkpoint
    pd.DataFrame(runs).to_csv(out / 'summary_runs.csv', index=False)
    environment = json.loads((out / 'environment.json').read_text(encoding='utf-8'))
    environment.update({'completed_utc': now(), 'test_evaluated': True, 'test_sha256': sha(ROOT / base['test']),
                        'tokenization': {'train': train_ds.stats, 'validation': val_ds.stats, 'test': test_ds.stats}})
    dump(out / 'environment.json', environment)
    print(pd.DataFrame(runs).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
