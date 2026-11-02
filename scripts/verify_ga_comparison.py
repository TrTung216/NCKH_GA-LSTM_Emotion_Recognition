"""Audit actual GA lineage, budget, selection, hashes, and independently re-score."""
import argparse
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch
import sys
import numpy as np
import pandas as pd
from compare_lstm_xlstm import ROOT, sha
import evaluate as independent_evaluator
from ga_search import key, rank


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=ROOT / 'results/ga_lstm_xlstm_comparison')
    args = parser.parse_args()
    out = args.outdir.resolve()
    lock = read(out / 'locked_protocol.json')
    spec, base = lock['spec'], lock['base_training_config']
    selected = read(out / 'selected_configs.json')['selected']
    env = read(out / 'environment.json')
    for path, expected in lock['hashes'].items():
        assert sha(ROOT / path) == expected, f'Changed file: {path}'
    assert sha(out / 'tokenizer.model') == lock['tokenizer_sha256']
    assert sha(ROOT / base['test']) == env['test_sha256']
    for source in (out / 'code').glob('*.py'):
        assert sha(source) == sha(ROOT / 'scripts' / source.name)
    audit = out / 'verification'
    audit.mkdir(exist_ok=True)
    checked = []

    def verify_run(parent, kind, seed, expected_cfg, splits):
        folder = parent / f'{kind}_seed{seed}'
        run = read(folder / 'run_metadata.json')
        actual_config = read(parent / f'{kind}_seed{seed}_config.json')
        assert actual_config == expected_cfg, folder
        h = pd.read_csv(folder / 'history.csv')
        assert np.isfinite(h.select_dtypes('number').to_numpy()).all()
        assert h.epoch.tolist() == list(range(1, len(h) + 1))
        assert len(h) == run['epochs_run'] and len(h) <= expected_cfg['max_epochs']
        best = h.loc[h.validation_macro_f1.idxmax()]
        assert int(best.epoch) == run['best_epoch']
        assert np.isclose(best.validation_macro_f1, run['validation_macro_f1'], atol=1e-12, rtol=0)
        if len(h) < expected_cfg['max_epochs']:
            assert len(h) - int(best.epoch) == expected_cfg['patience']
        for split in splits:
            target = audit / ('_'.join(folder.relative_to(out).parts) + f'_{split}.json')
            argv = ['evaluate.py', '--gold', str(ROOT / base[split]),
                    '--pred', str(folder / f'predictions_{split}.csv'), '--output', str(target)]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                independent_evaluator.main()
            independent = read(target)
            saved = read(folder / f'metrics_{split}.json')
            for metric in ['accuracy', 'macro_f1', 'weighted_f1']:
                assert np.isclose(independent[metric], saved[metric], atol=1e-12, rtol=0), (folder, split, metric)
            assert independent['confusion_matrix_rows_true_columns_predicted'] == saved['confusion_matrix_rows_true_columns_predicted']
            assert independent['confusion_matrix_labels'] == saved['confusion_matrix_labels']
            if split == 'validation':
                assert np.isclose(saved['macro_f1'], run['validation_macro_f1'], atol=1e-12, rtol=0)
            checked.append(str(folder.relative_to(out)) + '/' + split)
        if splits == ['validation']:
            assert not (folder / 'metrics_test.json').exists()
            assert not (folder / 'predictions_test.csv').exists()
        return run

    initial = []
    for kind in spec['models']:
        state = read(out / 'search' / kind / 'search_state.json')
        trials = state['trials']
        assert len(trials) == spec['ga']['unique_trial_budget']
        assert [r['trial_id'] for r in trials] == list(range(len(trials)))
        assert len({key(r['genes']) for r in trials}) == len(trials)
        initial.append([r['genes'] for r in trials[:spec['ga']['population_size']]])
        assert initial[-1][0] == {name: base[name] for name in spec['search_space']}
        for r in trials:
            assert all(v in spec['search_space'][name] for name, v in r['genes'].items())
            if r['generation']:
                assert len(r['parents']) == 2
                prior = state['generations'][r['generation'] - 1]['population_trial_ids']
                assert set(r['parents']).issubset(prior)
            if r['status'] == 'ok':
                cfg = {**base, **r['genes'], 'max_epochs': spec['ga']['proxy_epochs'], 'patience': spec['ga']['proxy_epochs'] + 1}
                run = verify_run(out / 'search' / kind / f'trial_{r["trial_id"]:03d}', kind, spec['ga']['training_seed'], cfg, ['validation'])
                assert run['epochs_run'] == spec['ga']['proxy_epochs']
                assert np.isclose(run['validation_macro_f1'], r['fitness'], atol=1e-12, rtol=0)
        assert len(state['generations']) == spec['ga']['offspring_generations'] + 1
        for generation in state['generations']:
            ids = generation['population_trial_ids']
            assert len(set(ids)) == spec['ga']['population_size']
            assert generation['best_trial_id'] == min((trials[i] for i in ids), key=rank)['trial_id']
            if generation['generation']:
                previous = state['generations'][generation['generation'] - 1]['population_trial_ids']
                elite_ids = {r['trial_id'] for r in sorted((trials[i] for i in previous), key=rank)[:spec['ga']['elites']]}
                assert elite_ids.issubset(ids)
        top = sorted((r for r in trials if r['status'] == 'ok'), key=rank)[:spec['rerank']['top_k']]
        reranked = read(out / f'reranking_{kind}.json')
        assert [r['trial_id'] for r in top] == [r['trial_id'] for r in reranked]
        for r in reranked:
            cfg = {**base, **r['genes'], 'max_epochs': spec['rerank']['max_epochs'], 'patience': spec['rerank']['patience']}
            runs = [verify_run(out / 'rerank' / kind / f'trial_{r["trial_id"]:03d}', kind, seed, cfg, ['validation']) for seed in spec['rerank']['seeds']]
            assert np.isclose(np.mean([x['validation_macro_f1'] for x in runs]), r['mean_validation_macro_f1'], atol=1e-12, rtol=0)
        best = min(reranked, key=lambda r: (-r['mean_validation_macro_f1'], r['trial_id']))
        assert selected[kind]['trial_id'] == best['trial_id']
        assert selected[kind]['genes'] == best['genes']
        final_cfg = {**base, **best['genes'], 'max_epochs': spec['final']['max_epochs'], 'patience': spec['final']['patience']}
        assert selected[kind]['config'] == final_cfg
        for seed in spec['final']['seeds']:
            verify_run(out / 'final', kind, seed, final_cfg, ['validation', 'test'])
    assert initial[0] == initial[1], 'Initial populations differ'
    summary = pd.read_csv(out / 'summary_runs.csv')
    assert len(summary) == len(spec['models']) * len(spec['final']['seeds'])
    assert set(zip(summary.model, summary.seed)) == {(k, s) for k in spec['models'] for s in spec['final']['seeds']}
    for _, row in summary.iterrows():
        scores = read(out / 'final' / f'{row.model}_seed{row.seed}/metrics_test.json')
        for metric in ['accuracy', 'macro_f1', 'weighted_f1']:
            assert np.isclose(row[f'test_{metric}'], scores[metric], atol=1e-12, rtol=0)
    report = {'status': 'passed', 'n_prediction_sets_independently_rescored': len(checked),
              'hashes_verified': True, 'same_initial_population_verified': True,
              'unique_trial_budget_and_elitism_verified': True, 'reranking_and_selection_verified': True,
              'effective_training_configs_verified': True, 'epoch_selection_verified': True,
              'no_test_predictions_in_search_or_rerank': True, 'checked': checked}
    (audit / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'checked'}, indent=2))


if __name__ == '__main__':
    main()
