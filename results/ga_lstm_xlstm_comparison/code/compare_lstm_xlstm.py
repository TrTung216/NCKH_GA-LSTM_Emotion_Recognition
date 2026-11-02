"""Fixed multi-seed LSTM vs official NeurIPS xLSTM (mLSTM-only) experiment.

Fit the tokenizer on train only, select epochs on validation, then evaluate
all locked runs on test. Never select a seed or tune using test scores.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
XLSTM_COMMIT = 'ab22eadbd245f293dd8dec38ed29963d73758a12'
XLSTM_SOURCE = ROOT / 'vendor' / f'xlstm-{XLSTM_COMMIT}'
if XLSTM_SOURCE.exists():
    sys.path.insert(0, str(XLSTM_SOURCE))
if (ROOT / '.experiment_deps').exists():
    sys.path.insert(0, str(ROOT / '.experiment_deps'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Official mLSTM uses CUDA cumsum, which has no deterministic implementation
    # in PyTorch 2.5.1. Record this limit; use deterministic kernels where available.
    torch.use_deterministic_algorithms(True, warn_only=True)


def normalize(text):
    return unicodedata.normalize('NFC', text)


def read_split(path, labels):
    df = pd.read_csv(path, keep_default_na=False)
    if df.empty or not df.sample_id.is_unique or not df.text.str.strip().all():
        raise ValueError(f'Invalid rows/IDs/text in {path}')
    if not all(labels.get(label) == idx for label, idx in zip(df.label, df.label_id)):
        raise ValueError(f'Inconsistent labels in {path}')
    return df


class TextDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        raw = [tokenizer.encode(normalize(t)) for t in frame.text]
        self.stats = {'n_samples': len(raw), 'truncated': sum(len(t) + 1 > max_length for t in raw),
                      'unknown_tokens': sum(t.count(tokenizer.unk_id()) for t in raw),
                      'tokens_before_truncation': sum(map(len, raw))}
        self.tokens = [torch.tensor(t[:max_length - 1] + [tokenizer.eos_id()]) for t in raw]
        self.labels = frame.label_id.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


def collate(batch):
    tokens, labels = zip(*batch)
    return nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0), torch.tensor(labels)


class Classifier(nn.Module):
    def __init__(self, kind, vocab_size, cfg):
        super().__init__()
        dim = cfg['embedding_dim']
        self.kind = kind
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.input_dropout = nn.Dropout(cfg['dropout'])
        if kind == 'lstm':
            self.encoder = nn.LSTM(dim, dim, num_layers=cfg['num_layers'], batch_first=True,
                                   dropout=cfg['dropout'] if cfg['num_layers'] > 1 else 0)
            self.norm = nn.LayerNorm(dim)
        else:
            self.encoder = xLSTMBlockStack(xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)),
                slstm_block=None, context_length=cfg['max_length'], num_blocks=cfg['num_layers'],
                embedding_dim=dim, slstm_at=[], dropout=cfg['dropout']))
            self.norm = nn.Identity()  # Official stack already has final normalization.
        self.head = nn.Sequential(nn.Dropout(cfg['dropout']), nn.Linear(dim, 7))

    def forward(self, ids):
        mask = ids.ne(0).unsqueeze(-1)
        x = self.input_dropout(self.embedding(ids))
        x = self.encoder(x)[0] if self.kind == 'lstm' else self.encoder(x)
        # Both encoders are causal: right padding cannot affect valid tokens.
        x = self.norm(x)
        pooled = (x * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(pooled)


def metrics(y, pred, label_names):
    return {'n_samples': len(y), 'metric_scale': '0 to 1',
            'accuracy': accuracy_score(y, pred),
            'macro_f1': f1_score(y, pred, labels=list(range(7)), average='macro', zero_division=0),
            'weighted_f1': f1_score(y, pred, labels=list(range(7)), average='weighted', zero_division=0),
            'per_class': classification_report(y, pred, labels=list(range(7)), target_names=label_names,
                                               output_dict=True, zero_division=0),
            'confusion_matrix_labels': label_names,
            'confusion_matrix_rows_true_columns_predicted': confusion_matrix(y, pred, labels=list(range(7))).tolist()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    result = []
    for ids, _ in loader:
        logits = model(ids.to(device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite evaluation logits')
        result.extend(logits.argmax(-1).cpu().tolist())
    return result


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def train_one(kind, seed, train_ds, val_ds, val, names, vocab_size, cfg, out, device):
    folder = out / f'{kind}_seed{seed}'
    folder.mkdir()
    seed_all(seed)
    model = Classifier(kind, vocab_size, cfg).to(device)
    # Explicitly share initial embeddings across architectures for each paired seed.
    generator = torch.Generator().manual_seed(seed)
    initial = torch.randn(vocab_size, cfg['embedding_dim'], generator=generator)
    initial[0].zero_()
    with torch.no_grad():
        model.embedding.weight.copy_(initial.to(device))
    counts = np.bincount(train_ds.labels, minlength=7)
    weights = (len(train_ds) / (7 * counts)) ** cfg['class_weight_alpha']
    weights /= weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim < 2 or name.endswith('bias') else decay).append(param)
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': cfg['weight_decay']},
                                   {'params': no_decay, 'weight_decay': 0.0}], lr=cfg['learning_rate'])
    loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], collate_fn=collate)
    best, best_epoch, stale, history = -1.0, 0, 0, []
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    sync(device)
    start = time.perf_counter()
    for epoch in range(1, cfg['max_epochs'] + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum, n = 0.0, 0
        for ids, y in loader:
            ids, y = ids.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(ids), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: {kind}, seed {seed}, epoch {epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * len(y)
            n += len(y)
        pred = predict(model, val_loader, device)
        scores = metrics(val.label_id.tolist(), pred, names)
        sync(device)
        row = {'epoch': epoch, 'train_loss_batch_weighted_mean': loss_sum / n,
               'validation_macro_f1': scores['macro_f1'], 'validation_accuracy': scores['accuracy'],
               'seconds': time.perf_counter() - epoch_start}
        history.append(row)
        print(f'{kind} seed={seed} epoch={epoch} loss={loss_sum/n:.4f} val_F1={scores["macro_f1"]:.4f} seconds={row["seconds"]:.1f}', flush=True)
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
        if scores['macro_f1'] > best:
            best, best_epoch, stale = scores['macro_f1'], epoch, 0
            torch.save({'model_state': model.state_dict(), 'kind': kind, 'seed': seed,
                        'config': cfg, 'vocab_size': vocab_size, 'best_epoch': best_epoch}, folder / 'best.pt')
            dump(folder / 'metrics_validation.json', scores)
            pd.DataFrame({'sample_id': val.sample_id, 'pred_label': [names[p] for p in pred]}).to_csv(
                folder / 'predictions_validation.csv', index=False)
        else:
            stale += 1
        if stale >= cfg['patience']:
            break
    sync(device)
    result = {'model': kind, 'seed': seed, 'best_epoch': best_epoch, 'epochs_run': len(history),
              'validation_macro_f1': best, 'training_seconds_including_validation_and_checkpoints': time.perf_counter() - start,
              'parameters': sum(p.numel() for p in model.parameters()),
              'non_embedding_parameters': sum(p.numel() for n, p in model.named_parameters() if not n.startswith('embedding.')),
              'peak_cuda_allocated_mb': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None}
    dump(folder / 'run_metadata.json', result)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/lstm_xlstm_comparison.json')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'results/lstm_xlstm_comparison')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    out = args.outdir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite experiment: {out}')
    out.mkdir(parents=True, exist_ok=True)
    dump(out / 'locked_config.json', cfg)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    torch.set_num_threads(cfg['num_threads'])
    labels = json.loads((ROOT / 'label_mapping.json').read_text(encoding='utf-8'))['label2id']
    names = [k for k, v in sorted(labels.items(), key=lambda x: x[1])]
    train = read_split(ROOT / cfg['train'], labels)
    val = read_split(ROOT / cfg['validation'], labels)
    match = lambda t: ' '.join(normalize(t).lower().split())
    if set(train.text.map(match)) & set(val.text.map(match)):
        raise ValueError('Train/validation text overlap')
    spm.SentencePieceTrainer.train(sentence_iterator=iter(normalize(t) for t in train.text),
        model_prefix=str(out / 'tokenizer'), vocab_size=cfg['vocab_size'], model_type='unigram',
        character_coverage=1.0, normalization_rule_name='identity', pad_id=0, unk_id=1, bos_id=-1,
        eos_id=2, shuffle_input_sentence=False, num_threads=1, hard_vocab_limit=False, minloglevel=2)
    tokenizer = spm.SentencePieceProcessor(model_file=str(out / 'tokenizer.model'))
    train_ds = TextDataset(train, tokenizer, cfg['max_length'])
    val_ds = TextDataset(val, tokenizer, cfg['max_length'])
    environment = {'started_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
        'platform': platform.platform(), 'device': str(device),
        'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
        'torch_cuda': torch.version.cuda, 'deterministic_algorithms': 'warn_only',
        'determinism_limit': 'CUDA cumsum in official mLSTM is not deterministic in PyTorch 2.5.1; seeded reruns need not be bitwise identical.',
        'precision': 'float32',
        'packages': {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'pandas', 'scikit-learn', 'sentencepiece', 'einops']},
        'xlstm_source_commit': XLSTM_COMMIT,
        'xlstm_source_files_sha256': {str(p.relative_to(XLSTM_SOURCE)): sha(p) for p in sorted((XLSTM_SOURCE / 'xlstm').rglob('*.py'))},
        'hashes': {cfg[k]: sha(ROOT / cfg[k]) for k in ['train', 'validation']},
        'script_sha256': sha(Path(__file__)), 'config_sha256': sha(args.config),
        'tokenizer_sha256': sha(out / 'tokenizer.model'), 'label_mapping_sha256': sha(ROOT / 'label_mapping.json'),
        'tokenization': {'train': train_ds.stats, 'validation': val_ds.stats}, 'actual_vocab_size': tokenizer.get_piece_size(),
        'test_evaluated': False}
    dump(out / 'environment.json', environment)
    runs = []
    for seed in cfg['seeds']:
        for kind in cfg['models']:
            runs.append(train_one(kind, seed, train_ds, val_ds, val, names, tokenizer.get_piece_size(), cfg, out, device))
    dump(out / 'training_complete.json', {'completed_utc': datetime.now(timezone.utc).isoformat(), 'runs': runs})
    if not args.validation_only:
        # Test is first loaded only after every architecture/seed has finished training.
        test = read_split(ROOT / cfg['test'], labels)
        test_ds = TextDataset(test, tokenizer, cfg['max_length'])
        test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], collate_fn=collate)
        environment['hashes'][cfg['test']] = sha(ROOT / cfg['test'])
        environment['tokenization']['test'] = test_ds.stats
        for run in runs:
            folder = out / f'{run["model"]}_seed{run["seed"]}'
            checkpoint = torch.load(folder / 'best.pt', map_location=device, weights_only=True)
            model = Classifier(run['model'], tokenizer.get_piece_size(), cfg).to(device)
            model.load_state_dict(checkpoint['model_state'])
            pred = predict(model, test_loader, device)
            scores = metrics(test.label_id.tolist(), pred, names)
            dump(folder / 'metrics_test.json', scores)
            pd.DataFrame({'sample_id': test.sample_id, 'pred_label': [names[p] for p in pred]}).to_csv(folder / 'predictions_test.csv', index=False)
            run.update({f'test_{k}': scores[k] for k in ['accuracy', 'macro_f1', 'weighted_f1']})
            del model, checkpoint
        environment['test_evaluated'] = True
    dump(out / 'environment.json', environment)
    pd.DataFrame(runs).to_csv(out / 'summary_runs.csv', index=False)
    print(pd.DataFrame(runs).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
