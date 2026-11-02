"""Post-hoc descriptive analysis of saved predictions; never trains or tunes."""
import json
from pathlib import Path
import sys
import unicodedata
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.experiment_deps'))
import sentencepiece as spm

OUT = ROOT / 'reports'
OUT.mkdir(exist_ok=True)
FIXED = ROOT / 'results/lstm_xlstm_comparison'
GA = ROOT / 'results/ga_lstm_xlstm_comparison'
SEEDS = [42, 123, 2026]
METHODS = {'LSTM': (FIXED, 'lstm'), 'xLSTM': (FIXED, 'xlstm'),
           'GA+LSTM': (GA / 'final', 'lstm'), 'GA+xLSTM': (GA / 'final', 'xlstm')}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    tokenizer = spm.SentencePieceProcessor(model_file=str(FIXED / 'tokenizer.model'))
    lengths = {}
    for split, filename in [('train', 'data_variants/train_no_validation_overlap.csv'), ('validation', 'data/validation.csv'), ('test', 'data/test.csv')]:
        frame = pd.read_csv(ROOT / filename, keep_default_na=False)
        counts = np.array([len(tokenizer.encode(unicodedata.normalize('NFC', text))) + 1 for text in frame.text])
        lengths[split] = {'n': len(frame), 'median_tokens_including_eos': float(np.median(counts)),
                         'p95_tokens_including_eos': float(np.quantile(counts, .95)),
                         'maximum': int(counts.max()), 'over_128': int((counts > 128).sum())}
    test = pd.read_csv(ROOT / 'data/test.csv', keep_default_na=False)
    n_tokens = np.array([len(tokenizer.encode(unicodedata.normalize('NFC', text))) + 1 for text in test.text])
    labels = list(read(ROOT / 'label_mapping.json')['label2id'])
    evidence = {'lengths': lengths, 'methods': {}, 'length_buckets': [], 'per_seed': [], 'class_delta': {}}
    class_means = {}
    for method, (root, kind) in METHODS.items():
        matrices, class_scores, history_rows, accuracies = [], [], [], []
        for seed in SEEDS:
            folder = root / f'{kind}_seed{seed}'
            scores = read(folder / 'metrics_test.json')
            matrices.append(scores['confusion_matrix_rows_true_columns_predicted'])
            class_scores.append([scores['per_class'][name]['f1-score'] for name in labels])
            h = pd.read_csv(folder / 'history.csv')
            meta = read(folder / 'run_metadata.json')
            selected = h[h.epoch == meta['best_epoch']].iloc[0]
            history_rows.append({'seed': seed, 'best_epoch': int(meta['best_epoch']), 'epochs_run': len(h),
                'loss_at_selected_epoch': float(selected.train_loss_batch_weighted_mean),
                'loss_at_last_epoch': float(h.iloc[-1].train_loss_batch_weighted_mean),
                'best_validation_macro_f1': float(meta['validation_macro_f1']),
                'last_validation_macro_f1': float(h.iloc[-1].validation_macro_f1),
                'mean_epoch_seconds': float(h.seconds.mean())})
            pred = test[['sample_id', 'label']].merge(pd.read_csv(folder / 'predictions_test.csv'), on='sample_id', validate='one_to_one')
            assert len(pred) == len(test) and pred.sample_id.tolist() == test.sample_id.tolist()
            accuracies.append(pred.label.eq(pred.pred_label).to_numpy())
            a = np.array(matrices[-1])[0]
            evidence['per_seed'].append({'method': method, 'seed': seed, 'test_macro_f1': scores['macro_f1'],
                'anger_precision': scores['per_class']['Anger']['precision'], 'anger_recall': scores['per_class']['Anger']['recall'],
                'anger_f1': scores['per_class']['Anger']['f1-score'], 'anger_correct': int(a[0]),
                'anger_predicted_count': int(np.array(matrices[-1])[:, 0].sum())})
        cm = np.mean(matrices, axis=0)
        class_means[method] = np.mean(class_scores, axis=0)
        evidence['methods'][method] = {'mean_confusion_counts': cm.tolist(),
            'mean_row_normalized_confusion_pct': (cm / cm.sum(axis=1, keepdims=True) * 100).tolist(),
            'mean_class_f1': dict(zip(labels, class_means[method])), 'histories': history_rows,
            'mean_epoch_seconds': float(np.mean([r['mean_epoch_seconds'] for r in history_rows]))}
        for name, mask in [('≤16', n_tokens <= 16), ('17–32', (n_tokens > 16) & (n_tokens <= 32)), ('>32', n_tokens > 32)]:
            vals = [a[mask].mean() for a in accuracies]
            evidence['length_buckets'].append({'method': method, 'bucket_tokens_with_eos': name, 'n_samples': int(mask.sum()),
                'accuracy_mean': float(np.mean(vals)), 'accuracy_std': float(np.std(vals, ddof=1))})
    for new, old in [('xLSTM', 'LSTM'), ('GA+LSTM', 'LSTM'), ('GA+xLSTM', 'xLSTM'), ('GA+LSTM', 'GA+xLSTM')]:
        delta = (class_means[new] - class_means[old]) * 100
        evidence['class_delta'][f'{new} minus {old}'] = {'class_f1_delta_pp': dict(zip(labels, delta)),
            'macro_f1_delta_pp': float(delta.mean()), 'class_contribution_to_macro_pp': dict(zip(labels, delta / 7))}
    (OUT / 'analysis_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(evidence['length_buckets']).to_csv(OUT / 'accuracy_by_length.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(evidence['per_seed']).to_csv(OUT / 'anger_diagnostics.csv', index=False, encoding='utf-8-sig')
    print(json.dumps({'lengths': lengths, 'anger_diagnostics': evidence['per_seed'],
          'anger_confusions_pct': {k: v['mean_row_normalized_confusion_pct'][0] for k, v in evidence['methods'].items()},
          'epoch_seconds': {k: v['mean_epoch_seconds'] for k, v in evidence['methods'].items()}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
