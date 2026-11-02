"""Evaluate single-label predictions with strict sample-ID alignment."""
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', type=Path, required=True)
    parser.add_argument('--pred', type=Path, required=True, help='CSV columns: sample_id,pred_label; use exact label names')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    labels = list(json.loads((ROOT / 'label_mapping.json').read_text(encoding='utf-8'))['label2id'])
    gold = pd.read_csv(args.gold, keep_default_na=False)
    pred = pd.read_csv(args.pred, keep_default_na=False)
    if not {'sample_id', 'label'}.issubset(gold) or not {'sample_id', 'pred_label'}.issubset(pred):
        raise ValueError('Expected gold: sample_id,label; predictions: sample_id,pred_label')
    if gold.empty or not gold.sample_id.is_unique or not pred.sample_id.is_unique:
        raise ValueError('Gold must be nonempty; sample IDs must be unique in both files')
    if set(gold.sample_id) != set(pred.sample_id):
        raise ValueError('Prediction IDs must match gold IDs exactly, with no missing/extra rows')
    if not set(gold.label).issubset(labels) or not set(pred.pred_label).issubset(labels):
        raise ValueError(f'Labels must use these exact names: {labels}')
    merged = gold.merge(pred[['sample_id', 'pred_label']], on='sample_id', validate='one_to_one')
    y, yp = merged.label, merged.pred_label
    scores = {'n_samples': len(merged), 'metric_scale': '0 to 1',
        'accuracy': accuracy_score(y, yp),
        'macro_f1': f1_score(y, yp, labels=labels, average='macro', zero_division=0),
        'weighted_f1': f1_score(y, yp, labels=labels, average='weighted', zero_division=0),
        'per_class': classification_report(y, yp, labels=labels, output_dict=True, zero_division=0),
        'confusion_matrix_labels': labels,
        'confusion_matrix_rows_true_columns_predicted': confusion_matrix(y, yp, labels=labels).tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: scores[k] for k in ['n_samples', 'accuracy', 'macro_f1', 'weighted_f1']}, indent=2))

if __name__ == '__main__':
    main()
