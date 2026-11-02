"""Create a train-only variant excluding normalized copies of validation text.

Original data/ files and all validation/test files remain byte-identical.
This script never reads test data or validation labels to select removals.
"""
from pathlib import Path
import hashlib
import json
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def key(text):
    return ' '.join(unicodedata.normalize('NFC', text).lower().split())

def main():
    train_path = ROOT / 'data/train.csv'
    validation_path = ROOT / 'data/validation.csv'
    train = pd.read_csv(train_path, keep_default_na=False)
    validation = pd.read_csv(validation_path, usecols=['text'], keep_default_na=False)
    if not train.sample_id.is_unique:
        raise ValueError('Training sample IDs must be unique')
    exclude = train.text.map(key).isin(set(validation.text.map(key)))
    cleaned = train.loc[~exclude].copy()
    out = ROOT / 'data_variants'
    out.mkdir(exist_ok=True)
    cleaned.to_csv(out / 'train_no_validation_overlap.csv', index=False, encoding='utf-8-sig')
    cleaned.to_json(out / 'train_no_validation_overlap.jsonl', orient='records', lines=True, force_ascii=False)
    report = {
        'variant': 'train_no_validation_overlap',
        'rule': 'Remove training rows whose NFC/lowercase/collapsed-whitespace text occurs in validation; no relabeling or within-train deduplication.',
        'normalization_applies_to_matching_only': True,
        'train_rows_before': len(train), 'train_rows_after': len(cleaned),
        'removed_sample_ids': train.loc[exclude, 'sample_id'].tolist(),
        'label_counts_after': cleaned.label.value_counts().sort_index().to_dict(),
        'validation_unchanged': True, 'test_read_by_this_script': False,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [train_path, validation_path]},
    }
    (ROOT / 'audit/training_variant.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
