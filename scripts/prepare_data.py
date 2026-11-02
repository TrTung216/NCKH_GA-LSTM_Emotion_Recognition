"""Convert the official workbooks without changing text, labels, order or splits."""
from pathlib import Path
import hashlib
import itertools
import json
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['Anger', 'Disgust', 'Enjoyment', 'Fear', 'Other', 'Sadness', 'Surprise']
FILES = {'train': ('train_nor_811.xlsx', '10VYzfK7JLg-vfmqH0UmKX62z_uaXU-Hp'),
         'validation': ('valid_nor_811.xlsx', '1EsSFZ94fj2yTvFKO6EyxM0wBRcG0s1KE'),
         'test': ('test_nor_811.xlsx', '1D16FCKKgJ0T6t2aSA3biWVwvD9fa4G9a')}

def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def normalize_for_audit(text):
    return ' '.join(unicodedata.normalize('NFC', text).lower().split())

def main():
    for name in ['data', 'audit']:
        (ROOT / name).mkdir(exist_ok=True)
    frames, sources, split_audit = {}, [], {}
    for split, (filename, fid) in FILES.items():
        p = ROOT / 'original' / filename
        raw = pd.read_excel(p)
        assert {'Sentence', 'Emotion'}.issubset(raw.columns)
        assert not raw[['Sentence', 'Emotion']].isna().any().any()
        assert set(raw.Emotion) == set(LABELS)
        assert raw.Sentence.map(lambda x: isinstance(x, str) and bool(x.strip())).all()
        df = pd.DataFrame({
            'sample_id': [f'{split}_{i:05d}' for i in range(len(raw))],
            'source_row': range(2, len(raw) + 2),
            'source_index': raw.iloc[:, 0],
            'text': raw.Sentence,
            'label': raw.Emotion,
            'label_id': raw.Emotion.map({label: i for i, label in enumerate(LABELS)})})
        df.to_csv(ROOT / 'data' / f'{split}.csv', index=False, encoding='utf-8-sig')
        df.to_json(ROOT / 'data' / f'{split}.jsonl', orient='records', lines=True, force_ascii=False)
        check = pd.read_csv(ROOT / 'data' / f'{split}.csv', keep_default_na=False)
        assert check.text.tolist() == raw.Sentence.tolist()
        assert check.label.tolist() == raw.Emotion.tolist()
        lengths = df.text.str.split().str.len()
        split_audit[split] = {
            'rows': len(df), 'missing_text_or_label': 0,
            'duplicate_text_extra_rows': int(df.text.duplicated().sum()),
            'unique_texts': int(df.text.nunique()),
            'source_index_is_unique': bool(raw.iloc[:, 0].is_unique),
            'whitespace_length': {'median': float(lengths.median()),
                'p95': float(lengths.quantile(.95)), 'max': int(lengths.max())}}
        frames[split] = df
        sources.append({'split': split, 'original_filename': filename,
            'source_url': f'https://drive.google.com/file/d/{fid}/view',
            'download_url': f'https://drive.google.com/uc?export=download&id={fid}',
            'size_bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    counts = pd.DataFrame({s: d.label.value_counts().reindex(LABELS) for s, d in frames.items()})
    counts['total'] = counts.sum(axis=1)
    counts.index.name = 'label'
    counts.to_csv(ROOT / 'audit' / 'label_distribution.csv', encoding='utf-8-sig')
    overlap = []
    for a, b in itertools.combinations(frames, 2):
        xa, xb = frames[a], frames[b]
        overlap.append({'split_a': a, 'split_b': b,
            'exact_unique_text_overlap': len(set(xa.text) & set(xb.text)),
            'normalized_unique_text_overlap': len(set(xa.text.map(normalize_for_audit)) & set(xb.text.map(normalize_for_audit)))})
    all_df = pd.concat([d.assign(split=s) for s, d in frames.items()], ignore_index=True)
    duplicates = all_df[all_df.text.duplicated(False)].sort_values(['text', 'split', 'sample_id'])
    duplicates.to_csv(ROOT / 'audit' / 'duplicate_rows.csv', index=False, encoding='utf-8-sig')
    conflicts = all_df.groupby('text').label.nunique()
    conflict_texts = conflicts[conflicts > 1].index
    all_df[all_df.text.isin(conflict_texts)].to_csv(ROOT / 'audit' / 'conflicting_labels.csv', index=False, encoding='utf-8-sig')
    save_json(ROOT / 'audit' / 'quality_report.json', {
        'splits': split_audit, 'cross_split_overlap': overlap,
        'total_rows': len(all_df), 'unique_texts': int(all_df.text.nunique()),
        'texts_with_conflicting_labels': len(conflict_texts),
        'normalization_for_audit_only': 'NFC + lowercase + collapse whitespace',
        'original_splits_preserved': True, 'near_duplicate_detection_performed': False})
    save_json(ROOT / 'label_mapping.json', {
        'note': 'Local package mapping; not an official dataset numeric encoding. Use consistently for every model.',
        'label2id': {label: i for i, label in enumerate(LABELS)},
        'id2label': {str(i): label for i, label in enumerate(LABELS)}})
    save_json(ROOT / 'sources.json', {
        'retrieved_on': '2026-10-04',
        'official_dataset_page': 'https://nlp.uit.edu.vn/datasets',
        'official_folder': 'https://drive.google.com/drive/folders/1HooABJyrddVGzll7fgkJ6VzkG_XuWfRu',
        'paper': 'https://arxiv.org/abs/1911.09339',
        'variant': 'official distributed normalized workbooks, filenames *_nor_811.xlsx',
        'files': sources})
    print(counts.to_string())
    print(json.dumps({'total': len(all_df), 'unique_texts': int(all_df.text.nunique()), 'overlap': overlap}, ensure_ascii=False))

if __name__ == '__main__':
    main()
