"""Run one fixed TF-IDF + LinearSVC pilot on validation, without reading test.

No hyperparameter search. This is a CPU baseline, not an xLSTM/GA experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unicodedata
import warnings
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=ROOT / 'data_variants/train_no_validation_overlap.csv')
    parser.add_argument('--validation', type=Path, default=ROOT / 'data/validation.csv')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'results/baseline_validation')
    args = parser.parse_args()
    if not args.train.exists():
        raise FileNotFoundError('Run scripts/prepare_training_variant.py first')
    train = pd.read_csv(args.train, keep_default_na=False)
    val = pd.read_csv(args.validation, keep_default_na=False)
    labels = list(json.loads((ROOT / 'label_mapping.json').read_text(encoding='utf-8'))['label2id'])
    if set(train.label) != set(labels) or not set(val.label).issubset(labels):
        raise ValueError('Unexpected label inventory')
    if not train.sample_id.is_unique or not val.sample_id.is_unique:
        raise ValueError('Sample IDs must be unique')
    nfc = lambda texts: [unicodedata.normalize('NFC', t) for t in texts]
    model = Pipeline([
        ('features', FeatureUnion([
            ('word', TfidfVectorizer(ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b',
                min_df=2, max_features=50000, sublinear_tf=True, lowercase=True)),
            ('char', TfidfVectorizer(analyzer='char', ngram_range=(3, 5),
                min_df=2, max_features=50000, sublinear_tf=True, lowercase=True)),
        ])),
        ('classifier', LinearSVC(C=1.0, class_weight='balanced', dual=True,
            random_state=42, max_iter=10000, tol=1e-4)),
    ])
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(nfc(train.text), train.label)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    prediction = model.predict(nfc(val.text))
    predict_seconds = time.perf_counter() - start
    args.outdir.mkdir(parents=True, exist_ok=True)
    pred_path = args.outdir / 'predictions_validation.csv'
    pd.DataFrame({'sample_id': val.sample_id, 'pred_label': prediction}).to_csv(pred_path, index=False, encoding='utf-8-sig')
    metric_path = args.outdir / 'metrics_validation.json'
    subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate.py'),
        '--gold', str(args.validation), '--pred', str(pred_path), '--output', str(metric_path)], check=True)
    meta = {
        'experiment': 'fixed_preset_tfidf_word_char_linear_svc_validation_only',
        'train_rows': len(train), 'validation_rows': len(val), 'model_training_seed': 42,
        'tuned_on_validation': False, 'test_predictions_generated': False,
        'fit_seconds_on_this_cpu': fit_seconds, 'predict_seconds_on_this_cpu': predict_seconds,
        'preprocessing': 'NFC, vectorizer lowercasing; preserve diacritics; no stopword list; char ngrams retain punctuation.',
        'tfidf_word_ngrams': [1, 2], 'tfidf_character_ngrams': [3, 5],
        'tfidf_min_df': 2, 'tfidf_max_features_per_branch': 50000, 'tfidf_sublinear_tf': True,
        'classifier_C': 1.0, 'classifier_class_weight': 'balanced', 'classifier_dual': True,
        'classifier_max_iter': 10000, 'classifier_tol': 1e-4,
        'fitted_feature_counts': {name: len(vectorizer.vocabulary_) for name, vectorizer in model.named_steps['features'].transformer_list},
        'classifier_iterations': int(model.named_steps['classifier'].n_iter_),
        'python': platform.python_version(), 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__,
        'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.train, args.validation]},
    }
    (args.outdir / 'run_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'fit_seconds': fit_seconds, 'features': meta['fitted_feature_counts'], 'scikit_learn': sklearn.__version__}, indent=2))

if __name__ == '__main__':
    main()
