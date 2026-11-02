"""Independently re-score saved predictions and validate checkpoint selection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from compare_lstm_xlstm import ROOT, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=ROOT / 'results/lstm_xlstm_comparison')
    args = parser.parse_args()
    out = args.outdir.resolve()
    cfg = json.loads((out / 'locked_config.json').read_text(encoding='utf-8'))
    env = json.loads((out / 'environment.json').read_text(encoding='utf-8'))
    for path, digest in env['hashes'].items():
        assert sha(ROOT / path) == digest, f'Data changed: {path}'
    assert sha(out / 'tokenizer.model') == env['tokenizer_sha256']
    assert sha(ROOT / 'scripts/compare_lstm_xlstm.py') == env['script_sha256']
    runs = pd.read_csv(out / 'summary_runs.csv')
    expected = {(model, seed) for model in cfg['models'] for seed in cfg['seeds']}
    assert set(zip(runs.model, runs.seed)) == expected and len(runs) == len(expected)
    audit = out / 'verification'
    audit.mkdir(exist_ok=True)
    checked = []
    for _, run in runs.iterrows():
        folder = out / f'{run.model}_seed{run.seed}'
        history = pd.read_csv(folder / 'history.csv')
        best = history.loc[history.validation_macro_f1.idxmax()]
        assert int(best.epoch) == run.best_epoch
        assert np.isclose(best.validation_macro_f1, run.validation_macro_f1)
        assert len(history) == run.epochs_run
        assert np.isfinite(history.select_dtypes('number').to_numpy()).all()
        for split in ['validation'] + (['test'] if env['test_evaluated'] else []):
            destination = audit / f'{folder.name}_{split}.json'
            subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate.py'),
                            '--gold', str(ROOT / cfg[split]), '--pred', str(folder / f'predictions_{split}.csv'),
                            '--output', str(destination)], check=True, capture_output=True, text=True)
            independent = json.loads(destination.read_text(encoding='utf-8'))
            saved = json.loads((folder / f'metrics_{split}.json').read_text(encoding='utf-8'))
            for key in ['accuracy', 'macro_f1', 'weighted_f1']:
                assert np.isclose(saved[key], independent[key], atol=1e-12, rtol=0), (folder.name, split, key)
            for key in ['n_samples', 'confusion_matrix_labels', 'confusion_matrix_rows_true_columns_predicted']:
                assert saved[key] == independent[key], (folder.name, split, key)
            checked.append(f'{folder.name}/{split}')
    report = {'status': 'passed', 'independently_rescored': checked, 'data_hashes_verified': True,
              'tokenizer_and_training_script_hashes_verified': True, 'best_epoch_selection_verified': True,
              'all_configured_models_and_seeds_present': True}
    (audit / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
