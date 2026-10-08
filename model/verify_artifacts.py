"""Independently verify saved predictions, metrics, splits and model reloads."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from config import DATA_PATH, ARTIFACT_DIR, REPORT_DIR, FEATURES
from wqi.core import load_data, metrics
from wqi.inference import Predictor


def verify(output=ARTIFACT_DIR, reports=REPORT_DIR):
    evaluation = json.loads((reports / 'evaluation.json').read_text())
    metadata = json.loads((output / 'metadata.json').read_text())
    predictions = pd.read_csv(reports / 'test_predictions.csv')
    X, y = load_data(DATA_PATH)
    splits = evaluation['split_indices']
    groups = [set(splits[name]) for name in ['train', 'validation', 'test']]
    for name in ['train', 'validation', 'test']:
        if len(set(splits[name])) != len(splits[name]) or len(splits[name]) != evaluation['split_sizes'][name]:
            raise AssertionError('Duplicate row IDs or incorrect partition size')
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError('Partitions overlap')
    if set.union(*groups) != set(X.index):
        raise AssertionError('Partitions do not cover the original dataset')
    if predictions.row_id.tolist() != splits['test']:
        raise AssertionError('Test row IDs differ from the recorded split')
    np.testing.assert_array_equal(predictions.actual, y.loc[predictions.row_id])
    for name, scores in evaluation['test'].items():
        actual = metrics(predictions.actual.to_numpy(), predictions[name].to_numpy())
        for key, value in actual.items():
            # Older float32 CSV exports round at about eight significant digits.
            # Allow float32 roundoff while still checking the stored predictions.
            np.testing.assert_allclose(value, scores[key], rtol=1e-7, atol=1e-7,
                                       err_msg=f'{name}: {key}')
    predictor = Predictor(output)
    if metadata.get('initialization', 'repeat') != evaluation.get('initialization', 'repeat'):
        raise AssertionError('Snapshot initialization methods differ')
    expected_fusion = 'stacking' if evaluation['selected_fusion'] == 'stacking' else 'average'
    if metadata['fusion'] != expected_fusion or metadata['alpha'] != evaluation['alpha']:
        raise AssertionError('Snapshot fusion settings differ')
    for key, value in evaluation['test']['hybrid_selected'].items():
        np.testing.assert_allclose(value, metadata['test_metrics'][key], rtol=1e-7, atol=1e-7)
    restored, _, _ = predictor.predict(X.loc[predictions.row_id])
    np.testing.assert_allclose(restored, predictions.hybrid_selected, atol=1e-6, rtol=1e-6)
    if metadata['dataset_sha256'] != hashlib.sha256(DATA_PATH.read_bytes()).hexdigest():
        raise AssertionError('Dataset hash changed')
    if metadata['features'] != FEATURES:
        raise AssertionError('Features differ')
    if not np.isclose(sum(metadata['initialization_weights'].values()), 1.0):
        raise AssertionError('Initialization weights do not sum to one')
    evidence = {
        'status': 'passed', 'checked_test_rows': len(predictions),
        'checked_models': list(evaluation['test']),
        'split_sizes': evaluation['split_sizes'],
        'initialization': metadata.get('initialization', 'repeat'),
        'dataset_sha256': metadata['dataset_sha256'],
        'max_saved_reload_probability_error': float(np.max(np.abs(
            restored - predictions.hybrid_selected.to_numpy()))),
        'versions': metadata['versions'], 'metric_tolerance': 1e-7,
        'sha256': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(output.iterdir()) if p.is_file()},
        'limitations': ['Verification does not establish equality to the authors results.']}
    (reports / 'verification.json').write_text(json.dumps(evidence, indent=2, allow_nan=False))
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ARTIFACT_DIR)
    parser.add_argument('--reports', type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.reports), indent=2))
