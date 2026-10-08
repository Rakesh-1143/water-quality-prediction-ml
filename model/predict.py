"""Predict potability for an unlabeled CSV using the saved hybrid model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import pandas as pd
from config import ABSTRACT_ARTIFACT_DIR
from wqi.inference import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ABSTRACT_ARTIFACT_DIR)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Output must differ from the input CSV')
    try:
        frame = pd.read_csv(args.input)
        result = Predictor(args.model).predict_frame(frame)
        result.insert(0, 'row_number', range(1, len(result) + 1))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, float_format='%.17g')
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Prediction failed: {exc}\n')
    print(f'Saved {len(result)} predictions to {args.output}')


if __name__ == '__main__':
    main()
