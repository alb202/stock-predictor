#!/usr/bin/env python3
"""Regenerates prediction plots from an existing run's ``predictions.csv``
without re-running the model. Usage::

    python scripts/plot_predictions.py predictions/predict_20240101_120000/predictions.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from stock_predictor.plotting.prediction_plots import generate_prediction_plots
from stock_predictor.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _infer_target_columns(df: pd.DataFrame) -> list[str]:
    suffix = "__pred"
    return sorted({c[: -len(suffix)] for c in df.columns if c.endswith(suffix)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_csv", help="path to a predictions.csv written by scripts/predict.py")
    parser.add_argument("--out-dir", default=None, help="defaults to <predictions_csv's folder>/plots")
    parser.add_argument("--top-n", type=int, default=20, help="symbols shown in the best-symbols plots")
    args = parser.parse_args()

    setup_logging()
    csv_path = Path(args.predictions_csv)
    df = pd.read_csv(csv_path)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent / "plots"

    targets = _infer_target_columns(df)
    paths = generate_prediction_plots(df, targets, out_dir, top_n=args.top_n)
    logger.info("Wrote %d plots to %s", len(paths), out_dir)


if __name__ == "__main__":
    main()
