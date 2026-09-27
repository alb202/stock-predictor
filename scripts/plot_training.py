#!/usr/bin/env python3
"""Regenerates training plots from an existing run's ``metrics.json``
without retraining. Usage::

    python scripts/plot_training.py artifacts/train_20240101_120000/metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_predictor.plotting.training_plots import plot_all_training_results
from stock_predictor.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", help="path to a metrics.json written by scripts/train.py")
    parser.add_argument("--out-dir", default=None, help="defaults to <metrics_json's folder>/plots")
    args = parser.parse_args()

    setup_logging()
    metrics_path = Path(args.metrics_json)
    run = json.loads(metrics_path.read_text())
    out_dir = Path(args.out_dir) if args.out_dir else metrics_path.parent / "plots"

    paths = plot_all_training_results(run, out_dir)
    logger.info("Wrote %d plots to %s", len(paths), out_dir)


if __name__ == "__main__":
    main()
