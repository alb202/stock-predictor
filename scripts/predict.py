#!/usr/bin/env python3
"""Runs a trained model over an explicit date range and writes a timestamped
run folder under ``predictions_dir`` (project requirement 6) containing:

    config.yaml         -- exact copy of the config used for this run
    predictions.csv       -- one row per (symbol, date), every target's
                              pred / sigma / confidence / actual / abs_error
    summary.json          -- row counts, date range, per-target summary stats
    plots/                -- confidence + detail plots (prediction_plots.py)

Usage
-----
    python scripts/predict.py \\
        --config config/config.yaml \\
        --run-dir artifacts/train_20240101_120000 \\
        --start-date 2024-01-01 --end-date 2024-01-31
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from stock_predictor.config import load_config
from stock_predictor.features.preprocessing import Preprocessor
from stock_predictor.inference.predict import predict_date_range
from stock_predictor.plotting.prediction_plots import generate_prediction_plots
from stock_predictor.training.trainer import build_model, resolve_device
from stock_predictor.utils.io import copy_config, make_timestamped_dir, save_json
from stock_predictor.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run-dir", required=True, help="a train.py output folder (has model.pt + preprocessor.json)")
    parser.add_argument("--start-date", default=None, help="e.g. 2024-01-01; omit for config's default")
    parser.add_argument("--end-date", default=None, help="e.g. 2024-01-31; omit for config's default")
    parser.add_argument("--symbols", nargs="*", default=None, help="restrict to specific symbols")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    run_dir = Path(args.run_dir)

    preprocessor = Preprocessor.load(run_dir / "preprocessor.json", config.preprocessing)
    model = build_model(preprocessor, config.model)
    device = resolve_device(config.training.device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.to(device)

    logger.info(
        "Predicting %s targets for %s to %s ...",
        preprocessor.target_columns, args.start_date, args.end_date,
    )
    predictions = predict_date_range(
        config, model, preprocessor,
        start_date=args.start_date, end_date=args.end_date, symbols=args.symbols,
    )
    logger.info("Produced %d predictions.", len(predictions))

    out_dir = make_timestamped_dir(config.project.predictions_dir, prefix="predict")
    copy_config(args.config, out_dir)
    predictions.to_csv(out_dir / "predictions.csv", index=False)

    summary = {
        "run_dir_used": str(run_dir),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "symbols": args.symbols,
        "n_predictions": len(predictions),
        "n_symbols": predictions["symbol"].nunique() if len(predictions) else 0,
        "n_dates": predictions["date"].nunique() if len(predictions) else 0,
        "target_columns": preprocessor.target_columns,
        "per_target_summary": {
            target: {
                "mean_pred": float(predictions[f"{target}__pred"].mean()),
                "mean_confidence": float(predictions[f"{target}__confidence"].mean()),
                "n_with_actual": int(predictions[f"{target}__actual"].notna().sum()),
            }
            for target in preprocessor.target_columns
            if len(predictions)
        },
    }
    save_json(summary, out_dir / "summary.json")

    if not args.no_plots and len(predictions):
        plot_paths = generate_prediction_plots(predictions, preprocessor.target_columns, out_dir / "plots")
        logger.info("Wrote %d prediction plots to %s", len(plot_paths), out_dir / "plots")

    logger.info("Run artifacts written to: %s", out_dir)


if __name__ == "__main__":
    main()
