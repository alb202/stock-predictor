#!/usr/bin/env python3
"""Trains the model per ``config.yaml``'s walk-forward + hyperparameter
settings, and writes one timestamped run folder under
``artifacts_dir`` containing:

    config.yaml       -- exact copy of the config used for this run
    metrics.json       -- every setting + every epoch's metrics (all splits)
    model.pt            -- production model weights (state_dict)
    preprocessor.json  -- fitted scalers/vocabularies for the production model
    plots/              -- training curve plots (see plotting/training_plots.py)

Usage
-----
    python scripts/train.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from stock_predictor.config import load_config
from stock_predictor.data.loader import DataLoader
from stock_predictor.plotting.training_plots import plot_all_training_results
from stock_predictor.training.trainer import Trainer, training_output_to_dict
from stock_predictor.utils.io import copy_config, make_timestamped_dir, save_json
from stock_predictor.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-plots", action="store_true", help="skip writing training plots")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    logger.info("Loading and joining tables per config.yaml ...")
    dataset = DataLoader(config).load(include_targets=True)
    logger.info(
        "Loaded %d rows, %d targets: %s",
        len(dataset.frame), len(dataset.target_columns), dataset.target_columns,
    )

    logger.info("Starting walk-forward training (%d CV splits + 1 final fit) ...",
                config.splitting.n_splits)
    trainer = Trainer(config, dataset)
    run = trainer.run()
    logger.info(
        "Training finished in %.1fs. Held-out test metrics: %s",
        run.total_seconds, run.test_metrics,
    )

    run_dir = make_timestamped_dir(config.project.artifacts_dir, prefix="train")
    copy_config(args.config, run_dir)

    metrics_dict = training_output_to_dict(run, config)
    save_json(metrics_dict, run_dir / "metrics.json")
    torch.save(run.model.state_dict(), run_dir / "model.pt")
    run.preprocessor.save(run_dir / "preprocessor.json")

    if not args.no_plots:
        plot_paths = plot_all_training_results(metrics_dict, run_dir / "plots")
        logger.info("Wrote %d training plots to %s", len(plot_paths), run_dir / "plots")

    logger.info("Run artifacts written to: %s", run_dir)


if __name__ == "__main__":
    main()
