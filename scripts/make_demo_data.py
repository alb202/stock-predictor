#!/usr/bin/env python3
"""Writes a synthetic demo dataset to disk so the rest of the pipeline can
be run end-to-end without a real market database.

Usage
-----
    python scripts/make_demo_data.py --out-dir data/demo_csv --format csv

Then point ``config/config.yaml``'s ``data_source`` at it, e.g.::

    data_source:
      type: csv_dir
      path: data/demo_csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stock_predictor.data.synthetic import generate_synthetic_tables, write_tables
from stock_predictor.utils.logging_utils import setup_logging

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/demo_csv")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--n-symbols", type=int, default=40)
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup_logging()
    logger.info(
        "Generating synthetic data: %d symbols, %s to %s ...",
        args.n_symbols, args.start_date, args.end_date,
    )
    tables = generate_synthetic_tables(
        n_symbols=args.n_symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        seed=args.seed,
    )
    write_tables(tables, args.out_dir, fmt=args.format)

    for name, df in tables.items():
        logger.info("  %-26s %8d rows  %2d cols", name, len(df), len(df.columns))

    logger.info("Wrote demo tables to %s (%s format).", args.out_dir, args.format)
    logger.info(
        "Point config.yaml's data_source at it:\n"
        "  data_source:\n"
        "    type: %s_dir\n"
        "    path: %s",
        args.format, args.out_dir,
    )


if __name__ == "__main__":
    main()
