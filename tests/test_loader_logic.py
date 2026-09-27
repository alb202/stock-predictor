"""Exercises DataLoader's join logic against the synthetic dataset.

This test fakes out :class:`stock_predictor.data.source.DataSource` with a
plain in-memory lookup so it can run in environments without the ``duckdb``
package installed -- it is only the SQL plumbing in ``DataSource`` that
needs duckdb; the join/branch logic in ``DataLoader`` is pure pandas and is
what this test actually cares about.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from stock_predictor.config import load_config
from stock_predictor.data.loader import DataLoader
from stock_predictor.data.synthetic import generate_synthetic_tables


class FakeDataSource:
    """Mimics the subset of DataSource.read_table used by DataLoader."""

    def __init__(self, tables: dict[str, pd.DataFrame]) -> None:
        self.tables = tables

    def read_table(
        self,
        table_name: str,
        columns=None,
        symbols=None,
        date_column: str | None = "date",
        start_date=None,
        end_date=None,
        max_rows=None,
    ) -> pd.DataFrame:
        df = self.tables.get(table_name, pd.DataFrame()).copy()
        if df.empty:
            return df
        if symbols:
            df = df[df["symbol"].isin(symbols)]
        if date_column and date_column in df.columns:
            if start_date:
                df = df[df[date_column] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df[date_column] <= pd.Timestamp(end_date)]
        if columns:
            df = df[columns]
        if max_rows:
            df = df.head(max_rows)
        return df.reset_index(drop=True)


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    cfg = load_config(config_path)

    tables = generate_synthetic_tables(n_symbols=6, start_date="2021-01-01", end_date="2021-12-31", seed=3)
    source = FakeDataSource(tables)

    loader = DataLoader(cfg, source=source)
    loaded = loader.load(include_targets=True)

    frame = loaded.frame
    assert len(frame) > 0, "loaded frame is empty"
    assert "close" in frame.columns
    assert loaded.target_columns, "no target columns resolved"
    for branch, cols in loaded.numeric_branches.items():
        print(f"branch={branch:28s} n_cols={len(cols):3d}")
    print("categorical_columns:", loaded.categorical_columns)
    print("index_membership_column:", loaded.index_membership_column)
    print("target_columns:", loaded.target_columns)
    print("frame shape:", frame.shape)

    # Sanity: sector context columns should actually vary by sector (i.e.
    # the per-row sector -> ETF join picked different ETFs for different
    # sectors), not silently be all-NaN for every row.
    sector_cols = loaded.numeric_branches.get("context_sector", [])
    if sector_cols:
        non_null_frac = frame[sector_cols[0]].notna().mean()
        print(f"context_sector[0] non-null fraction: {non_null_frac:.2f}")
        assert non_null_frac > 0, "sector ETF context join produced no matches"

    market_cols = loaded.numeric_branches.get("context_market", [])
    if market_cols:
        non_null_frac = frame[market_cols[0]].notna().mean()
        print(f"context_market[0] non-null fraction: {non_null_frac:.2f}")
        assert non_null_frac > 0.9, "market ETF context join mostly missing"

    print("\nOK: loader join logic smoke test passed.")


if __name__ == "__main__":
    main()
