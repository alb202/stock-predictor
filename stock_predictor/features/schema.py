"""Resolves which columns of a loaded table are ids / categorical / numeric.

This is what makes the "add or remove a column via the config file" project
requirement actually work: every branch of the model asks this module for
its column list rather than hard-coding names, and if the config leaves a
table's ``numeric_columns`` empty, :func:`resolve_numeric_columns` falls
back to auto-detecting every numeric, non-id column instead of failing.
"""

from __future__ import annotations

import pandas as pd

from stock_predictor.config import TableConfig

# Columns that are structurally identifiers/keys anywhere in the schema and
# should never be treated as a numeric feature even if auto-detected.
_ALWAYS_EXCLUDE = {"symbol", "date", "start_date", "index_name", "sector_key",
                    "industry_key", "category_key"}


def resolve_numeric_columns(
    df: pd.DataFrame,
    table_cfg: TableConfig,
    fallback_columns: list[str] | None = None,
) -> list[str]:
    """Return the numeric feature columns to use for ``table_cfg``.

    Resolution order: explicit ``numeric_columns`` from config > explicit
    ``fallback_columns`` (used by regression_indicators_ma to default to the
    same names as regression_indicators) > auto-detect every numeric dtype
    column not already claimed as an id/categorical/target column.
    """
    if table_cfg.numeric_columns:
        cols = [c for c in table_cfg.numeric_columns if c in df.columns]
        return cols

    if fallback_columns:
        cols = [c for c in fallback_columns if c in df.columns]
        if cols:
            return cols

    claimed = (
        set(table_cfg.id_columns)
        | set(table_cfg.categorical_columns)
        | set(table_cfg.target_columns)
        | set(table_cfg.exclude_columns)
        | _ALWAYS_EXCLUDE
    )
    numeric_dtype_cols = [
        c for c in df.columns if c not in claimed and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric_dtype_cols


def resolve_categorical_columns(df: pd.DataFrame, table_cfg: TableConfig) -> list[str]:
    return [c for c in table_cfg.categorical_columns if c in df.columns]


def resolve_target_columns(df: pd.DataFrame, table_cfg: TableConfig) -> list[str]:
    """Return the ma_future target columns to use.

    Falls back to "every non-id numeric column" when ``target_columns`` is
    left empty in the config, minus anything listed in ``exclude_columns``.
    """
    if table_cfg.target_columns:
        return [c for c in table_cfg.target_columns if c in df.columns]

    claimed = set(table_cfg.id_columns) | set(table_cfg.exclude_columns) | _ALWAYS_EXCLUDE
    return [
        c for c in df.columns if c not in claimed and pd.api.types.is_numeric_dtype(df[c])
    ]
