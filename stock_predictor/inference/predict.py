"""Runs a trained model over an explicit ``[start_date, end_date]`` window
(project requirement 4) and returns one row per (symbol, date) with each
target's predicted value, predicted uncertainty (sigma) and a derived
confidence score, plus the realized value/error when it's already known
(useful for backtesting a date range that has since resolved).

Saving the result into a timestamped ``predictions/`` folder together with
a copy of the config (project requirement 6) is handled by
``scripts/predict.py``, which calls :func:`predict_date_range`.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import torch

from stock_predictor.config import Config
from stock_predictor.data.loader import DataLoader, LoadedDataset
from stock_predictor.features.dataset import StockDataset, collate_samples
from stock_predictor.features.preprocessing import Preprocessor
from stock_predictor.model.architecture import MultiTaskFusionNet
from stock_predictor.training.trainer import resolve_device


def _load_for_date_range(
    config: Config,
    start_date: str | None,
    end_date: str | None,
    symbols: list[str] | None,
    source=None,
) -> LoadedDataset:
    """Loads a :class:`LoadedDataset` restricted to a date range, without
    touching the config file on disk (the override is applied to an
    in-memory copy of ``config.data_source``)."""
    overridden_source_cfg = dataclasses.replace(
        config.data_source,
        start_date=start_date or config.data_source.start_date,
        end_date=end_date or config.data_source.end_date,
        symbols=symbols if symbols is not None else config.data_source.symbols,
    )
    overridden_config = dataclasses.replace(config, data_source=overridden_source_cfg)
    loader = DataLoader(overridden_config, source=source)
    return loader.load(include_targets=True)


@torch.no_grad()
def predict_date_range(
    config: Config,
    model: MultiTaskFusionNet,
    preprocessor: Preprocessor,
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: list[str] | None = None,
    source=None,
) -> pd.DataFrame:
    """Predict every ``ma_future`` target for every (symbol, date) row in
    ``[start_date, end_date]``.

    Uses the *training-time* ``preprocessor`` to standardize features and
    encode categoricals (never refit here), so predictions are on the same
    scale the model was trained on. Returns a wide DataFrame: one row per
    (symbol, date), with ``{target}__pred``, ``{target}__sigma`` and
    ``{target}__confidence`` columns per target, plus ``{target}__actual``
    / ``{target}__abs_error`` wherever the realized value is already known.
    """
    dataset = _load_for_date_range(config, start_date, end_date, symbols, source=source)
    if len(dataset.frame) == 0:
        return pd.DataFrame()

    device = resolve_device(config.training.device)
    model = model.to(device)
    model.eval()

    row_idx = np.arange(len(dataset.frame))
    torch_dataset = StockDataset(dataset, preprocessor, row_idx)
    loader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_samples,
    )

    all_symbols: list[str] = []
    all_dates: list[str] = []
    all_mu: list[np.ndarray] = []
    all_log_var: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    all_mask: list[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device)
        mu, log_var = model.predict_stacked(batch.numeric, batch.categorical, batch.index_membership)
        all_symbols += batch.symbols
        all_dates += batch.dates
        all_mu.append(mu.cpu().numpy())
        if log_var is not None:
            all_log_var.append(log_var.cpu().numpy())
        all_target.append(batch.targets.cpu().numpy())
        all_mask.append(batch.target_mask.cpu().numpy())

    mu_std = np.concatenate(all_mu, axis=0)
    target_std = np.concatenate(all_target, axis=0)
    mask = np.concatenate(all_mask, axis=0)

    mu_orig = preprocessor.inverse_transform_targets(mu_std)
    target_orig = preprocessor.inverse_transform_targets(target_std)

    if all_log_var:
        log_var_std = np.concatenate(all_log_var, axis=0)
        sigma_std = np.exp(0.5 * np.clip(log_var_std, -10, 10))
        sigma_orig = preprocessor.inverse_transform_target_std(sigma_std)
    else:
        sigma_orig = np.full_like(mu_orig, np.nan)

    result = pd.DataFrame({"symbol": all_symbols, "date": all_dates})
    for j, target in enumerate(preprocessor.target_columns):
        target_scale = max(preprocessor.target_stats[target].std, 1e-8) if target in preprocessor.target_stats else 1.0
        sigma_col = sigma_orig[:, j]
        # A simple, bounded (0, 1] confidence score: 1.0 for a
        # near-certain prediction, shrinking toward 0 as predicted sigma
        # grows relative to the target's overall historical scale.
        confidence = 1.0 / (1.0 + sigma_col / target_scale)

        result[f"{target}__pred"] = mu_orig[:, j]
        result[f"{target}__sigma"] = sigma_col
        result[f"{target}__confidence"] = confidence
        observed = mask[:, j] > 0.5
        result[f"{target}__actual"] = np.where(observed, target_orig[:, j], np.nan)
        result[f"{target}__abs_error"] = np.where(
            observed, np.abs(target_orig[:, j] - mu_orig[:, j]), np.nan
        )

    return result
