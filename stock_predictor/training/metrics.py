"""Per-target regression metrics, computed in the original (unstandardized)
units of each ``ma_future`` column so they're directly interpretable.
"""

from __future__ import annotations

import numpy as np


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    """Metrics for one target column: MAE, RMSE, R^2, Pearson correlation,
    and directional accuracy (how often the predicted sign matches the
    actual sign -- a common sanity check for return-like targets).

    All inputs are 1-D arrays of equal length; ``mask`` is 1.0 where the
    label is observed and 0.0 where it's missing.
    """
    observed = mask > 0.5
    n = int(observed.sum())
    if n == 0:
        return {
            "n": 0, "mae": float("nan"), "rmse": float("nan"),
            "r2": float("nan"), "pearson_r": float("nan"), "directional_accuracy": float("nan"),
        }

    yt = y_true[observed]
    yp = y_pred[observed]
    errors = yt - yp

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    if n > 1 and yt.std() > 1e-12 and yp.std() > 1e-12:
        pearson_r = float(np.corrcoef(yt, yp)[0, 1])
    else:
        pearson_r = float("nan")

    directional_accuracy = float(np.mean(np.sign(yt) == np.sign(yp)))

    return {
        "n": n, "mae": mae, "rmse": rmse, "r2": r2,
        "pearson_r": pearson_r, "directional_accuracy": directional_accuracy,
    }


def metrics_for_all_targets(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, target_columns: list[str]
) -> dict[str, dict[str, float]]:
    """``regression_metrics`` applied independently to each target column."""
    return {
        target: regression_metrics(y_true[:, j], y_pred[:, j], mask[:, j])
        for j, target in enumerate(target_columns)
    }
