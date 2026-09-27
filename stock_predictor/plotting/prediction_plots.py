"""Plots built from a prediction run's wide predictions frame (the output of
``inference/predict.py::predict_date_range``, as saved by
``scripts/predict.py``).

Per project requirements: plot each target's prediction confidence for
every symbol/date (requirement: "plot the confidence of each prediction"),
show every prediction parameter rather than only confidence (pred, sigma,
confidence, and actual/error when known), and make it easy to spot the
symbols with the best predicted output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def plot_confidence_over_time(df: pd.DataFrame, target: str, out_dir: Path) -> Path:
    """Every (symbol, date) prediction for one target, plotted as date (x)
    vs. predicted value (y), colored by confidence -- shows at a glance
    where in time the model is more/less sure of itself."""
    pred_col, conf_col = f"{target}__pred", f"{target}__confidence"
    dates = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        dates, df[pred_col], c=df[conf_col], cmap="viridis", s=14, alpha=0.75, vmin=0, vmax=1
    )
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("date")
    ax.set_ylabel(f"predicted {target}")
    ax.set_title(f"Predicted '{target}' over time, colored by confidence")
    fig.colorbar(scatter, ax=ax, label="confidence")
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = out_dir / f"confidence_over_time_{_safe(target)}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_best_symbols(
    df: pd.DataFrame, target: str, out_dir: Path, as_of_date: str | None = None, top_n: int = 20
) -> Path:
    """The single "where should I look first" plot: for one date (the most
    recent date in the predictions by default), ranks symbols by predicted
    value and shows predicted value, sigma (as an error bar) and confidence
    (as bar color) together -- every prediction parameter in one place, not
    just confidence, per project requirement."""
    pred_col, sigma_col, conf_col = f"{target}__pred", f"{target}__sigma", f"{target}__confidence"

    if as_of_date is None:
        as_of_date = df["date"].max()
    day = df[df["date"] == as_of_date].copy()
    day = day.sort_values(pred_col, ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(day))))
    y_pos = np.arange(len(day))
    colors = plt.cm.viridis(day[conf_col].to_numpy())
    ax.barh(y_pos, day[pred_col], xerr=day[sigma_col], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(day["symbol"])
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel(f"predicted {target} (error bars = predicted sigma)")
    ax.set_title(f"Top {len(day)} symbols by predicted '{target}' -- {as_of_date}\n"
                 "(bar color = confidence: yellow = high, purple = low)")
    fig.tight_layout()

    out_path = out_dir / f"best_symbols_{_safe(target)}_{_safe(str(as_of_date))}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_prediction_detail_grid(df: pd.DataFrame, target: str, out_dir: Path, as_of_date: str | None = None, top_n: int = 20) -> Path:
    """A small-multiples view of every prediction parameter (pred, sigma,
    confidence, and actual-vs-pred error when available) for the same
    ranked symbol list as :func:`plot_best_symbols`, so no parameter is
    hidden behind just the headline confidence number."""
    pred_col, sigma_col, conf_col = f"{target}__pred", f"{target}__sigma", f"{target}__confidence"
    actual_col, error_col = f"{target}__actual", f"{target}__abs_error"

    if as_of_date is None:
        as_of_date = df["date"].max()
    day = df[df["date"] == as_of_date].copy()
    day = day.sort_values(pred_col, ascending=False).head(top_n)
    y_pos = np.arange(len(day))

    has_actual = day[actual_col].notna().any()
    n_panels = 4 if has_actual else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, max(4, 0.35 * len(day))), sharey=True)

    axes[0].barh(y_pos, day[pred_col], color="#3B6FA0")
    axes[0].set_title("predicted value")
    axes[1].barh(y_pos, day[sigma_col], color="#D97642")
    axes[1].set_title("predicted sigma")
    axes[2].barh(y_pos, day[conf_col], color="#4C9A6A")
    axes[2].set_title("confidence")
    axes[2].set_xlim(0, 1)
    if has_actual:
        axes[3].barh(y_pos, day[error_col], color="#A0463B")
        axes[3].set_title("abs. error vs. actual")

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(day["symbol"])
    axes[0].invert_yaxis()
    fig.suptitle(f"Prediction detail for '{target}' -- {as_of_date}", fontsize=12)
    fig.tight_layout()

    out_path = out_dir / f"prediction_detail_{_safe(target)}_{_safe(str(as_of_date))}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_calibration(df: pd.DataFrame, target: str, out_dir: Path) -> Path | None:
    """Predicted vs. actual scatter (only for rows with a realized value),
    colored by confidence -- a well-calibrated model should cluster near
    the y = x line, especially for its highest-confidence predictions."""
    pred_col, actual_col, conf_col = f"{target}__pred", f"{target}__actual", f"{target}__confidence"
    known = df.dropna(subset=[actual_col])
    if known.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        known[actual_col], known[pred_col], c=known[conf_col], cmap="viridis", s=14, alpha=0.75, vmin=0, vmax=1
    )
    lo = min(known[actual_col].min(), known[pred_col].min())
    hi = max(known[actual_col].max(), known[pred_col].max())
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(f"actual {target}")
    ax.set_ylabel(f"predicted {target}")
    ax.set_title(f"Calibration: predicted vs. actual '{target}'")
    fig.colorbar(scatter, ax=ax, label="confidence")
    fig.tight_layout()

    out_path = out_dir / f"calibration_{_safe(target)}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def generate_prediction_plots(
    df: pd.DataFrame, target_columns: list[str], out_dir: str | Path, top_n: int = 20
) -> list[Path]:
    """Runs every plotting function above for every target and returns the
    list of written PNG paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for target in target_columns:
        pred_col = f"{target}__pred"
        if pred_col not in df.columns or df[pred_col].dropna().empty:
            continue
        paths.append(plot_confidence_over_time(df, target, out_dir))
        paths.append(plot_best_symbols(df, target, out_dir, top_n=top_n))
        paths.append(plot_prediction_detail_grid(df, target, out_dir, top_n=top_n))
        calib = plot_calibration(df, target, out_dir)
        if calib is not None:
            paths.append(calib)
    return paths
