"""Plots built from a training run's ``metrics.json`` (see
``training/trainer.py::training_output_to_dict`` for the schema, and
``scripts/plot_training.py`` for the CLI entrypoint).

Every function here takes the already-loaded run dict and an output
directory, and returns the list of PNG paths it wrote, so
``scripts/plot_training.py`` can print a summary and (optionally) publish
them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_METRIC_KEYS = ["mae", "rmse", "r2", "directional_accuracy", "pearson_r"]


def _all_splits(run: dict) -> list[dict]:
    """CV splits plus the final production split, in chronological order."""
    return [*run["cv_splits"], run["final_split"]]


def plot_loss_curves(run: dict, out_dir: Path) -> list[Path]:
    """One figure per split: train loss vs. validation loss over epochs."""
    paths = []
    for split in _all_splits(run):
        epochs = split["epochs"]
        if not epochs:
            continue
        xs = [e["epoch"] for e in epochs]
        train_loss = [e["train_loss"] for e in epochs]
        val_loss = [e["val_loss"] for e in epochs]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, train_loss, label="train loss", color="#3B6FA0")
        ax.plot(xs, val_loss, label="val loss", color="#D97642")
        ax.axvline(split["best_epoch"], color="gray", linestyle="--", alpha=0.6, label="best epoch")
        label = "final" if split["split_index"] == -1 else f"split_{split['split_index']}"
        ax.set_title(f"Loss curve -- {label}\ntrain {split['train_date_range'][0]} to "
                      f"{split['train_date_range'][1]} | val {split['val_date_range'][0]} to "
                      f"{split['val_date_range'][1]}", fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(fontsize=8)
        fig.tight_layout()

        out_path = out_dir / f"loss_curve_{label}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        paths.append(out_path)
    return paths


def plot_target_metric_curves(run: dict, out_dir: Path) -> list[Path]:
    """For each target variable, one figure with MAE/RMSE/R2/directional
    accuracy over epochs for the production ("final") split, with each CV
    split overlaid faintly for comparison."""
    paths = []
    targets = run["target_columns"]
    final_epochs = run["final_split"]["epochs"]
    if not final_epochs:
        return paths

    for target in targets:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, metric_key in zip(axes.flat, ["mae", "rmse", "r2", "directional_accuracy"]):
            for split in run["cv_splits"]:
                xs = [e["epoch"] for e in split["epochs"]]
                ys = [e["val_metrics"].get(target, {}).get(metric_key, np.nan) for e in split["epochs"]]
                ax.plot(xs, ys, color="gray", alpha=0.35, linewidth=1)
            xs = [e["epoch"] for e in final_epochs]
            ys = [e["val_metrics"].get(target, {}).get(metric_key, np.nan) for e in final_epochs]
            ax.plot(xs, ys, color="#3B6FA0", linewidth=2, label="final split")
            ax.set_title(metric_key)
            ax.set_xlabel("epoch")
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"Validation metrics over training -- {target}\n"
                     "(gray = each CV split, blue = production/final split)", fontsize=10)
        fig.tight_layout()

        out_path = out_dir / f"target_metrics_{_safe(target)}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        paths.append(out_path)
    return paths


def plot_split_timeline(run: dict, out_dir: Path) -> Path:
    """A Gantt-style overview of every split's train/val date ranges, plus
    the final held-out test block, so the walk-forward layout is easy to
    sanity-check at a glance."""
    import pandas as pd

    splits = _all_splits(run)
    fig, ax = plt.subplots(figsize=(9, 0.6 * (len(splits) + 1) + 1))
    for row, split in enumerate(splits):
        label = "final" if split["split_index"] == -1 else f"split_{split['split_index']}"
        train_start, train_end = pd.Timestamp(split["train_date_range"][0]), pd.Timestamp(split["train_date_range"][1])
        val_start, val_end = pd.Timestamp(split["val_date_range"][0]), pd.Timestamp(split["val_date_range"][1])
        ax.barh(row, (train_end - train_start).days, left=train_start, color="#3B6FA0", label="train" if row == 0 else None)
        ax.barh(row, (val_end - val_start).days, left=val_start, color="#D97642", label="val" if row == 0 else None)
        ax.text(train_start, row, f" {label}", va="center", ha="right", fontsize=8)

    ax.set_yticks([])
    ax.set_xlabel("date")
    ax.set_title("Walk-forward split layout (train -> embargo gap -> val)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    out_path = out_dir / "split_timeline.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_test_summary(run: dict, out_dir: Path) -> Path:
    """Bar chart of held-out test-set metrics, one group of bars per target."""
    targets = run["target_columns"]
    test_metrics = run["test_metrics"]
    metric_keys = ["mae", "rmse", "r2", "directional_accuracy"]

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(4 * len(metric_keys), 4))
    for ax, metric_key in zip(axes, metric_keys):
        values = [test_metrics.get(t, {}).get(metric_key, np.nan) for t in targets]
        ax.bar(range(len(targets)), values, color="#3B6FA0")
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric_key)
    fig.suptitle(f"Held-out test metrics (n={run['n_test']})", fontsize=11)
    fig.tight_layout()

    out_path = out_dir / "test_metrics_summary.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_all_training_results(run: dict, out_dir: str | Path) -> list[Path]:
    """Runs every plotting function above and returns all written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths += plot_loss_curves(run, out_dir)
    paths += plot_target_metric_curves(run, out_dir)
    paths.append(plot_split_timeline(run, out_dir))
    paths.append(plot_test_summary(run, out_dir))
    return paths


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)
