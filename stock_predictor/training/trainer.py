"""Orchestrates walk-forward training.

:meth:`Trainer.run` does three things, in order:

1.  For each walk-forward split from :class:`WalkForwardSplitter`, fits a
    fresh :class:`Preprocessor` on that split's training rows only, trains
    a fresh model with early stopping on that split's validation rows, and
    records per-epoch metrics. This measures how well the whole pipeline
    (feature scaling + architecture + hyperparameters) generalizes across
    different historical market regimes.
2.  Fits one more "final" split covering *all* pre-test history (see
    ``WalkForwardSplitter.final_split``) to produce the production model:
    the one actually used for test-set scoring and for
    ``inference/predict.py``.
3.  Scores the production model once on the held-out test block.

Every setting and every epoch's metrics are collected into a
JSON-serializable dict by :func:`training_output_to_dict`, written out by
``scripts/train.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from stock_predictor.config import Config, config_to_dict
from stock_predictor.data.loader import LoadedDataset
from stock_predictor.features.dataset import StockDataset, collate_samples
from stock_predictor.features.preprocessing import Preprocessor
from stock_predictor.model.architecture import MultiTaskFusionNet
from stock_predictor.model.losses import masked_gaussian_nll, masked_mse
from stock_predictor.splits.time_series_split import Split, WalkForwardSplitter
from stock_predictor.training.metrics import metrics_for_all_targets


def resolve_device(device_setting: str) -> torch.device:
    if device_setting != "auto":
        return torch.device(device_setting)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_metrics: dict[str, dict[str, float]]
    learning_rate: float
    seconds: float


@dataclass
class SplitResult:
    split_index: int
    train_date_range: tuple[str, str]
    val_date_range: tuple[str, str]
    n_train: int
    n_val: int
    epochs: list[EpochMetrics] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    best_val_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class RunResult:
    target_columns: list[str]
    cv_splits: list[SplitResult]
    final_split: SplitResult
    test_metrics: dict[str, dict[str, float]]
    n_test: int
    total_seconds: float
    model: MultiTaskFusionNet
    preprocessor: Preprocessor


def build_model(preprocessor: Preprocessor, model_config) -> MultiTaskFusionNet:
    """Constructs a model whose branch/embedding sizes match a fitted
    preprocessor. Used both by :class:`Trainer` and by ``scripts/predict.py``
    to rebuild the architecture before loading saved weights."""
    numeric_branch_dims = {
        branch: len(cols) for branch, cols in preprocessor.numeric_branches.items()
    }
    return MultiTaskFusionNet(
        numeric_branch_dims=numeric_branch_dims,
        categorical_cardinalities=preprocessor.categorical_cardinalities(),
        index_vocab_size=preprocessor.index_vocab_size(),
        target_columns=preprocessor.target_columns,
        config=model_config,
    )


class Trainer:
    """Runs walk-forward CV, a final production fit, and test scoring."""

    def __init__(self, config: Config, dataset: LoadedDataset) -> None:
        self.config = config
        self.dataset = dataset
        self.device = resolve_device(config.training.device)
        self.splitter = WalkForwardSplitter(config.splitting)
        torch.manual_seed(config.project.random_seed)

    # ------------------------------------------------------------------ #

    def run(self) -> RunResult:
        started = time.time()
        dates = self.dataset.frame["date"]

        cv_splits_raw, test_idx, _held_out_idx = self.splitter.split(dates)
        cv_results = [
            self._run_split(split, max_epochs=self.config.training.max_epochs)
            for split in cv_splits_raw
        ]

        final_split_raw = self.splitter.final_split(dates)
        final_result, final_model, final_preprocessor = self._run_split(
            final_split_raw, max_epochs=self.config.training.max_epochs, return_model=True
        )

        test_metrics, n_test = self._evaluate_on_index(
            final_model, final_preprocessor, test_idx, tag="test"
        )

        return RunResult(
            target_columns=self.dataset.target_columns,
            cv_splits=cv_results,
            final_split=final_result,
            test_metrics=test_metrics,
            n_test=n_test,
            total_seconds=time.time() - started,
            model=final_model,
            preprocessor=final_preprocessor,
        )

    # ------------------------------------------------------------------ #

    def _build_model(self, preprocessor: Preprocessor) -> MultiTaskFusionNet:
        return build_model(preprocessor, self.config.model).to(self.device)

    def _make_loader(
        self, preprocessor: Preprocessor, row_idx: np.ndarray, shuffle: bool
    ) -> TorchDataLoader:
        ds = StockDataset(self.dataset, preprocessor, row_idx)
        return TorchDataLoader(
            ds,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=self.config.training.num_workers,
            collate_fn=collate_samples,
        )

    def _compute_loss(
        self, mu: torch.Tensor, log_var: torch.Tensor | None, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if self.config.training.loss == "gaussian_nll" and log_var is not None:
            return masked_gaussian_nll(mu, log_var, target, mask)
        return masked_mse(mu, target, mask)

    def _run_split(
        self, split: Split, max_epochs: int, return_model: bool = False
    ) -> Any:
        preprocessor = Preprocessor.fit(self.dataset, split.train_idx, self.config.preprocessing)
        model = self._build_model(preprocessor)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.training.lr_scheduler_factor,
            patience=self.config.training.lr_scheduler_patience,
        )

        train_loader = self._make_loader(preprocessor, split.train_idx, shuffle=True)
        val_loader = self._make_loader(preprocessor, split.val_idx, shuffle=False)

        result = SplitResult(
            split_index=split.split_index,
            train_date_range=split.train_date_range,
            val_date_range=split.val_date_range,
            n_train=len(split.train_idx),
            n_val=len(split.val_idx),
        )

        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0

        for epoch in range(max_epochs):
            epoch_start = time.time()
            train_loss = self._train_one_epoch(model, train_loader, optimizer)
            val_loss, val_metrics, _preds = self._evaluate_loader(model, val_loader, preprocessor)
            scheduler.step(val_loss)

            result.epochs.append(
                EpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_metrics=val_metrics,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    seconds=time.time() - epoch_start,
                )
            )

            if val_loss < result.best_val_loss - 1e-6:
                result.best_val_loss = val_loss
                result.best_epoch = epoch
                result.best_val_metrics = val_metrics
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.training.early_stopping_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        if return_model:
            return result, model, preprocessor
        return result

    def _train_one_epoch(self, model: MultiTaskFusionNet, loader: TorchDataLoader, optimizer: torch.optim.Optimizer) -> float:
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            mu, log_var = model.predict_stacked(batch.numeric, batch.categorical, batch.index_membership)
            loss = self._compute_loss(mu, log_var, batch.targets, batch.target_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1
        return total_loss / max(1, total_batches)

    @torch.no_grad()
    def _evaluate_loader(
        self, model: MultiTaskFusionNet, loader: TorchDataLoader, preprocessor: Preprocessor
    ) -> tuple[float, dict[str, dict[str, float]], dict[str, np.ndarray]]:
        model.eval()
        total_loss = 0.0
        total_batches = 0
        all_mu, all_target, all_mask = [], [], []
        for batch in loader:
            batch = batch.to(self.device)
            mu, log_var = model.predict_stacked(batch.numeric, batch.categorical, batch.index_membership)
            loss = self._compute_loss(mu, log_var, batch.targets, batch.target_mask)
            total_loss += float(loss.item())
            total_batches += 1
            all_mu.append(mu.cpu().numpy())
            all_target.append(batch.targets.cpu().numpy())
            all_mask.append(batch.target_mask.cpu().numpy())

        if not all_mu:
            return float("nan"), {}, {}

        mu_arr = np.concatenate(all_mu, axis=0)
        target_arr = np.concatenate(all_target, axis=0)
        mask_arr = np.concatenate(all_mask, axis=0)

        mu_orig = preprocessor.inverse_transform_targets(mu_arr)
        target_orig = preprocessor.inverse_transform_targets(target_arr)
        metrics = metrics_for_all_targets(target_orig, mu_orig, mask_arr, preprocessor.target_columns)

        avg_loss = total_loss / max(1, total_batches)
        return avg_loss, metrics, {"mu": mu_orig, "target": target_orig, "mask": mask_arr}

    def _evaluate_on_index(
        self, model: MultiTaskFusionNet, preprocessor: Preprocessor, row_idx: np.ndarray, tag: str
    ) -> tuple[dict[str, dict[str, float]], int]:
        loader = self._make_loader(preprocessor, row_idx, shuffle=False)
        _loss, metrics, _preds = self._evaluate_loader(model, loader, preprocessor)
        return metrics, len(row_idx)


# -------------------------------------------------------------------------- #
# JSON serialization of a run's metrics/settings
# -------------------------------------------------------------------------- #


def training_output_to_dict(run: RunResult, config: Config) -> dict[str, Any]:
    """Everything about a training run that should end up in the run's
    ``metrics.json`` (per project requirement: settings like epoch count and
    split sizes, plus every metric, for every training run)."""

    def split_to_dict(s: SplitResult) -> dict[str, Any]:
        return {
            "split_index": s.split_index,
            "train_date_range": s.train_date_range,
            "val_date_range": s.val_date_range,
            "n_train": s.n_train,
            "n_val": s.n_val,
            "best_epoch": s.best_epoch,
            "best_val_loss": s.best_val_loss,
            "best_val_metrics": s.best_val_metrics,
            "epochs": [
                {
                    "epoch": e.epoch,
                    "train_loss": e.train_loss,
                    "val_loss": e.val_loss,
                    "val_metrics": e.val_metrics,
                    "learning_rate": e.learning_rate,
                    "seconds": e.seconds,
                }
                for e in s.epochs
            ],
        }

    return {
        "config": config_to_dict(config),
        "target_columns": run.target_columns,
        "cv_splits": [split_to_dict(s) for s in run.cv_splits],
        "final_split": split_to_dict(run.final_split),
        "test_metrics": run.test_metrics,
        "n_test": run.n_test,
        "total_seconds": run.total_seconds,
    }
