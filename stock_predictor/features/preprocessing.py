"""Fits numeric scalers and categorical vocabularies on the training split
only, then applies them consistently to validation/test/inference data.

Fitting only on the training rows of each walk-forward split (never on
val/test/future rows) is what keeps the whole pipeline leakage-free -- see
``training/trainer.py`` for where a fresh :class:`Preprocessor` is fit per
split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_predictor.config import PreprocessingConfig
from stock_predictor.data.loader import LoadedDataset

UNK_TOKEN = "<UNK>"
PAD_INDEX = 0  # reserved for <UNK> / padding in every categorical vocabulary


@dataclass
class ColumnStats:
    mean: float
    std: float


@dataclass
class Vocabulary:
    token_to_index: dict[str, int]

    @classmethod
    def build(cls, values: pd.Series, min_freq: int) -> "Vocabulary":
        counts = values.astype(str).value_counts()
        kept = counts[counts >= min_freq].index.tolist()
        token_to_index = {UNK_TOKEN: PAD_INDEX}
        for token in kept:
            token_to_index[token] = len(token_to_index)
        return cls(token_to_index=token_to_index)

    def encode(self, values: pd.Series) -> np.ndarray:
        return values.astype(str).map(lambda v: self.token_to_index.get(v, PAD_INDEX)).to_numpy(
            dtype=np.int64
        )

    @property
    def size(self) -> int:
        return len(self.token_to_index)


@dataclass
class Preprocessor:
    """Holds every fitted statistic needed to transform raw feature frames."""

    config: PreprocessingConfig
    numeric_stats: dict[str, ColumnStats] = field(default_factory=dict)
    categorical_vocabs: dict[str, Vocabulary] = field(default_factory=dict)
    index_vocab: Vocabulary | None = None
    numeric_branches: dict[str, list[str]] = field(default_factory=dict)
    categorical_columns: list[str] = field(default_factory=list)
    index_membership_column: str | None = None
    target_columns: list[str] = field(default_factory=list)
    target_stats: dict[str, ColumnStats] = field(default_factory=dict)

    # -- fitting ------------------------------------------------------------ #

    @classmethod
    def fit(cls, dataset: LoadedDataset, train_idx: np.ndarray, config: PreprocessingConfig) -> "Preprocessor":
        frame = dataset.frame.iloc[train_idx]
        pre = cls(
            config=config,
            numeric_branches=dataset.numeric_branches,
            categorical_columns=dataset.categorical_columns,
            index_membership_column=dataset.index_membership_column,
            target_columns=dataset.target_columns,
        )

        for col in dataset.all_numeric_columns():
            series = pd.to_numeric(frame[col], errors="coerce")
            mean = float(series.mean()) if series.notna().any() else 0.0
            std = float(series.std()) if series.notna().any() else 1.0
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            pre.numeric_stats[col] = ColumnStats(mean=mean, std=std)

        for col in dataset.categorical_columns:
            pre.categorical_vocabs[col] = Vocabulary.build(frame[col], config.min_category_freq)

        if dataset.index_membership_column:
            exploded = frame[dataset.index_membership_column].explode().dropna()
            if len(exploded) > 0:
                pre.index_vocab = Vocabulary.build(exploded, min_freq=1)
            else:
                pre.index_vocab = Vocabulary(token_to_index={UNK_TOKEN: PAD_INDEX})

        for col in dataset.target_columns:
            series = pd.to_numeric(frame[col], errors="coerce")
            mean = float(series.mean()) if series.notna().any() else 0.0
            std = float(series.std()) if series.notna().any() else 1.0
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            pre.target_stats[col] = ColumnStats(mean=mean, std=std)

        return pre

    # -- transforming --------------------------------------------------- #

    def transform_numeric(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Standardize, clip and impute every numeric branch. Returns one
        ``(n_rows, n_branch_cols)`` float32 array per branch name."""
        out: dict[str, np.ndarray] = {}
        for branch, cols in self.numeric_branches.items():
            if not cols:
                out[branch] = np.zeros((len(frame), 0), dtype=np.float32)
                continue
            arr = np.zeros((len(frame), len(cols)), dtype=np.float32)
            for j, col in enumerate(cols):
                stats = self.numeric_stats.get(col, ColumnStats(mean=0.0, std=1.0))
                values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
                standardized = (values - stats.mean) / stats.std
                standardized = np.clip(standardized, -self.config.clip_std, self.config.clip_std)
                standardized = np.where(
                    np.isfinite(standardized), standardized, self.config.fill_value_after_scaling
                )
                arr[:, j] = standardized
            out[branch] = arr
        return out

    def transform_categorical(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            col: self.categorical_vocabs[col].encode(frame[col])
            for col in self.categorical_columns
            if col in self.categorical_vocabs
        }

    def transform_index_membership(self, frame: pd.DataFrame, max_len: int = 6) -> np.ndarray:
        """Encode each row's list of index memberships as a fixed-width,
        zero-padded array of vocabulary ids (mean-pooled by the model)."""
        if not self.index_membership_column or self.index_vocab is None:
            return np.zeros((len(frame), max_len), dtype=np.int64)
        vocab = self.index_vocab.token_to_index
        out = np.zeros((len(frame), max_len), dtype=np.int64)
        for i, memberships in enumerate(frame[self.index_membership_column]):
            for j, name in enumerate((memberships or [])[:max_len]):
                out[i, j] = vocab.get(str(name), PAD_INDEX)
        return out

    def transform_targets(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (standardized_targets, observed_mask) as float32 arrays,
        both shaped ``(n_rows, n_targets)``. Missing labels are masked out
        of the loss rather than imputed."""
        n = len(frame)
        values = np.zeros((n, len(self.target_columns)), dtype=np.float32)
        mask = np.zeros((n, len(self.target_columns)), dtype=np.float32)
        for j, col in enumerate(self.target_columns):
            if col not in frame.columns:
                continue
            stats = self.target_stats.get(col, ColumnStats(mean=0.0, std=1.0))
            raw = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
            observed = np.isfinite(raw)
            standardized = np.where(observed, (raw - stats.mean) / stats.std, 0.0)
            values[:, j] = standardized
            mask[:, j] = observed.astype(np.float32)
        return values, mask

    def inverse_transform_targets(self, standardized: np.ndarray) -> np.ndarray:
        """Convert standardized target predictions back to the original scale."""
        out = np.zeros_like(standardized)
        for j, col in enumerate(self.target_columns):
            stats = self.target_stats.get(col, ColumnStats(mean=0.0, std=1.0))
            out[:, j] = standardized[:, j] * stats.std + stats.mean
        return out

    def inverse_transform_target_std(self, standardized_sigma: np.ndarray) -> np.ndarray:
        """Convert a predicted standardized-space sigma to the original scale
        (scaling only -- no mean shift is applied to a standard deviation)."""
        out = np.zeros_like(standardized_sigma)
        for j, col in enumerate(self.target_columns):
            stats = self.target_stats.get(col, ColumnStats(mean=0.0, std=1.0))
            out[:, j] = standardized_sigma[:, j] * stats.std
        return out

    # -- cardinalities used to size embedding tables --------------------- #

    def categorical_cardinalities(self) -> dict[str, int]:
        return {col: vocab.size for col, vocab in self.categorical_vocabs.items()}

    def index_vocab_size(self) -> int:
        return self.index_vocab.size if self.index_vocab is not None else 1

    # -- persistence ------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_stats": {k: vars(v) for k, v in self.numeric_stats.items()},
            "categorical_vocabs": {
                k: v.token_to_index for k, v in self.categorical_vocabs.items()
            },
            "index_vocab": self.index_vocab.token_to_index if self.index_vocab else None,
            "numeric_branches": self.numeric_branches,
            "categorical_columns": self.categorical_columns,
            "index_membership_column": self.index_membership_column,
            "target_columns": self.target_columns,
            "target_stats": {k: vars(v) for k, v in self.target_stats.items()},
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path, config: PreprocessingConfig) -> "Preprocessor":
        data = json.loads(Path(path).read_text())
        pre = cls(
            config=config,
            numeric_branches=data["numeric_branches"],
            categorical_columns=data["categorical_columns"],
            index_membership_column=data["index_membership_column"],
            target_columns=data["target_columns"],
        )
        pre.numeric_stats = {k: ColumnStats(**v) for k, v in data["numeric_stats"].items()}
        pre.categorical_vocabs = {
            k: Vocabulary(token_to_index=v) for k, v in data["categorical_vocabs"].items()
        }
        pre.index_vocab = (
            Vocabulary(token_to_index=data["index_vocab"]) if data["index_vocab"] else None
        )
        pre.target_stats = {k: ColumnStats(**v) for k, v in data["target_stats"].items()}
        return pre
