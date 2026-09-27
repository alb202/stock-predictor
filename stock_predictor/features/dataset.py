"""A :class:`torch.utils.data.Dataset` over the preprocessed modeling frame.

Each sample is a dict of tensors: one tensor per numeric branch (price,
regression, regression_indicators, ..., context_sector, ...), one tensor
per categorical column, an index-membership tensor, and the target/mask
tensors. Keeping branches separate (rather than concatenating everything
into one flat feature vector before the Dataset) is what lets the model
apply a distinct small encoder per branch -- see
``model/architecture.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from stock_predictor.data.loader import LoadedDataset
from stock_predictor.features.preprocessing import Preprocessor


@dataclass
class SampleBatch:
    """A collated batch, as produced by :func:`collate_samples`."""

    numeric: dict[str, torch.Tensor]       # branch -> (batch, n_branch_cols)
    categorical: dict[str, torch.Tensor]   # column -> (batch,)
    index_membership: torch.Tensor         # (batch, max_index_len)
    targets: torch.Tensor                  # (batch, n_targets)
    target_mask: torch.Tensor              # (batch, n_targets)
    symbols: list[str]
    dates: list[str]

    def to(self, device: torch.device) -> "SampleBatch":
        return SampleBatch(
            numeric={k: v.to(device) for k, v in self.numeric.items()},
            categorical={k: v.to(device) for k, v in self.categorical.items()},
            index_membership=self.index_membership.to(device),
            targets=self.targets.to(device),
            target_mask=self.target_mask.to(device),
            symbols=self.symbols,
            dates=self.dates,
        )


class StockDataset(Dataset):
    """Wraps a preprocessed slice of :class:`LoadedDataset` for a DataLoader.

    Transformation happens once, eagerly, in ``__init__`` (the tables here
    are feature-engineered and modest in width, so this comfortably fits in
    memory for the row counts this project targets -- see README for notes
    on scaling to a much larger universe).
    """

    def __init__(self, dataset: LoadedDataset, preprocessor: Preprocessor, row_idx: np.ndarray) -> None:
        frame = dataset.frame.iloc[row_idx].reset_index(drop=True)
        self.symbols = frame["symbol"].astype(str).tolist()
        self.dates = frame["date"].astype(str).tolist()

        numeric = preprocessor.transform_numeric(frame)
        self.numeric = {k: torch.from_numpy(v) for k, v in numeric.items()}

        categorical = preprocessor.transform_categorical(frame)
        self.categorical = {k: torch.from_numpy(v) for k, v in categorical.items()}

        self.index_membership = torch.from_numpy(preprocessor.transform_index_membership(frame))

        targets, mask = preprocessor.transform_targets(frame)
        self.targets = torch.from_numpy(targets)
        self.target_mask = torch.from_numpy(mask)

        self.n = len(frame)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        return {
            "numeric": {k: v[i] for k, v in self.numeric.items()},
            "categorical": {k: v[i] for k, v in self.categorical.items()},
            "index_membership": self.index_membership[i],
            "targets": self.targets[i],
            "target_mask": self.target_mask[i],
            "symbol": self.symbols[i],
            "date": self.dates[i],
        }


def collate_samples(items: list[dict]) -> SampleBatch:
    numeric_keys = items[0]["numeric"].keys()
    categorical_keys = items[0]["categorical"].keys()
    return SampleBatch(
        numeric={k: torch.stack([it["numeric"][k] for it in items]) for k in numeric_keys},
        categorical={
            k: torch.stack([it["categorical"][k] for it in items]) for k in categorical_keys
        },
        index_membership=torch.stack([it["index_membership"] for it in items]),
        targets=torch.stack([it["targets"] for it in items]),
        target_mask=torch.stack([it["target_mask"] for it in items]),
        symbols=[it["symbol"] for it in items],
        dates=[it["date"] for it in items],
    )
