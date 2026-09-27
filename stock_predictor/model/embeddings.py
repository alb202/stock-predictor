"""Embedding layers for categorical instrument metadata.

Two kinds of categorical information are embedded:

* Single-valued columns (``sector_key``, ``industry_key``, ``security_type``)
  each get their own :class:`torch.nn.Embedding` table.
* Index membership (a stock can belong to several indices at once) is
  embedded per index and mean-pooled over the (padded) memberships of each
  row, so a symbol in two indices gets the average of both embeddings.
"""

from __future__ import annotations

import torch
from torch import nn


class CategoricalEmbeddingGroup(nn.Module):
    """One embedding table per categorical column, concatenated on output."""

    def __init__(self, cardinalities: dict[str, int], embedding_dim: int) -> None:
        super().__init__()
        self.columns = list(cardinalities.keys())
        self.embeddings = nn.ModuleDict(
            {col: nn.Embedding(card, embedding_dim, padding_idx=0) for col, card in cardinalities.items()}
        )
        self.output_dim = embedding_dim * len(cardinalities)

    def forward(self, categorical: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.columns:
            batch_size = next(iter(categorical.values())).shape[0] if categorical else 0
            device = next(iter(categorical.values())).device if categorical else "cpu"
            return torch.zeros(batch_size, 0, device=device)
        parts = [self.embeddings[col](categorical[col]) for col in self.columns]
        return torch.cat(parts, dim=-1)


class IndexMembershipEmbedding(nn.Module):
    """Mean-pooled embedding over a padded list of index memberships.

    Padding uses index 0 (reserved for ``<UNK>``/padding in the vocabulary
    built by ``features/preprocessing.py``); padded slots are excluded from
    the mean whenever at least one real membership is present.
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.output_dim = embedding_dim

    def forward(self, index_ids: torch.Tensor) -> torch.Tensor:
        # index_ids: (batch, max_len), 0 = padding/UNK
        mask = (index_ids != 0).float().unsqueeze(-1)          # (batch, max_len, 1)
        embedded = self.embedding(index_ids)                    # (batch, max_len, dim)
        summed = (embedded * mask).sum(dim=1)                   # (batch, dim)
        counts = mask.sum(dim=1).clamp(min=1.0)                 # (batch, 1)
        return summed / counts
