"""Per-branch numeric encoders and the gated-attention fusion layer that
combines every branch's embedding into one vector per (symbol, date) row.

Each configured table (price, regression, regression_indicators, ta__change,
context_sector, ...) gets its own small MLP encoder (a "branch"). This
keeps the model modular: adding or removing a table only changes which
branches exist, never the branches' internal code (see
``model/architecture.py`` for where branches are constructed from the
config-resolved column lists).
"""

from __future__ import annotations

import torch
from torch import nn


class BranchEncoder(nn.Module):
    """Standard MLP with LayerNorm + GELU + Dropout, one per numeric branch."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        if input_dim == 0:
            # A branch with no resolved columns (e.g. an optional table that
            # wasn't present in this database) contributes a zero vector
            # rather than being omitted, so the fusion layer's input width
            # stays fixed across runs with slightly different schemas.
            self.net: nn.Module | None = None
            self.output_dim = hidden_dims[-1] if hidden_dims else input_dim
            return

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev_dim = h
        self.net = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            return torch.zeros(x.shape[0], self.output_dim, device=x.device)
        return self.net(x)


class GatedAttentionFusion(nn.Module):
    """Learns a per-branch, per-sample importance weight before concatenation.

    Each branch embedding is scored by a small shared gate network; scores
    are softmax-normalized across branches and used to scale each branch's
    embedding before everything is concatenated into the fusion trunk's
    input. This lets the model down-weight branches that are uninformative
    or frequently missing (e.g. a symbol with no qualifying regression
    trend) on a per-sample basis, instead of relying on a single fixed
    linear combination for every row.
    """

    def __init__(self, branch_dim: int, n_branches: int) -> None:
        super().__init__()
        self.gate = nn.Linear(branch_dim, 1)
        self.n_branches = n_branches

    def forward(self, branch_embeddings: list[torch.Tensor]) -> torch.Tensor:
        # Each branch tensor: (batch, branch_dim) with (possibly) different
        # branch_dim; the gate operates per-branch on its own embedding, so
        # branches are projected to a common width by the caller before this
        # is invoked -- see MultiTaskFusionNet.
        stacked = torch.stack(branch_embeddings, dim=1)          # (batch, n_branches, dim)
        scores = self.gate(stacked).squeeze(-1)                  # (batch, n_branches)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)    # (batch, n_branches, 1)
        weighted = stacked * weights * self.n_branches           # keep overall scale stable
        return weighted.reshape(weighted.shape[0], -1)
