"""The main torch model: :class:`MultiTaskFusionNet`.

Architecture, top to bottom (see docs/ARCHITECTURE.md for a diagram):

1.  **Categorical embeddings** -- ``sector_key``, ``industry_key``,
    ``security_type`` each get an ``nn.Embedding``; index memberships get a
    mean-pooled embedding (``model/embeddings.py``).
2.  **Branch encoders** -- every numeric table (price, regression,
    regression_indicators, regression_indicators_ma, ta__change,
    ta__ma_ratio, talib, context_sector, context_market,
    context_technical) gets its own small MLP encoder
    (``model/branches.py``). This is a *tabular* multi-branch design rather
    than a raw sequence model, because the source tables already encode
    history as engineered features (e.g. ``ta__change`` is literally a set
    of fixed-horizon lagged returns, and ``regression`` already summarizes
    a whole trend window as slope/R/n) -- re-deriving that from raw price
    sequences would add real complexity for little benefit.
3.  **Fusion** -- branch embeddings are combined with a gated-attention
    layer (each branch gets a learned, per-sample importance weight, so the
    model can down-weight branches that are missing or uninformative for a
    given row) and concatenated with the categorical embeddings, then fed
    through a shared trunk MLP.
4.  **Per-target heads** -- one small MLP head per ``ma_future`` column.
    Each head predicts a Gaussian ``(mu, log_var)`` when
    ``predict_uncertainty`` is enabled, which is what powers the
    "confidence" numbers used throughout ``plotting/`` and
    ``inference/predict.py``; otherwise it predicts a plain point estimate.
"""

from __future__ import annotations

import torch
from torch import nn

from stock_predictor.config import ModelConfig
from stock_predictor.model.branches import BranchEncoder, GatedAttentionFusion
from stock_predictor.model.embeddings import CategoricalEmbeddingGroup, IndexMembershipEmbedding


class PredictionHead(nn.Module):
    """One MLP head per target, predicting either (mu, log_var) or just mu."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, predict_uncertainty: bool) -> None:
        super().__init__()
        self.predict_uncertainty = predict_uncertainty
        out_dim = 2 if predict_uncertainty else 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskFusionNet(nn.Module):
    """Multi-branch tabular network with per-target Gaussian heads.

    Parameters
    ----------
    numeric_branch_dims:
        ``{branch_name: n_columns}`` for every numeric branch, in the exact
        order the dataset will hand over tensors (see
        ``features/dataset.py``).
    categorical_cardinalities:
        ``{column_name: vocab_size}`` for every embedded categorical column.
    index_vocab_size:
        Vocabulary size for the index-membership embedding.
    target_columns:
        Names of the ``ma_future`` columns being predicted, in order.
    """

    def __init__(
        self,
        numeric_branch_dims: dict[str, int],
        categorical_cardinalities: dict[str, int],
        index_vocab_size: int,
        target_columns: list[str],
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.branch_names = list(numeric_branch_dims.keys())
        self.target_columns = list(target_columns)
        self.config = config

        self.categorical_embed = CategoricalEmbeddingGroup(
            categorical_cardinalities, config.categorical_embedding_dim
        )
        self.index_embed = IndexMembershipEmbedding(index_vocab_size, config.categorical_embedding_dim)

        self.branch_encoders = nn.ModuleDict(
            {
                name: BranchEncoder(dim, config.branch_hidden_dims, config.dropout)
                for name, dim in numeric_branch_dims.items()
            }
        )
        branch_output_dim = config.branch_hidden_dims[-1] if config.branch_hidden_dims else 0

        self.use_attention_fusion = config.fusion_attention and len(self.branch_names) > 0
        if self.use_attention_fusion:
            self.fusion_gate = GatedAttentionFusion(branch_output_dim, len(self.branch_names))
            fused_branch_dim = branch_output_dim * len(self.branch_names)
        else:
            self.fusion_gate = None
            fused_branch_dim = branch_output_dim * len(self.branch_names)

        trunk_input_dim = (
            fused_branch_dim + self.categorical_embed.output_dim + self.index_embed.output_dim
        )
        trunk_layers: list[nn.Module] = []
        prev_dim = trunk_input_dim
        for h in config.fusion_hidden_dims:
            trunk_layers += [nn.Linear(prev_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(config.dropout)]
            prev_dim = h
        self.trunk = nn.Sequential(*trunk_layers)
        self.trunk_output_dim = prev_dim

        self.heads = nn.ModuleDict(
            {
                target: PredictionHead(
                    self.trunk_output_dim, config.head_hidden_dim, config.dropout, config.predict_uncertainty
                )
                for target in self.target_columns
            }
        )

    def encode(
        self,
        numeric: dict[str, torch.Tensor],
        categorical: dict[str, torch.Tensor],
        index_membership: torch.Tensor,
    ) -> torch.Tensor:
        """Return the shared trunk representation, before the target heads."""
        branch_embeddings = [self.branch_encoders[name](numeric[name]) for name in self.branch_names]

        if self.use_attention_fusion and branch_embeddings:
            fused = self.fusion_gate(branch_embeddings)
        else:
            fused = torch.cat(branch_embeddings, dim=-1) if branch_embeddings else torch.zeros(
                index_membership.shape[0], 0, device=index_membership.device
            )

        cat_vec = self.categorical_embed(categorical)
        idx_vec = self.index_embed(index_membership)
        trunk_input = torch.cat([fused, cat_vec, idx_vec], dim=-1)
        return self.trunk(trunk_input)

    def forward(
        self,
        numeric: dict[str, torch.Tensor],
        categorical: dict[str, torch.Tensor],
        index_membership: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Returns ``{target_name: (batch, 2 or 1)}`` raw head outputs."""
        shared = self.encode(numeric, categorical, index_membership)
        return {target: self.heads[target](shared) for target in self.target_columns}

    def predict_stacked(
        self,
        numeric: dict[str, torch.Tensor],
        categorical: dict[str, torch.Tensor],
        index_membership: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns ``(mu, log_var)`` stacked over targets, each ``(batch, n_targets)``.

        ``log_var`` is ``None`` when the model was configured without
        uncertainty heads.
        """
        outputs = self.forward(numeric, categorical, index_membership)
        mus, log_vars = [], []
        for target in self.target_columns:
            out = outputs[target]
            mus.append(out[:, 0])
            if self.config.predict_uncertainty:
                log_vars.append(out[:, 1])
        mu = torch.stack(mus, dim=-1)
        log_var = torch.stack(log_vars, dim=-1) if self.config.predict_uncertainty else None
        return mu, log_var
