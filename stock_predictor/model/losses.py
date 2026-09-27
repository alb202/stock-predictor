"""Loss functions for multi-target regression with (a) missing labels
(some symbols simply don't have every ``ma_future`` column populated for
every date) and (b) optional heteroscedastic uncertainty output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_gaussian_nll(
    mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Gaussian negative log-likelihood, averaged over observed entries only.

    ``mu``, ``log_var``, ``target``, ``mask``: all ``(batch, n_targets)``.
    ``log_var`` is clamped to keep training numerically stable early on,
    when the head's weights are still close to their random init.
    """
    log_var = log_var.clamp(min=-10.0, max=10.0)
    precision = torch.exp(-log_var)
    per_element = 0.5 * (log_var + precision * (target - mu) ** 2)
    per_element = per_element * mask
    denom = mask.sum().clamp(min=1.0)
    return per_element.sum() / denom


def masked_mse(mu: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_element = F.mse_loss(mu, target, reduction="none") * mask
    denom = mask.sum().clamp(min=1.0)
    return per_element.sum() / denom


def per_target_masked_mse(mu: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-target MSE, returned as a ``(n_targets,)`` tensor (for logging)."""
    per_element = (mu - target) ** 2 * mask
    denom = mask.sum(dim=0).clamp(min=1.0)
    return per_element.sum(dim=0) / denom
