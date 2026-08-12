"""Cox proportional-hazards prediction head."""

from __future__ import annotations

import torch
from torch import nn


class CoxHead(nn.Module):
    """Map a patient representation to a log partial hazard.

    The output is the linear predictor of a Cox model: an unbounded score where
    **higher means higher risk** (shorter expected time to event). It has no
    intercept, because the Cox partial likelihood cancels any baseline hazard; use
    :class:`~kalecancer.survival.baseline.BreslowBaselineHazard` to recover absolute
    survival probabilities.

    Args:
        in_features: Dimension of the patient representation.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.risk = nn.Linear(in_features, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict log partial hazards.

        Args:
            x: ``(batch, in_features)`` patient representations.

        Returns:
            ``(batch,)`` log partial hazards.
        """
        return self.risk(x).squeeze(-1)
