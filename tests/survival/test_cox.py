"""Tests for the Cox proportional-hazards head."""

from __future__ import annotations

import torch

from kalecancer.survival import CoxHead


def test_emits_one_risk_score_per_patient() -> None:
    risk = CoxHead(12)(torch.randn(5, 12))

    assert risk.shape == (5,)


def test_has_no_intercept() -> None:
    """The Cox partial likelihood cancels any intercept, so the head must not learn one."""
    head = CoxHead(4)

    assert head.risk.bias is None


def test_risk_is_differentiable() -> None:
    head = CoxHead(4)
    features = torch.randn(3, 4, requires_grad=True)

    head(features).sum().backward()

    assert features.grad is not None
