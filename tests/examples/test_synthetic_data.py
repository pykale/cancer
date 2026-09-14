"""Tests for ``kalecancer.survival.synthetic``."""

from __future__ import annotations

import torch

from examples.synthetic_data import make_synthetic_survival


def _concordance(risk: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> float:
    """Harrell's C computed naively in O(n^2); reference for tests only."""
    concordant, comparable = 0.0, 0.0
    n = times.shape[0]
    for i in range(n):
        if not events[i]:
            continue
        for j in range(n):
            if times[j] > times[i]:
                comparable += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return concordant / comparable


def test_shapes_and_dtypes() -> None:
    data = make_synthetic_survival(n_samples=128, n_features=8, seed=1)
    assert data.embeddings.shape == (128, 8)
    assert data.times.shape == (128,)
    assert data.events.shape == (128,)
    assert data.true_risk.shape == (128,)
    assert data.weights.shape == (8,)
    assert data.events.dtype == torch.bool
    assert torch.all(data.times > 0)


def test_reproducible_with_seed() -> None:
    a = make_synthetic_survival(seed=42)
    b = make_synthetic_survival(seed=42)
    assert torch.equal(a.embeddings, b.embeddings)
    assert torch.equal(a.times, b.times)
    assert torch.equal(a.events, b.events)


def test_censoring_rate_moves_event_fraction() -> None:
    light = make_synthetic_survival(censoring_rate=1.0e-4, seed=0)
    heavy = make_synthetic_survival(censoring_rate=1.0e-2, seed=0)
    assert light.events.float().mean() > heavy.events.float().mean()
    # both regimes must contain events and censored patients
    for data in (light, heavy):
        assert 0.0 < data.events.float().mean() < 1.0


def test_true_risk_is_concordant_with_outcomes() -> None:
    data = make_synthetic_survival(n_samples=512, n_features=16, seed=7)
    c_index = _concordance(data.true_risk, data.times, data.events)
    # the generating risk must rank outcomes clearly better than chance
    assert c_index > 0.7


def test_invalid_arguments_raise() -> None:
    import pytest

    with pytest.raises(ValueError):
        make_synthetic_survival(n_samples=1)
    with pytest.raises(ValueError):
        make_synthetic_survival(n_features=0)
