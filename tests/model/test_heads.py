"""Tests for ``kalecancer.model.predict.losses.CoxHead``."""

from __future__ import annotations

import torch

from examples.synthetic_data import make_synthetic_survival
from kalecancer.model.predict import CoxHead, neg_partial_log_likelihood


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


def test_output_shape_and_dim_check() -> None:
    import pytest

    model = CoxHead(in_features=8)

    z = torch.randn(5, 8)
    log_hazard = model(z)
    assert log_hazard.shape == (5, 1)

    with pytest.raises(ValueError):
        model(torch.randn(5, 7))  # wrong feature dimension
    with pytest.raises(ValueError):
        model(torch.randn(5))  # wrong number of dimensions


def test_recovers_oracle_ranking() -> None:
    data = make_synthetic_survival(n_samples=512, n_features=16, seed=7)

    torch.manual_seed(0)
    model = CoxHead(in_features=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        optimizer.zero_grad()
        log_hazard = model(data.embeddings)
        loss = neg_partial_log_likelihood(log_hazard, data.times, data.events)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predicted_risk = model(data.embeddings).squeeze(-1)

    trained_c = _concordance(predicted_risk, data.times, data.events)
    oracle_c = _concordance(data.true_risk, data.times, data.events)

    assert trained_c >= 0.9 * oracle_c
    assert trained_c > 0.65


def test_weights_correlate_with_truth() -> None:
    data = make_synthetic_survival(n_samples=512, n_features=16, seed=7)

    torch.manual_seed(0)
    model = CoxHead(in_features=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        optimizer.zero_grad()
        log_hazard = model(data.embeddings)
        loss = neg_partial_log_likelihood(log_hazard, data.times, data.events)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        learned_weights = model.linear.weight.squeeze(0)
        cosine_similarity = torch.dot(learned_weights, data.weights) / (learned_weights.norm() * data.weights.norm())

    assert cosine_similarity.item() > 0.8
