"""Tests for ``kalecancer.evaluate.survival_metrics``."""

from __future__ import annotations

import numpy as np
import torch

from examples.synthetic_data import make_synthetic_survival
from kalecancer.evaluate.survival_metrics import concordance_index


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


def test_perfect_reversed_and_random_ranking() -> None:
    times = np.arange(1, 101, dtype=np.float64)
    events = np.ones(100, dtype=bool)

    # Highest risk assigned to the earliest event -> every comparable pair concordant.
    perfect_risk = -times
    assert concordance_index(perfect_risk, times, events) == 1.0

    # Highest risk assigned to the latest event -> every comparable pair discordant.
    reversed_risk = times
    assert concordance_index(reversed_risk, times, events) == 0.0

    rng = np.random.default_rng(42)
    n = 5000
    random_times = rng.uniform(1.0, 100.0, size=n)
    random_events = rng.random(n) < 0.7
    random_risk = rng.standard_normal(n)  # uncorrelated with outcome
    c = concordance_index(random_risk, random_times, random_events)
    assert abs(c - 0.5) < 0.05


def test_all_tied_risks_is_exactly_half() -> None:
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    events = np.array([True, False, True, True, False])
    risk = np.zeros(5)  # every pair ties in risk

    assert concordance_index(risk, times, events) == 0.5


def test_agrees_with_on_squared_reference() -> None:
    data = make_synthetic_survival(seed=3)

    reference = _concordance(data.true_risk, data.times, data.events)
    vectorised = concordance_index(data.true_risk.numpy(), data.times.numpy(), data.events.numpy())

    assert abs(vectorised - reference) < 1e-9


def test_censored_never_form_comparable_pair_as_earlier_subject() -> None:
    # Patient 0 is CENSORED at t=1, earlier than both other patients, with a
    # huge risk score. If censoring were (wrongly) ignored when picking the
    # earlier subject of a pair, patient 0 would form two extra "concordant"
    # pairs with patients 1 and 2, inflating C from 0.0 to 2/3. The only
    # legitimate comparable pair is (1, 2): patient 1 (event, t=2) is earlier
    # than patient 2 (t=3), and its lower risk makes that pair discordant.
    times = np.array([1.0, 2.0, 3.0])
    events = np.array([False, True, True])
    risk = np.array([100.0, -1.0, 5.0])

    assert concordance_index(risk, times, events) == 0.0


def test_no_comparable_pairs_raises() -> None:
    import pytest

    times = np.array([1.0, 2.0, 3.0])
    events = np.array([False, False, False])
    risk = np.zeros(3)

    with pytest.raises(ValueError):
        concordance_index(risk, times, events)
