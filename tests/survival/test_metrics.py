"""Tests for censoring-aware survival metrics."""

from __future__ import annotations

import pytest
import torch

from kalecancer.survival import (
    censoring_weights,
    concordance_index,
    survival_metrics,
    usable_eval_times,
)


@pytest.fixture
def cohort() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    time = torch.rand(64) * 2000 + 50
    event = (torch.rand(64) < 0.4).int()
    return torch.randn(64), event, time


def test_concordance_is_one_for_a_perfect_ranking() -> None:
    time = torch.tensor([10.0, 50.0, 100.0, 400.0])
    event = torch.ones(4, dtype=torch.int)

    # Higher risk must mean shorter survival.
    assert concordance_index(-time, event, time) == pytest.approx(1.0)


def test_concordance_is_zero_for_a_reversed_ranking() -> None:
    time = torch.tensor([10.0, 50.0, 100.0, 400.0])
    event = torch.ones(4, dtype=torch.int)

    assert concordance_index(time, event, time) == pytest.approx(0.0)


def test_evaluation_horizons_outside_follow_up_are_dropped() -> None:
    time = torch.tensor([100.0, 500.0, 900.0])

    horizons = usable_eval_times(torch.tensor([50.0, 365.0, 5000.0]), time)

    assert horizons.tolist() == [365.0]


def test_censoring_weights_are_derived_from_the_supplied_cohort(cohort) -> None:
    _, event, time = cohort
    query = torch.tensor([365.0, 1095.0])

    weights = censoring_weights(event, time, new_time=query)

    assert weights.shape == (2,)
    assert bool((weights > 0).all())


def test_reports_harrell_without_training_outcomes(cohort) -> None:
    risk, event, time = cohort

    metrics = survival_metrics(risk, event, time)

    assert set(metrics) == {"c_index", "num_patients", "num_events"}
    assert metrics["num_patients"] == 64.0


def test_adds_censoring_aware_metrics_with_training_outcomes(cohort) -> None:
    risk, event, time = cohort
    train, test = slice(0, 40), slice(40, None)

    metrics = survival_metrics(
        risk[test],
        event[test],
        time[test],
        train_risk=risk[train],
        train_event=event[train],
        train_time=time[train],
        eval_times=torch.tensor([365.0, 1095.0]),
    )

    assert {"c_index", "c_index_ipcw", "auc", "brier", "integrated_brier_score"} <= set(metrics)
    assert 0.0 <= metrics["integrated_brier_score"] <= 1.0
    assert set(metrics["auc"]) == {"365", "1095"}


def test_brier_score_needs_training_risk_scores(cohort) -> None:
    """Without training risks there is no baseline hazard, so no survival curve."""
    risk, event, time = cohort
    train, test = slice(0, 40), slice(40, None)

    metrics = survival_metrics(
        risk[test],
        event[test],
        time[test],
        train_event=event[train],
        train_time=time[train],
        eval_times=torch.tensor([365.0]),
    )

    assert "auc" in metrics
    assert "brier" not in metrics


def test_horizons_beyond_follow_up_are_skipped_entirely(cohort) -> None:
    risk, event, time = cohort
    train, test = slice(0, 40), slice(40, None)

    metrics = survival_metrics(
        risk[test],
        event[test],
        time[test],
        train_risk=risk[train],
        train_event=event[train],
        train_time=time[train],
        eval_times=torch.tensor([99999.0]),
    )

    assert "auc" not in metrics
