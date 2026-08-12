"""Tests for the Breslow baseline hazard."""

from __future__ import annotations

import pytest
import torch

from kalecancer.survival import BreslowBaselineHazard


@pytest.fixture
def cohort() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    time = torch.rand(64) * 2000 + 50
    event = (torch.rand(64) < 0.4).int()
    return torch.randn(64), event, time


def test_survival_is_monotone_and_within_zero_one(cohort) -> None:
    risk, event, time = cohort
    horizons = torch.tensor([365.0, 1095.0, 1825.0])

    survival = BreslowBaselineHazard().fit(risk, event, time).survival_function(risk, horizons)

    assert survival.shape == (len(risk), 3)
    assert bool(((survival >= 0) & (survival <= 1)).all())
    assert bool((survival[:, 0] >= survival[:, 1]).all() and (survival[:, 1] >= survival[:, 2]).all())


def test_higher_risk_gives_lower_survival(cohort) -> None:
    risk, event, time = cohort
    baseline = BreslowBaselineHazard().fit(risk, event, time)

    survival = baseline.survival_function(torch.tensor([-2.0, 2.0]), torch.tensor([1000.0]))

    assert float(survival[0, 0]) > float(survival[1, 0])


def test_cumulative_hazard_is_non_decreasing(cohort) -> None:
    risk, event, time = cohort
    baseline = BreslowBaselineHazard().fit(risk, event, time)

    hazard = baseline.baseline_cumulative_hazard(torch.tensor([100.0, 500.0, 1000.0, 2000.0]))

    assert bool((hazard.diff() >= 0).all())


def test_hazard_is_zero_before_the_first_event() -> None:
    risk = torch.zeros(3)
    time = torch.tensor([100.0, 200.0, 300.0])
    event = torch.tensor([1, 1, 1])

    baseline = BreslowBaselineHazard().fit(risk, event, time)

    assert float(baseline.baseline_cumulative_hazard(torch.tensor([50.0]))[0]) == 0.0


def test_fitting_requires_observed_events(cohort) -> None:
    risk, _, time = cohort

    with pytest.raises(ValueError, match="without observed events"):
        BreslowBaselineHazard().fit(risk, torch.zeros(len(risk)), time)


def test_must_be_fitted_before_use() -> None:
    with pytest.raises(RuntimeError, match="fit must be called"):
        BreslowBaselineHazard().baseline_cumulative_hazard(torch.tensor([100.0]))
