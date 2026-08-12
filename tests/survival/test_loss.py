"""Tests for the Cox partial-likelihood loss."""

from __future__ import annotations

import pytest
import torch

from kalecancer.survival import as_event_mask, cox_ph_loss, has_risk_set


@pytest.fixture
def cohort() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    time = torch.rand(64) * 2000 + 50
    event = (torch.rand(64) < 0.4).int()
    return torch.randn(64), event, time


def test_event_mask_accepts_the_int_convention() -> None:
    mask = as_event_mask(torch.tensor([1, 0, 1]))

    assert mask.dtype == torch.bool
    assert mask.tolist() == [True, False, True]


def test_event_mask_passes_through_booleans() -> None:
    mask = torch.tensor([True, False])

    assert as_event_mask(mask) is mask


def test_loss_is_finite_and_differentiable(cohort) -> None:
    risk, event, time = cohort
    risk = risk.clone().requires_grad_(True)

    loss = cox_ph_loss(risk, event, time)
    loss.backward()

    assert torch.isfinite(loss)
    assert risk.grad is not None and torch.isfinite(risk.grad).all()


def test_censored_patients_count_in_the_risk_set_until_they_leave() -> None:
    """A censored patient constrains the events that happen before it is censored."""
    event = torch.tensor([1, 0])

    # Censored at t=200, so still at risk when the event occurs at t=100: its risk
    # score belongs in the denominator of the partial likelihood.
    still_at_risk = torch.tensor([100.0, 200.0])
    assert not torch.isclose(
        cox_ph_loss(torch.tensor([1.0, -1.0]), event, still_at_risk),
        cox_ph_loss(torch.tensor([1.0, 3.0]), event, still_at_risk),
    )

    # Censored at t=50, before the event: outside the risk set, so its risk score
    # cannot influence the loss.
    already_gone = torch.tensor([100.0, 50.0])
    assert torch.isclose(
        cox_ph_loss(torch.tensor([1.0, -1.0]), event, already_gone),
        cox_ph_loss(torch.tensor([1.0, 3.0]), event, already_gone),
    )


def test_ranking_events_correctly_lowers_the_loss() -> None:
    """Higher risk for the patient who dies first must be preferred."""
    time = torch.tensor([100.0, 200.0, 300.0])
    event = torch.tensor([1, 1, 1])

    correct = cox_ph_loss(torch.tensor([2.0, 0.0, -2.0]), event, time)
    reversed_ranking = cox_ph_loss(torch.tensor([-2.0, 0.0, 2.0]), event, time)

    assert float(correct) < float(reversed_ranking)


def test_batch_without_events_has_no_risk_set(cohort) -> None:
    risk, _, time = cohort
    censored = torch.zeros(len(risk), dtype=torch.int)

    assert not has_risk_set(censored)
    # torchsurv returns zero here; the trainer skips such batches rather than stepping.
    assert float(cox_ph_loss(risk, censored, time)) == 0.0


def test_has_risk_set_detects_a_single_event() -> None:
    assert has_risk_set(torch.tensor([0, 0, 1, 0]))


def test_ties_methods_both_produce_finite_losses(cohort) -> None:
    risk, event, _ = cohort
    tied_time = torch.full((len(risk),), 100.0)

    for method in ("efron", "breslow"):
        assert torch.isfinite(cox_ph_loss(risk, event, tied_time, ties_method=method))
