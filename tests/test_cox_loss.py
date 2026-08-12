"""Tests for ``kalecancer.survival.cox``."""

from __future__ import annotations

import math

import torch

from kalecancer.survival.cox import neg_partial_log_likelihood


def test_matches_hand_computed_no_ties() -> None:
    # 3 patients, no ties, all events -> Efron reduces to Breslow.
    # Hazard ratios chosen as round numbers: exp(h) = 2, 3, 1.
    times = torch.tensor([3.0, 2.0, 1.0])
    events = torch.tensor([True, True, True])
    log_hazard = torch.tensor([math.log(2.0), math.log(3.0), math.log(1.0)])

    # Risk sets by time (descending: t=3, t=2, t=1), summed in hazard-ratio space:
    #   R(t=3) = {t=3}                 = 2
    #   R(t=2) = {t=3, t=2}            = 2 + 3 = 5
    #   R(t=1) = {t=3, t=2, t=1}       = 2 + 3 + 1 = 6
    # LL = sum_i [h_i - log(R_i)]
    #    = (ln2 - ln2) + (ln3 - ln5) + (ln1 - ln6)
    #    = ln(3 / 5) - ln6 = ln(3 / 30) = ln(0.1) = -ln10
    expected_nll_sum = math.log(10.0)
    expected_nll_mean = expected_nll_sum / 3.0

    nll_sum = neg_partial_log_likelihood(log_hazard, times, events, reduction="sum")
    nll_mean = neg_partial_log_likelihood(log_hazard, times, events, reduction="mean")

    assert torch.isclose(nll_sum, torch.tensor(expected_nll_sum), atol=1e-4)
    assert torch.isclose(nll_mean, torch.tensor(expected_nll_mean), atol=1e-4)


def test_efron_ties() -> None:
    # 4 patients: A at t=3, B and C tied as events at t=2, D at t=1.
    # Hazard ratios chosen as round numbers: exp(a)=2, exp(b)=3, exp(c)=4, exp(d)=1.
    times = torch.tensor([3.0, 2.0, 2.0, 1.0])
    events = torch.tensor([True, True, True, True])
    log_hazard = torch.tensor([math.log(2.0), math.log(3.0), math.log(4.0), math.log(1.0)])

    # Risk sets (time >= t):
    #   R(t=3) = {A}          = 2
    #   R(t=2) = {A, B, C}    = 2 + 3 + 4 = 9
    #   R(t=1) = {A, B, C, D} = 2 + 3 + 4 + 1 = 10
    # Tied death sum at t=2: T = b + c = 3 + 4 = 7, d_k = 2.
    # Efron correction for the tie group, l = 0, 1:
    #   log(R - (0/2)*T) + log(R - (1/2)*T) = log(9) + log(9 - 3.5) = log(9) + log(5.5)
    # Non-tied groups reduce to Breslow (d_k=1):
    #   A: a - log(R(t=3)) = ln2 - ln2 = 0
    #   D: d - log(R(t=1)) = ln1 - ln10 = -ln10
    # LL = 0 + [(b + c) - log(9) - log(5.5)] - ln10
    #    = ln12 - ln9 - ln5.5 - ln10 = ln(12 / (9 * 5.5 * 10)) = ln(12 / 495)
    expected_nll_sum = -math.log(12.0 / 495.0)
    expected_nll_mean = expected_nll_sum / 4.0

    nll_sum = neg_partial_log_likelihood(log_hazard, times, events, reduction="sum")
    nll_mean = neg_partial_log_likelihood(log_hazard, times, events, reduction="mean")

    assert torch.isclose(nll_sum, torch.tensor(expected_nll_sum), atol=1e-4)
    assert torch.isclose(nll_mean, torch.tensor(expected_nll_mean), atol=1e-4)


def test_censored_excluded_from_numerator() -> None:
    # 3 patients, no ties: event at t=4, CENSORED at t=3, event at t=2.
    # The censored patient's hazard ratio is deliberately huge (exp=100) so
    # its effect on a risk-set denominator is unmistakable if included, and
    # its absence from the likelihood's numerator is unmistakable too.
    times = torch.tensor([4.0, 3.0, 2.0])
    events = torch.tensor([True, False, True])
    log_hazard = torch.tensor([math.log(2.0), math.log(100.0), math.log(1.0)])

    # Risk sets (time >= t); the censored patient (t=3) is IN the risk set
    # for t=2 (3 >= 2) but contributes no event term of its own (events[1]=False),
    # and there is no likelihood term at t=3 at all since d_k=0 there:
    #   R(t=4) = {event@4}                      = 2                  (censored excluded: 3 < 4)
    #   R(t=2) = {event@4, censored@3, event@2} = 2 + 100 + 1 = 103
    # LL = [h(4) - log(R(4))] + [h(2) - log(R(2))]
    #    = (ln2 - ln2) + (ln1 - ln103) = -ln103
    # If the censored patient were (wrongly) given its own numerator term, or
    # (wrongly) dropped from R(t=2), this would not match.
    expected_nll_sum = math.log(103.0)

    nll_sum = neg_partial_log_likelihood(log_hazard, times, events, reduction="sum")

    assert torch.isclose(nll_sum, torch.tensor(expected_nll_sum), atol=1e-4)


def test_zero_events_raises() -> None:
    import pytest

    times = torch.tensor([3.0, 2.0, 1.0])
    events = torch.tensor([False, False, False])
    log_hazard = torch.zeros(3)

    with pytest.raises(ValueError):
        neg_partial_log_likelihood(log_hazard, times, events)


def test_reductions_consistent() -> None:
    torch.manual_seed(0)
    times = torch.tensor([5.0, 5.0, 5.0, 4.0, 4.0, 3.0, 2.0, 1.0])
    events = torch.tensor([True, True, False, True, True, True, False, True])
    log_hazard = torch.randn(8)

    per_event = neg_partial_log_likelihood(log_hazard, times, events, reduction="none")
    nll_sum = neg_partial_log_likelihood(log_hazard, times, events, reduction="sum")
    nll_mean = neg_partial_log_likelihood(log_hazard, times, events, reduction="mean")

    assert per_event.shape == (int(events.sum().item()),)
    assert torch.isclose(nll_sum, per_event.sum(), atol=1e-4)
    assert torch.isclose(nll_mean, per_event.sum() / events.sum(), atol=1e-4)


def test_gradient_flows() -> None:
    # Heterogeneous tie sizes (3, 2, 1, 4, 1, 1) regression-test the Efron
    # loop's masking: the log1p argument must be masked *before* the call
    # for inactive (d_k <= l) groups, or nan leaks through torch.where into
    # the gradient even though the forward value is masked out correctly.
    times = torch.tensor([6.0, 6.0, 6.0, 5.0, 5.0, 4.0, 3.0, 3.0, 3.0, 3.0, 2.0, 1.0])
    events = torch.ones(12, dtype=torch.bool)
    torch.manual_seed(0)
    log_hazard = torch.randn(12, requires_grad=True)

    loss = neg_partial_log_likelihood(log_hazard, times, events, reduction="mean")
    loss.backward()

    assert log_hazard.grad is not None
    assert torch.isfinite(log_hazard.grad).all()
    assert log_hazard.grad.abs().sum() > 0.0


def test_permutation_invariance() -> None:
    torch.manual_seed(1)
    times = torch.tensor([6.0, 6.0, 6.0, 5.0, 5.0, 4.0, 3.0, 3.0, 3.0, 3.0, 2.0, 1.0])
    events = torch.tensor([True, True, False, True, True, True, False, True, True, True, True, False])
    log_hazard = torch.randn(12)

    baseline = neg_partial_log_likelihood(log_hazard, times, events, reduction="sum")

    permutation = torch.randperm(12)
    shuffled = neg_partial_log_likelihood(
        log_hazard[permutation], times[permutation], events[permutation], reduction="sum"
    )

    assert torch.isclose(baseline, shuffled, atol=1e-4)
