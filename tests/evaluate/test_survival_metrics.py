"""Tests for ``kalecancer.evaluate.survival_metrics``."""

from __future__ import annotations

import numpy as np
import pytest

from kalecancer.evaluate.survival_metrics import (
    integrated_brier,
    kaplan_meier_groups,
    time_dependent_auc,
    usable_eval_times,
)
from kalecancer.survival.baseline import breslow_baseline_hazard, predict_survival_function
from kalecancer.survival.synthetic import make_synthetic_survival

_N_TRAIN = 700


def _train_test_split():
    data = make_synthetic_survival(n_samples=1000, n_features=8, seed=10)
    train_times = data.times[:_N_TRAIN].numpy()
    train_events = data.events[:_N_TRAIN].numpy()
    test_times = data.times[_N_TRAIN:].numpy()
    test_events = data.events[_N_TRAIN:].numpy()
    true_risk_train = data.true_risk[:_N_TRAIN].numpy()
    true_risk_test = data.true_risk[_N_TRAIN:].numpy()
    return data, train_times, train_events, test_times, test_events, true_risk_train, true_risk_test


def _eval_times_within_follow_up(test_times: np.ndarray, test_events: np.ndarray) -> np.ndarray:
    low, high = np.quantile(test_times[test_events], [0.1, 0.9])
    return np.linspace(low, high, 10)


def test_time_dependent_auc_true_risk_beats_random() -> None:
    _, train_times, train_events, test_times, test_events, _, true_risk_test = _train_test_split()
    eval_times = _eval_times_within_follow_up(test_times, test_events)

    rng = np.random.default_rng(0)
    random_risk_test = rng.standard_normal(test_times.shape[0])

    _, mean_auc_true = time_dependent_auc(
        train_times, train_events, test_times, test_events, true_risk_test, eval_times
    )
    _, mean_auc_random = time_dependent_auc(
        train_times, train_events, test_times, test_events, random_risk_test, eval_times
    )

    assert mean_auc_true > mean_auc_random


def test_integrated_brier_true_risk_beats_random() -> None:
    (
        _,
        train_times,
        train_events,
        test_times,
        test_events,
        true_risk_train,
        true_risk_test,
    ) = _train_test_split()
    eval_times = _eval_times_within_follow_up(test_times, test_events)

    rng = np.random.default_rng(0)
    random_risk_test = rng.standard_normal(test_times.shape[0])

    # Same training-set-fitted baseline hazard for both, so the comparison
    # isolates the effect of risk-score quality, not the baseline estimate.
    event_times, cumulative_baseline_hazard = breslow_baseline_hazard(true_risk_train, train_times, train_events)
    survival_probs_true = predict_survival_function(true_risk_test, event_times, cumulative_baseline_hazard, eval_times)
    survival_probs_random = predict_survival_function(
        random_risk_test, event_times, cumulative_baseline_hazard, eval_times
    )

    ibs_true = integrated_brier(train_times, train_events, test_times, test_events, survival_probs_true, eval_times)
    ibs_random = integrated_brier(train_times, train_events, test_times, test_events, survival_probs_random, eval_times)

    assert ibs_true < ibs_random


def test_kaplan_meier_groups_significant_for_true_risk_only() -> None:
    data, _, _, _, _, _, _ = _train_test_split()
    times = data.times.numpy()
    events = data.events.numpy()

    rng = np.random.default_rng(0)
    random_risk = rng.standard_normal(times.shape[0])

    true_result = kaplan_meier_groups(data.true_risk.numpy(), times, events, n_groups=2)
    random_result = kaplan_meier_groups(random_risk, times, events, n_groups=2)

    assert true_result["log_rank_p_value"] < 0.05
    assert random_result["log_rank_p_value"] >= 0.05
    assert set(true_result["km_curves"].keys()) == {0, 1}


def test_eval_times_outside_follow_up_range_raises() -> None:
    _, train_times, train_events, test_times, test_events, _, true_risk_test = _train_test_split()

    out_of_range_eval_times = np.array([test_times.max() + 1.0])

    with pytest.raises(ValueError):
        time_dependent_auc(train_times, train_events, test_times, test_events, true_risk_test, out_of_range_eval_times)


def test_no_evaluation_time_is_usable_without_follow_up() -> None:
    assert usable_eval_times(np.array([365.0, 730.0]), np.array([])).size == 0


def test_risk_groups_that_cannot_be_filled_are_reported() -> None:
    constant_risk = np.zeros(20)
    times = np.linspace(10.0, 200.0, 20)
    events = np.ones(20, dtype=bool)

    with pytest.raises(ValueError, match="non-empty risk groups"):
        kaplan_meier_groups(constant_risk, times, events, n_groups=2)


def test_a_single_risk_group_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        kaplan_meier_groups(np.arange(20.0), np.linspace(10.0, 200.0, 20), np.ones(20, dtype=bool), n_groups=1)
