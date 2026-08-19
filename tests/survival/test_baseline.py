"""Tests for ``kalecancer.survival.baseline``."""

from __future__ import annotations

import numpy as np

from kalecancer.survival.baseline import breslow_baseline_hazard, predict_survival_function
from kalecancer.survival.synthetic import make_synthetic_survival


def _naive_breslow_baseline_hazard(
    log_hazard: np.ndarray, times: np.ndarray, events: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Textbook O(n * k) Breslow loop; reference for tests only."""
    hazard_ratio = np.exp(log_hazard)
    event_times = np.array(sorted(set(times[events].tolist())))
    cumulative = np.empty_like(event_times)
    running_total = 0.0
    for k, t in enumerate(event_times):
        d_t = np.sum(events & (times == t))
        risk_set_sum = hazard_ratio[times >= t].sum()
        running_total += d_t / risk_set_sum
        cumulative[k] = running_total
    return event_times, cumulative


def test_survival_is_monotonic_and_bounded() -> None:
    data = make_synthetic_survival(n_samples=300, n_features=8, seed=1)
    event_times, cumulative_baseline_hazard = breslow_baseline_hazard(
        data.true_risk.numpy(), data.times.numpy(), data.events.numpy()
    )
    eval_times = np.linspace(0.0, float(data.times.max()), 50)

    survival = predict_survival_function(data.true_risk.numpy(), event_times, cumulative_baseline_hazard, eval_times)

    assert np.all(survival >= 0.0)
    assert np.all(survival <= 1.0)
    assert np.all(np.diff(survival, axis=1) <= 1e-9)


def test_higher_risk_gives_lower_survival() -> None:
    data = make_synthetic_survival(n_samples=300, n_features=8, seed=1)
    event_times, cumulative_baseline_hazard = breslow_baseline_hazard(
        data.true_risk.numpy(), data.times.numpy(), data.events.numpy()
    )
    eval_times = np.linspace(0.0, float(data.times.max()), 50)

    low_then_high_risk = np.array([0.0, 5.0])
    survival = predict_survival_function(low_then_high_risk, event_times, cumulative_baseline_hazard, eval_times)

    assert np.all(survival[1] <= survival[0] + 1e-12)
    # after the baseline hazard has actually accrued (H0 > 0), the gap must be strict somewhere
    assert np.any(survival[1] < survival[0] - 1e-9)


def test_matches_naive_breslow_loop() -> None:
    # This is the binding correctness check: our vectorised estimator must
    # agree with a direct, obviously-correct O(n * k) implementation of the
    # textbook Breslow formula.
    data = make_synthetic_survival(n_samples=200, n_features=6, seed=2)
    log_hazard = data.true_risk.numpy()
    times = data.times.numpy()
    events = data.events.numpy()

    our_event_times, our_H0 = breslow_baseline_hazard(log_hazard, times, events)
    naive_event_times, naive_H0 = _naive_breslow_baseline_hazard(log_hazard, times, events)

    assert np.allclose(our_event_times, naive_event_times)
    assert np.allclose(our_H0, naive_H0, atol=1e-8, rtol=1e-6)


def test_recovers_known_baseline_hazard_rate() -> None:
    # Sanity check, not the binding correctness test: the Breslow estimator
    # is a step function fit from a finite sample, and its variance grows
    # as the risk set shrinks in the tail (few patients left => a single
    # event causes a disproportionate jump). We therefore only check
    # correlation with the known H0(t) = baseline_rate * t over the range
    # where at least 20% of the cohort is still at risk.
    baseline_rate = 1.0e-3  # make_synthetic_survival's default
    data = make_synthetic_survival(n_samples=2000, seed=5)
    times = data.times.numpy()
    event_times, cumulative_baseline_hazard = breslow_baseline_hazard(
        data.true_risk.numpy(), times, data.events.numpy()
    )

    at_risk_count = np.array([(times >= t).sum() for t in event_times])
    at_risk_fraction = at_risk_count / times.shape[0]
    keep = at_risk_fraction >= 0.20

    expected = baseline_rate * event_times[keep]
    correlation = np.corrcoef(cumulative_baseline_hazard[keep], expected)[0, 1]

    assert correlation > 0.95


def test_matches_lifelines_baseline_cumulative_hazard() -> None:
    import pytest

    pytest.importorskip("lifelines")
    pd = pytest.importorskip("pandas")
    from lifelines import CoxPHFitter

    rng = np.random.default_rng(0)
    n_samples = 300
    covariates = rng.standard_normal((n_samples, 2))
    true_coef = np.array([0.8, -0.5])
    risk = covariates @ true_coef

    event_times = rng.exponential(scale=np.exp(-risk))
    censor_times = rng.exponential(scale=2.0, size=n_samples)
    times = np.minimum(event_times, censor_times)
    events = event_times <= censor_times

    frame = pd.DataFrame({"time": times, "event": events, "x1": covariates[:, 0], "x2": covariates[:, 1]})
    cph = CoxPHFitter()
    cph.fit(frame, duration_col="time", event_col="event")

    # lifelines centers covariates internally before computing the linear
    # predictor used for its baseline hazard; predict_log_partial_hazard
    # returns exactly that centered predictor.
    log_hazard = cph.predict_log_partial_hazard(frame).to_numpy().astype(np.float64)
    our_event_times, our_H0 = breslow_baseline_hazard(log_hazard, times, events)

    lifelines_H0 = cph.baseline_cumulative_hazard_["baseline cumulative hazard"].loc[our_event_times].to_numpy()

    assert np.allclose(our_H0, lifelines_H0, atol=1e-8, rtol=1e-6)
