import numpy as np
import pytest
from sksurv.linear_model.coxph import BreslowEstimator

from kalecancer.model import BreslowBaseline


@pytest.fixture
def cohort():
    rng = np.random.default_rng(0)
    log_hazard = rng.normal(0, 1, 300)
    # integer times, so many patients tie, censored and event alike
    time = rng.integers(1, 60, 300).astype(np.float64)
    event = rng.random(300) < 0.6
    return log_hazard, time, event


def _naive_cumulative_hazard(log_hazard, time, event):
    """The textbook Breslow loop: at each distinct event time, the events over the risk set's summed hazard."""
    event_times = np.unique(time[event])
    increments = [event[time == t].sum() / np.exp(log_hazard[time >= t]).sum() for t in event_times]
    return event_times, np.cumsum(increments)


def test_matches_the_textbook_loop(cohort):
    baseline = BreslowBaseline().fit(*cohort)
    event_times, cumulative = _naive_cumulative_hazard(*cohort)
    np.testing.assert_array_equal(baseline.event_times_, event_times)
    np.testing.assert_allclose(baseline.cumulative_hazard_, cumulative, rtol=1e-12)


def test_matches_sksurv(cohort):
    log_hazard, time, event = cohort
    baseline = BreslowBaseline().fit(log_hazard, time, event)
    reference = BreslowEstimator().fit(log_hazard, event, time)
    times = np.array([1.0, 1.5, 10.0, 33.3, 58.0, 59.0])
    np.testing.assert_allclose(
        np.r_[0.0, baseline.cumulative_hazard_][np.searchsorted(baseline.event_times_, times, side="right")],
        reference.cum_baseline_hazard_(times),
        rtol=1e-10,
    )
    new = np.array([-2.0, 0.0, 0.7, 3.0])
    expected = np.stack([fn(times) for fn in reference.get_survival_function(new)])
    np.testing.assert_allclose(baseline.survival(new, times), expected, rtol=1e-10)


def test_survival_curves_are_step_functions_from_one(cohort):
    log_hazard, time, event = cohort
    baseline = BreslowBaseline().fit(log_hazard, time, event)
    first = baseline.event_times_[0]
    times = np.r_[0.0, first - 0.5, np.linspace(first, 80.0, 40)]
    survival = baseline.survival([-1.0, 0.0, 2.0], times)
    assert survival.shape == (3, len(times))
    np.testing.assert_array_equal(survival[:, :2], 1.0)
    assert (np.diff(survival, axis=1) <= 0).all() and ((survival > 0) & (survival <= 1)).all()
    # higher risk, lower survival once the baseline hazard has accrued
    assert (survival[2, 2:] < survival[1, 2:]).all() and (survival[1, 2:] < survival[0, 2:]).all()
    # constant after the last event time
    assert survival[0, -1] == baseline.survival([-1.0], [1e9])[0, 0]


def test_a_missing_log_hazard_gives_missing_survival(cohort):
    survival = BreslowBaseline().fit(*cohort).survival([np.nan, 0.0], [5.0, 20.0])
    assert np.isnan(survival[0]).all() and np.isfinite(survival[1]).all()


def test_no_event_means_no_hazard():
    baseline = BreslowBaseline().fit([0.3, -1.0], [4.0, 9.0], np.array([False, False]))
    assert len(baseline.event_times_) == 0
    np.testing.assert_array_equal(baseline.survival([0.0, 5.0], [1.0, 100.0]), 1.0)


def test_large_log_hazards_do_not_overflow(cohort):
    log_hazard, time, event = cohort
    shifted = BreslowBaseline().fit(log_hazard + 800.0, time, event)
    baseline = BreslowBaseline().fit(log_hazard, time, event)
    # the partial likelihood ignores a constant shift, so the baseline absorbs it and survival is unchanged
    np.testing.assert_allclose(
        shifted.survival(log_hazard[:5] + 800.0, [10.0, 40.0]), baseline.survival(log_hazard[:5], [10.0, 40.0])
    )


def test_inputs_are_checked():
    with pytest.raises(RuntimeError, match="not fitted"):
        BreslowBaseline().survival([0.0], [1.0])
    with pytest.raises(ValueError, match="bool event"):
        BreslowBaseline().fit([0.0, 1.0], [1.0, 2.0], np.array([1, 0]))
    with pytest.raises(ValueError, match="share one length"):
        BreslowBaseline().fit([0.0], [1.0, 2.0], np.array([True, False]))
    with pytest.raises(ValueError, match="finite"):
        BreslowBaseline().fit([np.nan, 1.0], [1.0, 2.0], np.array([True, False]))
    assert repr(BreslowBaseline()) == "BreslowBaseline(not fitted)"
