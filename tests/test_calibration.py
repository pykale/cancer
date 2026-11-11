import numpy as np
import pandas as pd
import pytest

from kalecancer.evaluate import CalibrationError, EvalContext, calibration_table


def _ids(n: int) -> list[str]:
    return [f"p{k:04d}" for k in range(n)]


def _exponential_cohort(n: int, seed: int, censor_scale: float, horizon: float = 5.0):
    """A cohort whose true survival at ``horizon`` is known exactly (``exp(-rate * horizon)``), so a prediction
    using that same ``rate`` is perfectly calibrated by construction.
    """
    rng = np.random.default_rng(seed)
    rate = rng.uniform(0.05, 0.2, n)
    event_time = rng.exponential(1.0 / rate)
    censor_time = rng.exponential(censor_scale, n)
    time = np.minimum(event_time, censor_time)
    event = event_time <= censor_time
    ids = _ids(n)
    true_survival = np.exp(-rate * horizon)
    prediction = pd.DataFrame({f"survival[{horizon}]": true_survival}, index=ids)
    target = pd.DataFrame({"time": time, "event": event}, index=ids)
    return prediction, target


def test_perfectly_calibrated_cohort_gives_error_near_zero():
    prediction, target = _exponential_cohort(n=2000, seed=0, censor_scale=1.0 / 0.05)
    context = EvalContext(train_target=target, classes=None)

    error = CalibrationError()(prediction, target, context)

    assert error == pytest.approx(0.0, abs=0.03)


def test_systematically_over_optimistic_model_gives_error_near_the_shift():
    prediction, target = _exponential_cohort(n=2000, seed=0, censor_scale=1.0 / 0.05)
    column = prediction.columns[0]
    over_optimistic = prediction.assign(**{column: (prediction[column] + 0.2).clip(upper=1.0)})
    context = EvalContext(train_target=target, classes=None)

    error = CalibrationError()(over_optimistic, target, context)

    assert error == pytest.approx(0.2, abs=0.03)


def test_heavily_censored_cohort_is_close_to_the_lightly_censored_one():
    """``observed`` must be a Kaplan-Meier estimate, not a naive event rate: an event rate would be systematically
    biased by heavy censoring (it would misread "censored before the horizon" as "did not survive"), making a
    perfectly calibrated model look badly calibrated for reasons that have nothing to do with the model. A KM
    estimate stays close -- the only difference left is ordinary sampling noise from fewer effective events.
    """
    light_prediction, light_target = _exponential_cohort(n=3000, seed=3, censor_scale=1.0 / 0.02)
    heavy_prediction, heavy_target = _exponential_cohort(n=3000, seed=3, censor_scale=1.0 / 0.19)
    assert 1 - light_target["event"].mean() < 0.3  # light censoring, for contrast
    assert 1 - heavy_target["event"].mean() > 0.6  # ~70% censoring

    light_context = EvalContext(train_target=light_target, classes=None)
    heavy_context = EvalContext(train_target=heavy_target, classes=None)
    light_error = CalibrationError()(light_prediction, light_target, light_context)
    heavy_error = CalibrationError()(heavy_prediction, heavy_target, heavy_context)

    assert light_error < 0.05
    assert heavy_error < 0.05
    assert abs(heavy_error - light_error) < 0.05


def test_table_has_one_row_per_time_bin_and_bin_sizes_sum_to_the_cohort():
    n = 60
    rng = np.random.default_rng(2)
    ids = _ids(n)
    prediction = pd.DataFrame(
        {"survival[3.0]": rng.uniform(0.1, 0.9, n), "survival[6.0]": rng.uniform(0.05, 0.8, n)}, index=ids
    )
    target = pd.DataFrame({"time": rng.exponential(5.0, n) + 0.5, "event": rng.random(n) < 0.7}, index=ids)

    table = calibration_table(prediction, target, n_bins=5)

    assert len(table) == 2 * 5  # 2 horizons x 5 bins
    for _, group in table.groupby("time"):
        assert group["n"].sum() == n
        assert sorted(group["bin"].tolist()) == list(range(5))


def test_non_unique_quantile_cuts_raise_naming_the_horizon():
    ids = _ids(20)
    prediction = pd.DataFrame({"survival[5.0]": np.full(20, 0.5)}, index=ids)
    target = pd.DataFrame({"time": np.arange(1.0, 21.0), "event": np.ones(20, dtype=bool)}, index=ids)

    with pytest.raises(ValueError, match=r"calibration_table at time=5\.0"):
        calibration_table(prediction, target, n_bins=2)
    with pytest.raises(ValueError, match="quantile cuts are not unique"):
        calibration_table(prediction, target, n_bins=2)


def test_undefined_bin_is_nan_and_ignored_by_the_metric_rather_than_making_it_nan():
    ids = _ids(10)
    predicted = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.7, 0.75, 0.8, 0.85, 0.9])
    horizon = 10.0
    prediction = pd.DataFrame({f"survival[{horizon}]": predicted}, index=ids)
    # Bin 0 (low predicted survival): all dead by time 5, well before the horizon -- KM reaches 0, well-defined.
    # Bin 1 (high predicted survival): all censored at time 3, with no event at all -- nobody is still at risk by
    # the horizon, and survival never reached 0, so it is genuinely undefined there.
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    event = np.array([True, True, True, True, True, False, False, False, False, False])
    target = pd.DataFrame({"time": time, "event": event}, index=ids)

    table = calibration_table(prediction, target, n_bins=2)

    bin0 = table[table["bin"] == 0].iloc[0]
    bin1 = table[table["bin"] == 1].iloc[0]
    assert bin0["observed"] == pytest.approx(0.0)
    assert np.isnan(bin1["observed"])

    context = EvalContext(train_target=target, classes=None)
    error = CalibrationError(n_bins=2)(prediction, target, context)

    assert np.isfinite(error)
    assert error == pytest.approx(abs(0.0 - predicted[:5].mean()))
