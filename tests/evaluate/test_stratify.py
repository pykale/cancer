import numpy as np
import pandas as pd
import pytest
from sksurv.nonparametric import kaplan_meier_estimator

from kalecancer.evaluate import kaplan_meier, risk_groups


def _ids(n: int, prefix: str = "p") -> list[str]:
    return [f"{prefix}{k:04d}" for k in range(n)]


def test_median_split_puts_half_in_each_group():
    n = 100
    rng = np.random.default_rng(3)
    prediction = pd.DataFrame({"log_hazard": rng.normal(size=n)}, index=_ids(n))

    group = risk_groups(prediction, n_groups=2)

    counts = group.value_counts()
    assert counts[0] == 50
    assert counts[1] == 50


@pytest.mark.parametrize("n_groups", [3, 4])
def test_three_and_four_groups_split_evenly(n_groups):
    n = 360  # divisible by both 3 and 4, so each group is exactly equal-sized
    rng = np.random.default_rng(7)
    prediction = pd.DataFrame({"log_hazard": rng.normal(size=n)}, index=_ids(n))

    group = risk_groups(prediction, n_groups=n_groups)

    counts = group.value_counts()
    assert len(counts) == n_groups
    assert counts.min() == counts.max() == n // n_groups


def test_labels_name_the_groups_lowest_risk_first():
    n = 40
    rng = np.random.default_rng(8)
    prediction = pd.DataFrame({"log_hazard": rng.normal(size=n)}, index=_ids(n))

    group = risk_groups(prediction, n_groups=2, labels=["low", "high"])

    assert group.cat.categories.tolist() == ["low", "high"]
    assert group[prediction["log_hazard"].idxmin()] == "low"
    assert group[prediction["log_hazard"].idxmax()] == "high"


def test_known_cohort_produces_hand_checkable_km_steps():
    """8 patients, 2 groups of 4 by construction; each group's steps are computed by hand.

    A leading time=0, survival=1.0 row is always prepended (the conventional KM starting point), since neither
    group has an observation at time=0 itself.

    Group 0 (lowest log_hazard): times 1, 2, 2, 4; censored, event, censored, event.
        t=1: at_risk=4, events=0 -> survival=1.0
        t=2: at_risk=3, events=1 -> survival=1.0 * (1 - 1/3) = 0.6667
        t=4: at_risk=1, events=1 -> survival=0.6667 * (1 - 1/1) = 0.0
    Group 1 (highest log_hazard): times 1, 3, 5, 7; censored, event, event, event.
        t=1: at_risk=4, events=0 -> survival=1.0
        t=3: at_risk=3, events=1 -> survival=1.0 * (1 - 1/3) = 0.6667
        t=5: at_risk=2, events=1 -> survival=0.6667 * (1 - 1/2) = 0.3333
        t=7: at_risk=1, events=1 -> survival=0.3333 * (1 - 1/1) = 0.0
    """
    ids = _ids(8)
    log_hazard = np.array([-2.0, -1.9, -1.8, -1.7, 1.7, 1.8, 1.9, 2.0])
    prediction = pd.DataFrame({"log_hazard": log_hazard}, index=ids)
    time = np.array([1.0, 2.0, 2.0, 4.0, 1.0, 3.0, 5.0, 7.0])
    event = np.array([False, True, False, True, False, True, True, True])
    target = pd.DataFrame({"time": time, "event": event}, index=ids)

    table = kaplan_meier(prediction, target, n_groups=2)

    group0 = table[table["group"] == 0].reset_index(drop=True)
    assert group0["time"].tolist() == [0.0, 1.0, 2.0, 4.0]
    assert group0["at_risk"].tolist() == [4, 4, 3, 1]
    assert group0["events"].tolist() == [0, 0, 1, 1]
    assert group0["survival"].tolist() == pytest.approx([1.0, 1.0, 2 / 3, 0.0])

    group1 = table[table["group"] == 1].reset_index(drop=True)
    assert group1["time"].tolist() == [0.0, 1.0, 3.0, 5.0, 7.0]
    assert group1["at_risk"].tolist() == [4, 4, 3, 2, 1]
    assert group1["events"].tolist() == [0, 0, 1, 1, 1]
    assert group1["survival"].tolist() == pytest.approx([1.0, 1.0, 2 / 3, 1 / 3, 0.0])


def test_kaplan_meier_matches_sksurv_per_risk_group():
    """A randomised cohort with ties built in: integer-valued times give plenty of tied event times, and one time
    is forced to carry both an event and a censoring -- the case where a hand-rolled risk-set/event-count tally is
    most likely to diverge from sksurv's own estimator.
    """
    n = 80
    rng = np.random.default_rng(11)
    log_hazard = rng.normal(size=n)
    time = rng.integers(1, 15, size=n).astype(np.float64)
    event = rng.random(n) < 0.7  # ~30% censoring

    time[0], event[0] = 5.0, True
    time[1], event[1] = 5.0, False

    ids = _ids(n)
    prediction = pd.DataFrame({"log_hazard": log_hazard}, index=ids)
    target = pd.DataFrame({"time": time, "event": event}, index=ids)

    table = kaplan_meier(prediction, target, n_groups=2)
    group = risk_groups(prediction, n_groups=2).loc[ids]

    for name in (0, 1):
        mask = (group == name).to_numpy()
        expected_time, expected_survival = kaplan_meier_estimator(event[mask], time[mask])

        # sksurv's own estimator does not emit the conventional time=0 anchor row; drop it before comparing.
        observed = table[(table["group"] == name) & (table["time"] > 0)].sort_values("time")

        assert observed["time"].tolist() == pytest.approx(expected_time.tolist())
        assert observed["survival"].to_numpy() == pytest.approx(expected_survival, abs=1e-10)


def test_survival_is_monotonic_non_increasing_within_each_group():
    n = 60
    rng = np.random.default_rng(5)
    log_hazard = rng.normal(size=n)
    time = rng.exponential(5.0, size=n) + 1.0
    event = rng.random(n) < 0.7
    ids = _ids(n)

    prediction = pd.DataFrame({"log_hazard": log_hazard}, index=ids)
    target = pd.DataFrame({"time": time, "event": event}, index=ids)

    table = kaplan_meier(prediction, target, n_groups=2)

    for _, curve in table.groupby("group"):
        curve = curve.sort_values("time")
        assert (curve["survival"].diff().dropna() <= 1e-9).all()
        assert curve["survival"].max() <= 1.0 + 1e-9


def test_nan_log_hazard_is_excluded_and_reported():
    ids = _ids(10)
    log_hazard = np.array([0.1, 0.2, 0.3, np.nan, 0.5, 0.6, np.nan, 0.8, 0.9, 1.0])
    prediction = pd.DataFrame({"log_hazard": log_hazard}, index=ids)

    group = risk_groups(prediction, n_groups=2)

    assert group.attrs["n_excluded"] == 2
    assert len(group) == 8
    assert set(group.index) == {pid for pid, value in zip(ids, log_hazard, strict=True) if not np.isnan(value)}


def test_kaplan_meier_reports_the_same_excluded_count():
    ids = _ids(10)
    log_hazard = np.array([0.1, 0.2, 0.3, np.nan, 0.5, 0.6, np.nan, 0.8, 0.9, 1.0])
    prediction = pd.DataFrame({"log_hazard": log_hazard}, index=ids)
    target = pd.DataFrame(
        {"time": np.arange(1.0, 11.0), "event": np.ones(10, dtype=bool)},
        index=ids,
    )

    table = kaplan_meier(prediction, target, n_groups=2)

    assert table.attrs["n_excluded"] == 2


def test_n_groups_below_two_raises():
    prediction = pd.DataFrame({"log_hazard": [0.1, 0.2, 0.3]}, index=_ids(3))

    with pytest.raises(ValueError, match="n_groups must be at least 2"):
        risk_groups(prediction, n_groups=1)


def test_non_unique_quantile_cuts_raise():
    prediction = pd.DataFrame({"log_hazard": np.ones(20)}, index=_ids(20))

    with pytest.raises(ValueError, match="quantile cuts are not unique"):
        risk_groups(prediction, n_groups=2)


def test_labels_length_mismatch_raises():
    prediction = pd.DataFrame({"log_hazard": np.arange(10.0)}, index=_ids(10))

    with pytest.raises(ValueError, match=r"labels must have n_groups=2 entries"):
        risk_groups(prediction, n_groups=2, labels=["only_one"])


def test_kaplan_meier_with_tuple_labels_keeps_the_whole_tuple_per_row():
    """``DataFrame.insert`` treats a tuple value as one row value per element, not one scalar broadcast to every
    row; a 2-element tuple label landing in the ``group`` column of a differently-sized curve either raises a
    length mismatch or (worse) silently splits across rows whose counts happen to match the tuple's length.
    """
    ids = _ids(8)
    log_hazard = np.array([-2.0, -1.9, -1.8, -1.7, 1.7, 1.8, 1.9, 2.0])
    prediction = pd.DataFrame({"log_hazard": log_hazard}, index=ids)
    time = np.array([1.0, 2.0, 2.0, 4.0, 1.0, 3.0, 5.0, 7.0])
    event = np.array([False, True, False, True, False, True, True, True])
    target = pd.DataFrame({"time": time, "event": event}, index=ids)

    low, high = ("risk", "low"), ("risk", "high")

    table = kaplan_meier(prediction, target, n_groups=2, labels=[low, high])

    # group0's curve has 4 rows (time=0, 1, 2, 4) and group1's has 5 (time=0, 1, 3, 5, 7) -- see
    # test_known_cohort_produces_hand_checkable_km_steps for the hand-worked derivation.
    assert table["group"].tolist() == [low] * 4 + [high] * 5
