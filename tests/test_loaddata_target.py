"""Tests for ``kalecancer.loaddata.target.SurvivalTarget``.

The target is where a silent, catastrophic bug lives if it is allowed to: a
mis-declared ``event_value`` produces a model that predicts survival exactly
inverted, trains cleanly, and reports a plausible-looking c-index below 0.5.
Most of what follows is therefore about the guards, and about the invariant that
makes them possible -- that a target is keyed by **identifier**, never by row
position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from kalecancer.loaddata.base import Target
from kalecancer.survival.survival_target import SurvivalTarget
from tests.conftest import EVENT_COLUMN, EVENT_VALUE, TIME_COLUMN

# --------------------------------------------------------------------------- #
# protocol conformance
# --------------------------------------------------------------------------- #


def test_survival_target_satisfies_the_target_protocol(make_target):
    """The data layer declares ``Target``; the modelling layer implements it."""
    assert isinstance(make_target(), Target)


def test_protocol_requires_only_bind_and_for_(make_target):
    """``Target`` deliberately excludes the survival-specific members."""
    target = make_target()
    assert hasattr(target, "bind") and hasattr(target, "for_")
    # These exist on SurvivalTarget but are not part of the universal contract.
    assert not hasattr(Target, "events_for")
    assert not hasattr(Target, "summarise")


# --------------------------------------------------------------------------- #
# binding
# --------------------------------------------------------------------------- #


def test_bind_extracts_times_and_events_keyed_by_identifier(frame, make_target):
    target = make_target()
    target.bind(frame, "patient_id")

    expected_events = frame[EVENT_COLUMN].eq(EVENT_VALUE)
    np.testing.assert_array_equal(target.events_for(frame["patient_id"]), expected_events.to_numpy(dtype=float))
    np.testing.assert_allclose(target.times_for(frame["patient_id"]), frame[TIME_COLUMN].to_numpy(dtype=float))


def test_for_returns_scalar_float32_tensors(frame, make_target):
    target = make_target()
    target.bind(frame, "patient_id")

    values = target.for_("001")
    assert set(values) == {"time", "event"}
    for tensor in values.values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype is torch.float32
        assert tensor.ndim == 0, "scalars, so default_collate stacks them into (batch,)"

    row = frame.loc[frame["patient_id"] == "001"].iloc[0]
    assert values["time"].item() == pytest.approx(float(row[TIME_COLUMN]))
    assert values["event"].item() == pytest.approx(float(row[EVENT_COLUMN] == EVENT_VALUE))


def test_lookups_are_by_identifier_not_position(frame, make_target):
    """The invariant the whole design rests on: order of the query, not the frame."""
    target = make_target()
    target.bind(frame, "patient_id")

    shuffled = frame["patient_id"].tolist()[::-1]
    np.testing.assert_allclose(target.times_for(shuffled), frame[TIME_COLUMN].to_numpy(dtype=float)[::-1])


def test_bind_is_indifferent_to_the_frames_row_index(frame, make_target):
    """Subsetting a frame leaves a non-contiguous index; that must not matter."""
    scrambled = frame.iloc[::-1].copy()  # index now 79, 78, ... 0
    target = make_target()
    target.bind(scrambled, "patient_id")

    reference = make_target()
    reference.bind(frame, "patient_id")

    ids = frame["patient_id"].tolist()
    np.testing.assert_allclose(target.times_for(ids), reference.times_for(ids))
    np.testing.assert_array_equal(target.events_for(ids), reference.events_for(ids))


def test_rebinding_replaces_previous_state(frame, make_target):
    target = make_target()
    target.bind(frame, "patient_id")

    half = frame.iloc[:10]
    target.bind(half, "patient_id")

    assert target.times_for(half["patient_id"]).size == 10
    with pytest.raises(KeyError):
        target.for_(frame["patient_id"].iloc[50])


@pytest.mark.parametrize("method", ["for_", "events_for", "times_for"])
def test_unknown_identifier_raises_keyerror(frame, make_target, method):
    target = make_target()
    target.bind(frame, "patient_id")
    argument = "no-such-patient" if method == "for_" else ["no-such-patient"]
    with pytest.raises(KeyError):
        getattr(target, method)(argument)


# --------------------------------------------------------------------------- #
# event coding
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "event_value, expected",
    [
        (1, [1.0, 0.0, 0.0, 0.0]),
        ([1, 2], [1.0, 1.0, 0.0, 0.0]),
        ((1, 2), [1.0, 1.0, 0.0, 0.0]),
        ({1}, [1.0, 0.0, 0.0, 0.0]),
    ],
)
def test_event_value_accepts_a_scalar_or_a_collection(event_value, expected):
    """Competing-risks codings need more than one value to count as an event."""
    frame = pd.DataFrame({"pid": ["a", "b", "c", "d"], "t": [10, 20, 30, 40], "status": [1, 2, 0, 0]})
    target = SurvivalTarget(time="t", event="status", event_value=event_value)
    target.bind(frame, "pid")
    np.testing.assert_array_equal(target.events_for(["a", "b", "c", "d"]), expected)


def test_recorded_values_outside_event_value_are_censored():
    """Any status that was *recorded* and is not an event value is censored.

    This is the half of the ``isin`` behaviour that is correct and load-bearing:
    it is what makes the competing-risks case above work.
    """
    frame = pd.DataFrame({"pid": ["a", "b", "c"], "t": [10, 20, 30], "status": ["dead", "alive", "lost"]})
    target = SurvivalTarget(time="t", event="status", event_value="dead")
    target.bind(frame, "pid")
    np.testing.assert_array_equal(target.events_for(["a", "b", "c"]), [1.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# guards -- each of these is a silent wrong answer if it does not fire
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("column", ["time", "event"])
def test_missing_target_column_raises_and_lists_what_is_available(frame, column):
    target = SurvivalTarget(
        time="nope" if column == "time" else TIME_COLUMN,
        event="nope" if column == "event" else EVENT_COLUMN,
        event_value=EVENT_VALUE,
    )
    with pytest.raises(ValueError, match="not found") as excinfo:
        target.bind(frame, "patient_id")
    assert "nope" in str(excinfo.value)
    assert TIME_COLUMN in str(excinfo.value), "the message should list the real columns"


def test_non_numeric_time_raises(frame, make_target):
    target = make_target(time="sex")
    with pytest.raises(ValueError, match="non-numeric"):
        target.bind(frame, "patient_id")


def test_missing_time_raises_and_names_the_offending_patients(frame, make_target):
    frame.loc[2, TIME_COLUMN] = np.nan
    with pytest.raises(ValueError, match="missing or non-numeric") as excinfo:
        make_target().bind(frame, "patient_id")
    assert frame.loc[2, "patient_id"] in str(excinfo.value)


def test_missing_event_status_raises_and_names_the_offending_patients(frame, make_target):
    """An unknown outcome is refused, not quietly reclassified as censored.

    Censoring is a positive claim -- event-free throughout the recorded follow-up --
    and an absent status does not establish it. Assuming it removes events from the
    numerator while keeping their person-time in the denominator, which attenuates
    every estimate downstream and is invisible in the result.

    This matches the field: lifelines raises ``NaNs were detected in the dataset``
    and scikit-survival raises ``event indicator must be binary``. R's ``coxph``
    drops the rows and reports the count. Nothing in common use censors them.
    """
    frame.loc[[4, 11], EVENT_COLUMN] = np.nan

    with pytest.raises(ValueError, match="missing value") as excinfo:
        make_target().bind(frame, "patient_id")

    message = str(excinfo.value)
    assert "2 missing value(s)" in message
    assert frame.loc[4, "patient_id"] in message
    assert frame.loc[11, "patient_id"] in message


def test_missing_event_status_raises_before_the_degenerate_rate_guard(frame, make_target):
    """An all-missing status column reports the missingness, not an odd 0% event rate.

    Order matters for the message: ``isin`` maps every NaN to False, so without this
    guard running first the user is told ``event_value=['deceased'] matches 0% of
    rows ... Observed values: []`` -- which describes the symptom and hides the cause.
    """
    frame[EVENT_COLUMN] = np.nan
    with pytest.raises(ValueError, match="missing value"):
        make_target().bind(frame, "patient_id")


def test_dropping_the_unknown_rows_is_all_it_takes_to_proceed(frame, make_target):
    """The remedy the message points at has to actually work, in one line.

    ``TabularDataset`` accepts a DataFrame precisely so this can happen in the
    caller's script, where the decision is visible and reportable.
    """
    frame.loc[[4, 11], EVENT_COLUMN] = np.nan
    resolved = frame[frame[EVENT_COLUMN].notna()]

    target = make_target()
    target.bind(resolved, "patient_id")

    assert len(target.events_for(resolved["patient_id"].tolist())) == len(frame) - 2


def test_negative_time_raises(frame, make_target):
    frame.loc[5, TIME_COLUMN] = -1
    with pytest.raises(ValueError, match="negative"):
        make_target().bind(frame, "patient_id")


def test_zero_time_is_allowed(frame, make_target):
    """Zero follow-up is a real observation, not an error."""
    frame.loc[5, TIME_COLUMN] = 0
    target = make_target()
    target.bind(frame, "patient_id")
    assert target.times_for([frame.loc[5, "patient_id"]])[0] == 0.0


@pytest.mark.parametrize("event_value", ["no-such-status", "living"])
def test_degenerate_event_rate_raises(frame, event_value, make_target):
    """0% and 100% both mean the coding is wrong, not that the cohort is unusual."""
    frame[EVENT_COLUMN] = "living"
    with pytest.raises(ValueError, match="which cannot be right") as excinfo:
        make_target(event_value=event_value).bind(frame, "patient_id")
    assert "living" in str(excinfo.value), "the message should show the observed values"


def test_the_typo_that_motivates_explicit_event_value(frame, make_target):
    """``"dead"`` where the column says ``"deceased"`` -- caught at bind time."""
    with pytest.raises(ValueError, match="matches 0%"):
        make_target(event_value="dead").bind(frame, "patient_id")


# --------------------------------------------------------------------------- #
# summarise
# --------------------------------------------------------------------------- #


def test_summarise_reports_event_count_and_rate(frame, make_target):
    target = make_target()
    target.bind(frame, "patient_id")

    ids = frame["patient_id"].tolist()
    n_events = int(frame[EVENT_COLUMN].eq(EVENT_VALUE).sum())
    summary = target.summarise(ids)

    assert f"{n_events} events" in summary
    assert f"({n_events / len(ids):.1%})" in summary
    assert "median follow-up" in summary


def test_summarise_follows_up_only_the_censored(frame, make_target):
    """Median follow-up over everyone would be dominated by the event times."""
    target = make_target()
    target.bind(frame, "patient_id")

    ids = frame["patient_id"].tolist()
    censored = frame.loc[frame[EVENT_COLUMN] != EVENT_VALUE, TIME_COLUMN]
    expected_months = float(np.median(censored)) / 365.25 * 12.0

    assert f"median follow-up {expected_months:.1f} months" in target.summarise(ids)


def test_summarise_omits_follow_up_when_every_sample_had_the_event(frame, make_target):
    target = make_target()
    target.bind(frame, "patient_id")

    all_events = frame.loc[frame[EVENT_COLUMN] == EVENT_VALUE, "patient_id"].tolist()
    summary = target.summarise(all_events)

    assert "100.0%" in summary
    assert "follow-up" not in summary


def test_summarise_on_no_identifiers_does_not_divide_by_zero(frame, make_target):
    target = make_target()
    target.bind(frame, "patient_id")
    assert "0 events" in target.summarise([])


@pytest.mark.parametrize("unit, per_year", [("days", 365.25), ("weeks", 52.18), ("months", 12.0), ("years", 1.0)])
def test_summarise_converts_the_declared_unit_to_months(unit, per_year):
    frame = pd.DataFrame({"pid": list("abcd"), "t": [1.0, 2.0, 3.0, 4.0], "status": [1, 0, 0, 0]})
    target = SurvivalTarget(time="t", event="status", unit=unit)
    target.bind(frame, "pid")
    expected = 3.0 / per_year * 12.0  # median of the censored times 2, 3, 4
    assert f"median follow-up {expected:.1f} months" in target.summarise(list("abcd"))


@pytest.mark.parametrize("unit", ["DAYS", "Months", " years "])
def test_unit_is_case_and_whitespace_insensitive(unit):
    """A unit is typed by hand once; casing should not be a way to get it wrong."""
    assert SurvivalTarget(time="t", event="status", unit=unit).unit == unit.strip().lower()


def test_unknown_unit_raises_at_construction_and_lists_the_accepted_ones():
    """An unrecognised unit is a typo, not a request for the identity conversion.

    Falling back to no conversion made ``unit="day"`` silently mean years, so a
    1200-day follow-up printed as 14400 months. That is a 365x error in the one
    number whose job is to make a mis-specified target visible on sight. It also
    stops being cosmetic as soon as a horizon-based metric (``auc@24mo``) has to
    convert months into the time column's units.
    """
    with pytest.raises(ValueError, match="Unknown time unit 'fortnights'") as excinfo:
        SurvivalTarget(time="t", event="status", unit="fortnights")

    assert "days" in str(excinfo.value)  # the message names what would have worked
