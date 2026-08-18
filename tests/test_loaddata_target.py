"""``SurvivalTarget``: binding, validation, and the guards that stop silent bias.

The target takes arrays rather than a DataFrame, so these tests bind directly
rather than going through a cohort. That is the contract a non-tabular cohort
would have to satisfy, and testing it here keeps the two independent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from kalecancer.survival.survival_target import SurvivalTarget
from tests.conftest import EVENT_COLUMN, EVENT_VALUE, TIME_COLUMN


def bind(target: SurvivalTarget, frame: pd.DataFrame, identifier: str = "patient_id") -> SurvivalTarget:
    """Bind ``target`` to ``frame``, the way a cohort would."""
    target.bind(frame[identifier].tolist(), {c: frame[c].to_numpy() for c in target.required_columns})
    return target


# =========================================================================== #
# the contract
# =========================================================================== #


def test_required_columns_are_the_declared_time_and_event(make_target):
    assert make_target().required_columns == (TIME_COLUMN, EVENT_COLUMN)


def test_bind_takes_arrays_not_a_dataframe(frame, make_target):
    """Pandas must not appear in the contract, so plain arrays have to work."""
    target = make_target()
    target.bind(
        list(frame["patient_id"]),
        {TIME_COLUMN: np.asarray(frame[TIME_COLUMN]), EVENT_COLUMN: np.asarray(frame[EVENT_COLUMN])},
    )
    assert len(target.times_for(frame["patient_id"].tolist())) == len(frame)


def test_bind_accepts_plain_python_lists(frame, make_target):
    target = make_target()
    target.bind(
        frame["patient_id"].tolist(),
        {TIME_COLUMN: frame[TIME_COLUMN].tolist(), EVENT_COLUMN: frame[EVENT_COLUMN].tolist()},
    )
    assert target.events_for(["001"]).shape == (1,)


# =========================================================================== #
# binding and lookup
# =========================================================================== #


def test_bind_extracts_times_and_events_keyed_by_identifier(frame, make_target):
    target = bind(make_target(), frame)
    expected = frame[EVENT_COLUMN].eq(EVENT_VALUE).to_numpy(dtype=float)
    np.testing.assert_array_equal(target.events_for(frame["patient_id"].tolist()), expected)


def test_for_returns_scalar_float32_tensors(frame, make_target):
    target = bind(make_target(), frame)
    values = target.for_("001")

    assert set(values) == {"time", "event"}
    for tensor in values.values():
        assert tensor.dtype is torch.float32
        assert tensor.shape == ()


def test_named_keys_not_a_positional_pack(frame, make_target):
    """A ``tensor([time, event])`` built backwards runs clean and inverts survival."""
    values = bind(make_target(), frame).for_("001")
    assert values["time"].item() == pytest.approx(float(frame.loc[0, TIME_COLUMN]))
    assert values["event"].item() == 1.0


def test_lookups_are_by_identifier_not_position(frame, make_target):
    """Reversing the rows must not change what any patient's outcome is."""
    target = bind(make_target(), frame)
    reversed_target = bind(make_target(), frame.iloc[::-1].reset_index(drop=True))

    for identifier in frame["patient_id"].head(10):
        assert target.for_(identifier)["time"] == reversed_target.for_(identifier)["time"]


def test_bind_is_indifferent_to_the_frames_row_index(frame, make_target):
    shuffled = frame.sample(frac=1.0, random_state=0)
    target = bind(make_target(), shuffled)
    assert target.for_("001")["time"].item() == pytest.approx(float(frame.loc[0, TIME_COLUMN]))


def test_rebinding_replaces_previous_state(frame, make_target):
    target = bind(make_target(), frame)
    bind(target, frame.head(5))

    assert target.for_("001") is not None
    with pytest.raises(KeyError):
        target.for_("080")


@pytest.mark.parametrize("method", ["for_", "events_for", "times_for"])
def test_unknown_identifier_raises_keyerror(frame, make_target, method):
    target = bind(make_target(), frame)
    argument = "nope" if method == "for_" else ["nope"]
    with pytest.raises(KeyError):
        getattr(target, method)(argument)


def test_stratify_labels_are_the_event_indicators(frame, make_target):
    target = bind(make_target(), frame)
    ids = frame["patient_id"].tolist()
    np.testing.assert_array_equal(target.stratify_labels(ids), target.events_for(ids))


# =========================================================================== #
# event coding
# =========================================================================== #


@pytest.mark.parametrize(
    ("event_value", "expected"),
    [(1, [1.0, 0.0, 0.0]), ([1, 2], [1.0, 1.0, 0.0]), ((2,), [0.0, 1.0, 0.0])],
)
def test_event_value_accepts_a_scalar_or_a_collection(event_value, expected):
    frame = pd.DataFrame({"patient_id": ["a", "b", "c"], "t": [1, 2, 3], "e": [1, 2, 0]})
    target = bind(SurvivalTarget(time="t", event="e", event_value=event_value), frame)
    np.testing.assert_array_equal(target.events_for(["a", "b", "c"]), expected)


def test_recorded_values_outside_event_value_are_censored():
    """Cause-specific competing risks: a death from another cause is censored."""
    frame = pd.DataFrame({"patient_id": ["a", "b", "c"], "t": [5, 6, 7], "e": [0, 1, 2]})
    target = bind(SurvivalTarget(time="t", event="e", event_value=1), frame)
    np.testing.assert_array_equal(target.events_for(["a", "b", "c"]), [0.0, 1.0, 0.0])


def test_the_typo_that_motivates_explicit_event_value(frame, make_target):
    """``"dead"`` where the data says ``"deceased"`` must not silently mean zero events."""
    with pytest.raises(ValueError, match="matches 0%"):
        bind(make_target(event_value="dead"), frame)


@pytest.mark.parametrize("event_value", [EVENT_VALUE, "living"])
def test_degenerate_event_rate_raises(frame, event_value, make_target):
    frame[EVENT_COLUMN] = event_value
    with pytest.raises(ValueError, match="cannot be right"):
        bind(make_target(), frame)


# =========================================================================== #
# time validation
# =========================================================================== #


def test_non_numeric_time_raises(frame, make_target):
    frame[TIME_COLUMN] = frame[TIME_COLUMN].astype(object)
    frame.loc[2, TIME_COLUMN] = "not a number"
    with pytest.raises(ValueError, match="missing or non-numeric"):
        bind(make_target(), frame)


def test_missing_time_raises_and_names_the_offending_patients(frame, make_target):
    frame[TIME_COLUMN] = frame[TIME_COLUMN].astype(float)
    frame.loc[4, TIME_COLUMN] = np.nan
    with pytest.raises(ValueError, match="'005'"):
        bind(make_target(), frame)


def test_negative_time_raises(frame, make_target):
    frame.loc[0, TIME_COLUMN] = -1
    with pytest.raises(ValueError, match="negative"):
        bind(make_target(), frame)


def test_zero_time_is_allowed(frame, make_target):
    frame.loc[0, TIME_COLUMN] = 0
    assert bind(make_target(), frame).for_("001")["time"].item() == 0.0


# =========================================================================== #
# an unknown outcome is not a censored one
# =========================================================================== #


def test_missing_event_status_raises_and_names_the_offending_patients(frame, make_target):
    """Treating an absent status as censored biases every estimate towards the null.

    Censoring asserts the sample was event-free throughout its recorded time; a
    blank cell establishes nothing of the sort. It would drop events from the
    numerator while keeping their follow-up in the denominator.
    """
    frame.loc[[6, 7], EVENT_COLUMN] = None
    with pytest.raises(ValueError) as excinfo:
        bind(make_target(), frame)

    message = str(excinfo.value)
    assert "'007'" in message and "'008'" in message
    assert "not a censored one" in message


def test_missing_event_status_raises_before_the_degenerate_rate_guard(frame, make_target):
    """Otherwise the message sends the user off to check ``event_value`` instead."""
    frame[EVENT_COLUMN] = None
    with pytest.raises(ValueError, match="not a censored one"):
        bind(make_target(), frame)


def test_dropping_the_unknown_rows_is_all_it_takes_to_proceed(frame, make_target):
    """The documented remedy has to actually work, or the message is a dead end."""
    frame.loc[[6, 7], EVENT_COLUMN] = None
    resolved = frame[frame[EVENT_COLUMN].notna()].reset_index(drop=True)
    target = bind(make_target(), resolved)
    assert len(target.events_for(resolved["patient_id"].tolist())) == len(frame) - 2


# =========================================================================== #
# reporting
# =========================================================================== #


def test_summarise_reports_event_count_and_rate(frame, make_target):
    target = bind(make_target(), frame)
    n_events = int(frame[EVENT_COLUMN].eq(EVENT_VALUE).sum())
    summary = target.summarise(frame["patient_id"].tolist())

    assert f"{n_events} events" in summary
    assert f"({n_events / len(frame):.1%})" in summary


def test_summarise_follows_up_only_the_censored(frame, make_target):
    """Median follow-up over everyone would be a different, wrong number."""
    target = bind(make_target(), frame)
    censored = frame.loc[frame[EVENT_COLUMN] != EVENT_VALUE, TIME_COLUMN]
    assert f"{np.median(censored):.1f}" in target.summarise(frame["patient_id"].tolist())


def test_summarise_omits_follow_up_when_every_sample_had_the_event(frame, make_target):
    target = bind(make_target(), frame)
    with_events = frame.loc[frame[EVENT_COLUMN] == EVENT_VALUE, "patient_id"].tolist()
    assert "follow-up" not in target.summarise(with_events)


def test_summarise_on_no_identifiers_does_not_divide_by_zero(frame, make_target):
    assert "0 events" in bind(make_target(), frame).summarise([])


@pytest.mark.parametrize("unit", [None, "days", "months", "years", "cycles"])
def test_the_reported_median_is_the_columns_own_median_whatever_the_unit(frame, make_target, unit):
    """Nothing is converted, so no declared unit can make the number wrong."""
    target = bind(make_target(unit=unit), frame)
    censored = frame.loc[frame[EVENT_COLUMN] != EVENT_VALUE, TIME_COLUMN]
    summary = target.summarise(frame["patient_id"].tolist())

    assert f"{np.median(censored):.1f}" in summary
    assert (unit or SurvivalTarget.GENERIC_UNIT) in summary


def test_no_unit_reports_a_scale_free_label(frame, make_target):
    """The default states nothing about the column, which is why it cannot be wrong."""
    assert "time units" in bind(make_target(unit=None), frame).summarise(frame["patient_id"].tolist())


def test_a_unit_is_a_label_and_never_arithmetic(frame, make_target):
    """A unit has no correct arithmetic to do: the partial likelihood and the c-index
    are built from the ordering of event times, so rescaling changes nothing."""
    censored = frame.loc[frame[EVENT_COLUMN] != EVENT_VALUE, TIME_COLUMN]
    reported = [
        bind(make_target(unit=u), frame).summarise(frame["patient_id"].tolist()).split("follow-up ")[1].split()[0]
        for u in ("days", "months", "years")
    ]
    assert reported == [f"{np.median(censored):.1f}"] * 3


@pytest.mark.parametrize(("given", "expected"), [(" days ", "days"), ("Days", "Days"), (None, None)])
def test_unit_is_stripped_but_otherwise_untouched(given, expected):
    """A free label, so case is the caller's business; only stray whitespace goes."""
    assert SurvivalTarget(time="t", event="e", unit=given).unit == expected


def test_repr_surfaces_the_event_rate_once_bound(frame, make_target):
    assert "events" in repr(bind(make_target(), frame))
