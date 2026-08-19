"""Tests for ``kalecancer.prepdata.endpoints``."""

from __future__ import annotations

import math

import pytest
import torch

from kalecancer.prepdata.endpoints import overall_survival_endpoint

_TIMES = [100.0, 200.0, 300.0, 400.0, 500.0]
_EXPECTED_EVENTS = [False, True, False, True, False]


def test_string_int_bool_inputs_give_same_result() -> None:
    status_strings = ["living", "DECEASED", "Living", "deceased", "living"]
    status_ints = [0, 1, 0, 1, 0]
    status_bools = [False, True, False, True, False]

    times_s, events_s, idx_s = overall_survival_endpoint(status_strings, _TIMES)
    times_i, events_i, idx_i = overall_survival_endpoint(status_ints, _TIMES)
    times_b, events_b, idx_b = overall_survival_endpoint(status_bools, _TIMES)

    assert torch.equal(times_s, times_i) and torch.equal(times_i, times_b)
    assert torch.equal(events_s, events_i) and torch.equal(events_i, events_b)
    assert torch.equal(idx_s, idx_i) and torch.equal(idx_i, idx_b)
    assert events_s.tolist() == _EXPECTED_EVENTS


def test_unknown_status_raises_and_names_the_value() -> None:
    status = ["living", "deceased", "unknown", "moribund"]
    times = [100.0, 200.0, 300.0, 400.0]

    with pytest.raises(ValueError, match="unrecognised survival_status"):
        overall_survival_endpoint(status, times)

    try:
        overall_survival_endpoint(status, times)
    except ValueError as exc:
        assert "'unknown'" in str(exc)
        assert "'moribund'" in str(exc)


def test_missing_times_are_dropped_and_index_tracks_survivors() -> None:
    status = ["living", "deceased", "living", "deceased", "living"]
    times = [100.0, math.nan, 300.0, math.nan, 500.0]

    result_times, result_events, retained_index = overall_survival_endpoint(status, times)

    assert retained_index.tolist() == [0, 2, 4]
    assert result_times.tolist() == [100.0, 300.0, 500.0]
    assert result_events.tolist() == [False, False, False]


def test_non_positive_time_raises() -> None:
    status = ["living", "deceased", "living"]
    times = [100.0, 0.0, 300.0]

    with pytest.raises(ValueError, match="non-positive"):
        overall_survival_endpoint(status, times)


def test_row_count_matches_across_outputs() -> None:
    status = ["living", "deceased", "living", "deceased"]
    times = [10.0, math.nan, 30.0, 40.0]

    result_times, result_events, retained_index = overall_survival_endpoint(status, times)

    assert result_times.shape[0] == result_events.shape[0] == retained_index.shape[0] == 3


def test_output_dtypes() -> None:
    result_times, result_events, retained_index = overall_survival_endpoint(["living", "deceased"], [1.0, 2.0])

    assert result_times.dtype == torch.float32
    assert result_events.dtype == torch.bool
    assert retained_index.dtype == torch.int64


def test_float_status_column_maps_correctly() -> None:
    # pandas upcasts an int status column to float64 as soon as it has any
    # NaN elsewhere in the column, so a real HANCOCK column can arrive as
    # 0.0/1.0 rather than 0/1.
    status_floats = [0.0, 1.0, 0.0, 1.0, 0.0]

    _, events_f, _ = overall_survival_endpoint(status_floats, _TIMES)

    assert events_f.tolist() == _EXPECTED_EVENTS


def test_row_with_missing_time_and_bad_status_is_dropped_not_raised() -> None:
    # On real data, a missing follow-up time and an unrecognised/missing
    # status routinely co-occur on the same row. That row should simply be
    # dropped by the missing-time filter, not raise on its bad status.
    status = ["living", math.nan, "deceased"]
    times = [100.0, math.nan, 300.0]

    result_times, result_events, retained_index = overall_survival_endpoint(status, times)

    assert retained_index.tolist() == [0, 2]
    assert result_times.tolist() == [100.0, 300.0]
    assert result_events.tolist() == [False, True]


def test_retained_row_with_bad_status_still_raises() -> None:
    # Row 0's bad status is dropped along with its missing time -- but row 1's
    # bad status is on a row with a KNOWN time, so it must still raise.
    status = ["unknown", "moribund", "deceased"]
    times = [math.nan, 300.0, 100.0]

    with pytest.raises(ValueError, match="unrecognised survival_status"):
        overall_survival_endpoint(status, times)
