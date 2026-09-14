"""Survival endpoint construction for HANCOCK.

HANCOCK supports three time-to-event endpoints, each built from a
different pair of raw columns: overall survival (``survival_status`` +
``days_to_last_information``), recurrence-free survival (``recurrence`` +
``days_to_recurrence``), and disease-specific survival
(``survival_status_with_cause``). Only overall survival is implemented
here; the other two are deliberately deferred. Overall survival has the
most events of the three (213, vs 177 for recurrence and 115 for
disease-specific death) and therefore the most statistical power -- the
right endpoint to get the predict/evaluate pipeline working end to end
on first.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

_STRING_EVENT_MAP = {"living": False, "deceased": True}


def _status_to_event(value: object) -> bool | None:
    """Map one raw ``survival_status`` entry to an event bool, or ``None`` if unrecognised."""
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, str):
        key = value.strip().lower()
        return _STRING_EVENT_MAP.get(key)
    if isinstance(value, int | np.integer):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, float | np.floating):
        # pandas upcasts an int status column to float64 as soon as it has any
        # NaN elsewhere in the column, so 0.0/1.0 must map like 0/1. NaN != 0.0
        # and NaN != 1.0, so it falls through to "unrecognised" without a
        # separate isnan check.
        if value == 0.0:
            return False
        if value == 1.0:
            return True
        return None
    return None


def overall_survival_endpoint(
    survival_status: Sequence[object],
    days_to_last_information: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the overall-survival ``(time, event)`` pair from HANCOCK's raw columns.

    ``survival_status`` may be strings (``"living"`` / ``"deceased"``,
    case-insensitive), ints (``0`` / ``1``), floats (``0.0`` / ``1.0`` --
    pandas upcasts an int column to float64 as soon as it has any ``NaN``
    elsewhere in the column), or bools -- mixed within one call, since each
    row is mapped independently. ``1`` / ``True`` / ``"deceased"`` means the
    death event was observed; ``0`` / ``False`` / ``"living"`` means the
    patient was censored (alive at last follow-up).

    Rows with a missing (``NaN``) ``days_to_last_information`` are DROPPED,
    not raised on: a patient with no known follow-up time cannot contribute
    a time-to-event pair at all, and treating that as fatal would make the
    endpoint unusable on real data with incomplete follow-up. The returned
    ``retained_index`` records which original rows survived, so callers
    can apply the same selection to covariates, imaging bags, etc.

    The missing-time filter runs BEFORE status validation, so a row missing
    both its time and its status is simply dropped, not raised on -- on real
    data those absences routinely co-occur. Only rows retained after that
    filter must have a recognisable status.

    A present but non-positive time (``<= 0``) is a data-integrity error,
    not ordinary missingness, and raises rather than being dropped.

    Args:
        survival_status: Raw status per patient, length ``N``.
        days_to_last_information: Raw follow-up time per patient, length ``N``.

    Returns:
        A tuple ``(times, events, retained_index)``: ``times`` is
        ``float32`` shape ``(M,)``, ``events`` is ``bool`` shape ``(M,)``,
        ``retained_index`` is ``int64`` shape ``(M,)`` giving each
        retained row's position in the original (length-``N``) input.

    Raises:
        ValueError: If ``survival_status`` and ``days_to_last_information``
            have different lengths; if any ``survival_status`` value is
            unrecognised (message lists every distinct bad value found);
            or if any non-missing time is non-positive.
    """
    status_array = np.asarray(survival_status, dtype=object)
    times_array = np.asarray(days_to_last_information, dtype=np.float64)
    if status_array.shape[0] != times_array.shape[0]:
        raise ValueError(
            f"survival_status and days_to_last_information must share length, "
            f"got {status_array.shape[0]} and {times_array.shape[0]}"
        )

    # Drop missing-time rows FIRST: on real data, a missing follow-up time and
    # an unrecognised/missing status routinely co-occur on the same row, and a
    # row already being dropped for one reason should not also raise for the
    # other. Only rows that survive this filter need a valid status.
    retained_index = np.nonzero(~np.isnan(times_array))[0]
    retained_times = times_array[retained_index]
    retained_status = status_array[retained_index]

    mapped = [_status_to_event(value) for value in retained_status]
    unrecognised = sorted({repr(v) for v, m in zip(retained_status, mapped, strict=True) if m is None})
    if unrecognised:
        raise ValueError(
            "overall_survival_endpoint: unrecognised survival_status value(s) on rows with a known "
            f"follow-up time: {', '.join(unrecognised)}; expected 'living'/'deceased' (any case), 0/1, or bool"
        )
    events_retained = np.array(mapped, dtype=bool)

    non_positive = retained_times <= 0
    if np.any(non_positive):
        raise ValueError(
            "overall_survival_endpoint: found non-positive days_to_last_information "
            f"at original row(s) {retained_index[non_positive].tolist()}: "
            f"{retained_times[non_positive].tolist()}"
        )

    times = torch.as_tensor(retained_times, dtype=torch.float32)
    events = torch.as_tensor(events_retained, dtype=torch.bool)
    retained_index_t = torch.as_tensor(retained_index, dtype=torch.int64)
    return times, events, retained_index_t
