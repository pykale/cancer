"""Supervision targets, keyed by sample identifier.

Bound once at load time and keyed by identifier rather than row position, so a cohort
can be subset and recombined without the target needing realignment.

Nothing here is cancer-specific, tabular-specific or dependent on pandas:
``SurvivalTarget`` is the piece most likely to move into PyKale core, so it takes
plain arrays and depends only on numpy and torch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor


def _is_missing(value: Any) -> bool:
    """Whether one value is absent: ``None`` or float ``NaN``, whatever the column dtype."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _as_float(value: Any) -> float:
    """Coerce to float, returning NaN rather than raising, so the caller can name
    every offending sample at once."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


class SurvivalTarget:
    """Right-censored survival target: a time and an event indicator.

    Columns are named rather than positional, and the value meaning "the event
    happened" is declared. Both stop the same silent bug: a positional ``[time,
    status]`` pair or an assumed ``1 == event`` coding runs cleanly when reversed, and
    predicts survival inverted.

    Args:
        time (str): Column holding follow-up time.
        event (str): Column holding the event indicator.
        event_value (Any, optional): Value(s) meaning the event was observed. Any other
            *recorded* value is censored, which is what makes competing risks work:
            with ``0/1/2`` and ``event_value=1``, other-cause deaths are censored --
            the cause-specific convention. A *missing* value raises. Defaults to ``1``.
        unit (str | None, optional): What one unit of ``time`` is called. **A display
            label only** -- nothing is converted, so it cannot make a result wrong.
            ``None`` reports follow-up in "time units", right whatever the column holds.

    Raises:
        ValueError: On :meth:`bind`, if times or statuses are invalid or absent, or the
            event rate is 0% or 100% -- a mis-specified target, not an unusual cohort.
    """

    #: Used when no ``unit`` was given. Reporting the median in the column's own
    #: scale is correct whatever that scale is, so there is nothing left to assume.
    GENERIC_UNIT = "time units"

    def __init__(self, time: str, event: str, event_value: Any = 1, unit: str | None = None):
        self.time = time
        self.event = event
        self.event_values = list(event_value) if isinstance(event_value, list | tuple | set) else [event_value]
        self.unit = None if unit is None else str(unit).strip()
        self._row_of: dict[str, int] = {}
        self._times: Tensor = torch.empty(0)
        self._events: Tensor = torch.empty(0)

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The time and event columns, which a cohort must supply to :meth:`bind`."""
        return (self.time, self.event)

    def bind(self, identifiers: Sequence[str], values: Mapping[str, ArrayLike]) -> None:
        """Extract and validate times and events, keyed by identifier.

        Raises:
            ValueError: On non-numeric or negative times, a missing event status, or a
                degenerate event rate.
        """
        ids = list(identifiers)
        raw_times = np.asarray(values[self.time], dtype=object).ravel()
        raw_events = np.asarray(values[self.event], dtype=object).ravel()

        times = np.array([_as_float(v) for v in raw_times], dtype=float)
        bad = np.isnan(times)
        if bad.any():
            examples = [ids[i] for i in np.flatnonzero(bad)[:5]]
            raise ValueError(
                f"Column '{self.time}' has {int(bad.sum())} missing or non-numeric "
                f"value(s), e.g. for {examples}. Survival time must be known for every "
                f"sample; drop or impute these before constructing the cohort."
            )
        if (times < 0).any():
            raise ValueError(f"Column '{self.time}' contains negative values.")

        unknown = np.array([_is_missing(v) for v in raw_events])
        if unknown.any():
            examples = [ids[i] for i in np.flatnonzero(unknown)[:5]]
            raise ValueError(
                f"Column '{self.event}' has {int(unknown.sum())} missing value(s), e.g. for "
                f"{examples}. An unknown outcome is not a censored one: censoring asserts "
                f"the sample was event-free throughout its recorded time, which an absent "
                f"status does not establish. Treating these as censored would drop events "
                f"from the numerator while keeping their follow-up in the denominator, "
                f"biasing every estimate towards the null. Decide explicitly before "
                f"constructing the cohort -- drop these rows, or recode them as censored "
                f"if that is what a blank means in your data dictionary."
            )

        events = np.array([v in self.event_values for v in raw_events], dtype=float)
        rate = float(events.mean()) if events.size else 0.0
        if rate in (0.0, 1.0):
            observed = sorted({str(v) for v in raw_events})[:10]
            raise ValueError(
                f"event_value={self.event_values} matches {rate:.0%} of rows in column "
                f"'{self.event}', which cannot be right. Observed values: {observed}"
            )

        # Tensors rather than dicts of floats: for_() then returns a view instead of
        # allocating two tensors per sample per epoch.
        self._row_of = {identifier: i for i, identifier in enumerate(ids)}
        self._times = torch.tensor(times, dtype=torch.float32)
        self._events = torch.tensor(events, dtype=torch.float32)

    def for_(self, identifier: str) -> dict[str, Tensor]:
        """Return ``{"time": Tensor, "event": Tensor}`` for one sample."""
        row = self._row_of[identifier]
        return {"time": self._times[row], "event": self._events[row]}

    def events_for(self, identifiers: Sequence[str]) -> np.ndarray:
        """Event indicators for ``identifiers``, as a float array."""
        return self._gather(self._events, identifiers)

    def times_for(self, identifiers: Sequence[str]) -> np.ndarray:
        """Follow-up times for ``identifiers``, as a float array."""
        return self._gather(self._times, identifiers)

    def stratify_labels(self, identifiers: Sequence[str]) -> np.ndarray:
        """Labels to stratify a split on: the event indicator.

        The optional half of the ``Target`` contract, used by ``Cohort.split``.
        """
        return self.events_for(identifiers)

    def _gather(self, source: Tensor, identifiers: Sequence[str]) -> np.ndarray:
        rows = torch.tensor([self._row_of[i] for i in identifiers], dtype=torch.long)
        return source[rows].numpy().astype(float)

    def summarise(self, identifiers: Sequence[str]) -> str:
        """One-line description of events and follow-up across ``identifiers``."""
        events = self.events_for(identifiers)
        times = self.times_for(identifiers)
        n_events = int(events.sum())
        parts = [f"{n_events} events ({n_events / max(len(events), 1):.1%})"]

        # Median follow-up among the censored: the simple approximation, not reverse KM.
        censored = times[events == 0]
        if censored.size:
            parts.append(f"median follow-up {np.median(censored):.1f} {self.unit or self.GENERIC_UNIT}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        parts = [f"time={self.time!r}", f"event={self.event!r}", f"event_value={self.event_values}"]
        if len(self._row_of):
            parts.append(self.summarise(list(self._row_of)))
        return f"{type(self).__name__}({' | '.join(parts)})"
