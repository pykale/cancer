"""Supervision targets, keyed by sample identifier.

A target is bound once, at load time, against the values that carry it. It is keyed
by identifier rather than by row position, which is what lets a cohort be subset,
split and recombined without the target ever needing realignment.

Nothing here is cancer-specific, tabular-specific, or dependent on pandas:
``SurvivalTarget`` is the piece most likely to move into PyKale core alongside the
survival heads, losses and metrics that consume it, so it takes plain arrays and
keeps its dependencies to numpy and torch.
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
    """Whether one value is absent, across the dtypes a table can produce.

    Covers ``None`` and float ``NaN``, which is what a blank cell becomes whether
    the surrounding column was read as object, float, or a mix.
    """
    return value is None or (isinstance(value, float) and math.isnan(value))


def _as_float(value: Any) -> float:
    """Coerce one value to float, returning NaN rather than raising.

    Failures are collected and reported together by the caller, naming the samples
    involved -- more useful than the first exception numpy would throw.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


class SurvivalTarget:
    """Right-censored survival target: a time and an event indicator.

    The columns are named rather than positional, and the value that means "the
    event happened" is declared explicitly. Both choices exist to stop the same
    silent bug: with a positional ``[time, status]`` pair or an assumed ``1 ==
    event`` coding, getting it backwards runs cleanly and predicts survival
    inverted.

    Args:
        time (str): Column holding follow-up time.
        event (str): Column holding the event indicator.
        event_value (Any, optional): Value(s) in ``event`` meaning the event was
            observed, e.g. ``"deceased"`` or ``1``. Any other *recorded* value is
            treated as censored, which is what makes a competing-risks coding work:
            with ``0/1/2`` and ``event_value=1``, deaths from other causes are
            censored, the cause-specific convention. A *missing* value is not a
            value and raises -- see :meth:`bind`. Defaults to ``1``.
        unit (str, optional): Unit of the ``time`` column, one of ``"days"``,
            ``"weeks"``, ``"months"``, ``"years"``. Case-insensitive. Defaults to
            ``"days"``.

    Raises:
        ValueError: On construction, if ``unit`` is not a recognised time unit.
        ValueError: On :meth:`bind`, if times or statuses are invalid or absent, or
            the resulting event rate is 0% or 100% -- all of which indicate a
            mis-specified target rather than an unusual cohort.
    """

    _PER_YEAR = {"days": 365.25, "months": 12.0, "years": 1.0, "weeks": 52.18}

    def __init__(self, time: str, event: str, event_value: Any = 1, unit: str = "days"):
        self.time = time
        self.event = event
        self.event_values = list(event_value) if isinstance(event_value, list | tuple | set) else [event_value]
        # Validated here rather than defaulted at the point of use: the vocabulary is
        # four strings typed by hand, so 'day' or 'DAYS' silently meaning 'years' is a
        # 365x error in a number whose job is to make mis-specification visible.
        self.unit = str(unit).strip().lower()
        if self.unit not in self._PER_YEAR:
            raise ValueError(f"Unknown time unit '{unit}'. Expected one of {sorted(self._PER_YEAR)}.")
        self._row_of: dict[str, int] = {}
        self._times: Tensor = torch.empty(0)
        self._events: Tensor = torch.empty(0)

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The time and event columns, which a cohort must supply to :meth:`bind`."""
        return (self.time, self.event)

    def bind(self, identifiers: Sequence[str], values: Mapping[str, ArrayLike]) -> None:
        """Extract and validate times and events, keyed by identifier.

        Args:
            identifiers (Sequence[str]): Sample identifiers, in row order.
            values (Mapping[str, ArrayLike]): Arrays for :attr:`required_columns`,
                aligned with ``identifiers``.

        Raises:
            ValueError: On non-numeric or negative times, a missing event status, or
                a degenerate event rate.
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

        # Stored as tensors rather than dicts of Python floats so that for_() returns
        # a view instead of allocating two tensors per sample per epoch, and so the
        # array-returning accessors below are gathers rather than list comprehensions.
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

        The optional half of the ``Target`` contract, used by
        :meth:`~kalecancer.loaddata.base.Cohort.split`. Event status is what matters
        to balance here -- an unstratified 20% split of a few hundred patients can
        easily land with a badly skewed event rate.
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
            months = np.median(censored) / self._PER_YEAR[self.unit] * 12.0
            parts.append(f"median follow-up {months:.1f} months")
        return " | ".join(parts)

    def __repr__(self) -> str:
        parts = [f"time={self.time!r}", f"event={self.event!r}", f"event_value={self.event_values}"]
        if len(self._row_of):
            parts.append(self.summarise(list(self._row_of)))
        return f"{type(self).__name__}({' | '.join(parts)})"
