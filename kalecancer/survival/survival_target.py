"""Supervision targets, keyed by sample identifier.

A target is bound once, at load time, against the table that carries it. It is
keyed by identifier rather than row position, which is what lets a dataset be
subset and recombined without the target ever needing realignment.

Nothing here is cancer-specific or tabular-specific: ``SurvivalTarget`` is the
piece most likely to move into PyKale core alongside the survival heads, losses
and metrics that consume it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch


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
        ValueError: On :meth:`bind`, if the columns are missing, times or statuses
            are invalid or absent, or the resulting event rate is 0% or 100% -- all
            of which indicate a mis-specified target rather than an unusual cohort.
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
        self._times: dict = {}
        self._events: dict = {}

    def bind(self, frame: pd.DataFrame, identifier: str) -> None:
        """Extract and validate times and events, keyed by identifier.

        Args:
            frame (pd.DataFrame): Table containing the target columns.
            identifier (str): Column holding sample identifiers.

        Raises:
            ValueError: On missing columns, non-numeric or negative times, a missing
                event status, or a degenerate event rate.
        """
        missing = [c for c in (self.time, self.event) if c not in frame.columns]
        if missing:
            raise ValueError(f"Target column(s) {missing} not found. Available: {list(frame.columns)}")

        times = pd.to_numeric(frame[self.time], errors="coerce")
        if times.isna().any():
            bad = frame.loc[times.isna(), identifier].tolist()[:5]
            raise ValueError(
                f"Column '{self.time}' has {int(times.isna().sum())} missing or "
                f"non-numeric value(s), e.g. for {bad}. Survival time must be known "
                f"for every sample; drop or impute these before constructing the dataset."
            )
        if (times < 0).any():
            raise ValueError(f"Column '{self.time}' contains negative values.")

        observed = frame[self.event]
        unknown = observed.isna()
        if unknown.any():
            bad = frame.loc[unknown, identifier].tolist()[:5]
            raise ValueError(
                f"Column '{self.event}' has {int(unknown.sum())} missing value(s), e.g. for "
                f"{bad}. An unknown outcome is not a censored one: censoring asserts the "
                f"sample was event-free throughout its recorded time, which an absent status "
                f"does not establish. Treating these as censored would drop events from the "
                f"numerator while keeping their follow-up in the denominator, biasing every "
                f"estimate towards the null. Decide explicitly before constructing the "
                f"dataset -- drop these rows, or recode them as censored if that is what a "
                f"blank means in your data dictionary."
            )

        events = observed.isin(self.event_values)
        rate = float(events.mean())
        if rate in (0.0, 1.0):
            raise ValueError(
                f"event_value={self.event_values} matches {rate:.0%} of rows in column "
                f"'{self.event}', which cannot be right. Observed values: "
                f"{sorted(observed.dropna().unique().tolist())[:10]}"
            )

        ids = frame[identifier].tolist()
        self._times = dict(zip(ids, times.astype(float), strict=False))
        self._events = dict(zip(ids, events.astype(float), strict=False))

    def for_(self, identifier) -> dict:
        """Return ``{"time": Tensor, "event": Tensor}`` for one sample."""
        return {
            "time": torch.tensor(self._times[identifier], dtype=torch.float32),
            "event": torch.tensor(self._events[identifier], dtype=torch.float32),
        }

    def events_for(self, identifiers: Sequence) -> np.ndarray:
        """Event indicators for ``identifiers``, as a float array. Used for stratifying."""
        return np.array([self._events[i] for i in identifiers], dtype=float)

    def times_for(self, identifiers: Sequence) -> np.ndarray:
        """Follow-up times for ``identifiers``, as a float array."""
        return np.array([self._times[i] for i in identifiers], dtype=float)

    def summarise(self, identifiers: Sequence) -> str:
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
