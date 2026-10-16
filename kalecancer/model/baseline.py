"""Breslow baseline hazard: turns the relative risks of a Cox model into survival probabilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _vector(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (n,), got {array.shape}")
    return array


class BreslowBaseline:
    """Breslow estimate of the cumulative baseline hazard H0 of a Cox model.

    A Cox model predicts relative risks only; the survival of a patient with log-hazard ``x`` is
    ``S(t | x) = exp(-H0(t) exp(x))``. H0 jumps by ``d / sum(exp(x) over the patients still at risk)`` at each
    distinct event time, where ``d`` patients have the event. It is a right-continuous step function: 0 before the
    first event time, and constant after the last one.

    Attributes:
        event_times_: The distinct event times of the patients it was fitted on, ascending.
        cumulative_hazard_: H0 at each of ``event_times_``.
    """

    event_times_: np.ndarray
    cumulative_hazard_: np.ndarray

    def fit(self, log_hazard: ArrayLike, time: ArrayLike, event: ArrayLike) -> BreslowBaseline:
        """Estimate H0 from the log-hazards a model gives the patients, and their outcomes.

        With no event, H0 is 0 everywhere, so every predicted survival is 1.

        Args:
            log_hazard: The model's log-hazard per patient, finite.
            time: Follow-up time per patient, finite.
            event: ``True`` where the event was observed.

        Returns:
            The estimator itself.
        """
        log_hazard, time = _vector(log_hazard, "log_hazard"), _vector(time, "time")
        event = np.asarray(event)
        if event.dtype != np.bool_ or event.shape != time.shape or log_hazard.shape != time.shape:
            raise ValueError(
                f"log_hazard, time and a bool event must share one length, got shapes {log_hazard.shape}, "
                f"{time.shape} and {event.shape} (event dtype {event.dtype})"
            )
        if not (np.isfinite(log_hazard).all() and np.isfinite(time).all()):
            raise ValueError("log_hazard and time must be finite")

        # descending time, so the running sum over a prefix is the risk set of the time where the prefix ends
        order = np.argsort(-time, kind="stable")
        time, event, log_hazard = time[order], event[order], log_hazard[order]
        # H0 is kept as H0 * exp(shift), and survival subtracts the shift from the log-hazard it is given, so exp
        # overflows neither here nor there for log-hazards near the fitted ones
        shift = float(log_hazard.max()) if len(log_hazard) else 0.0
        risk_set = np.cumsum(np.exp(log_hazard - shift))

        starts = np.flatnonzero(np.r_[True, time[1:] != time[:-1]])
        # tied patients are all at risk at their time, so a group's risk set is complete only at its last member
        ends = np.r_[starts[1:], len(time)] - 1
        deaths = np.add.reduceat(event.astype(np.float64), starts) if len(time) else np.zeros(0)
        has_event = deaths > 0
        increments = deaths[has_event] / risk_set[ends][has_event]

        self.event_times_ = time[starts][has_event][::-1]
        self._shift = shift
        self._scaled_hazard = np.cumsum(increments[::-1])
        self.cumulative_hazard_ = self._scaled_hazard * np.exp(-shift)
        return self

    def survival(self, log_hazard: ArrayLike, times: ArrayLike) -> np.ndarray:
        """Survival probabilities ``S(t | x)`` for each log-hazard ``x`` at each of ``times``.

        Args:
            log_hazard: Log-hazards, NaN where a patient has no prediction (their survival is NaN too).
            times: The times to evaluate at.

        Returns:
            An array of shape ``(len(log_hazard), len(times))``.
        """
        if not hasattr(self, "cumulative_hazard_"):
            raise RuntimeError("this BreslowBaseline is not fitted; call fit first")
        log_hazard, times = _vector(log_hazard, "log_hazard"), _vector(times, "times")
        # the number of event times <= t indexes H0 with a leading 0 for the times before the first event
        baseline = np.r_[0.0, self._scaled_hazard][np.searchsorted(self.event_times_, times, side="right")]
        return np.exp(-np.outer(np.exp(log_hazard - self._shift), baseline))

    def __repr__(self) -> str:
        fitted = f"{len(self.event_times_)} event times" if hasattr(self, "event_times_") else "not fitted"
        return f"BreslowBaseline({fitted})"
