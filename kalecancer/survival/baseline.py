"""Breslow baseline hazard and absolute survival probabilities.

A Cox model only outputs a *relative* risk (``exp(log_hazard)``): partial
likelihood cancels the baseline hazard, so training never estimates it.
But integrated Brier score and calibration need *absolute* survival
probabilities ``S(t|x)``, which require that baseline back. This module
estimates it (Breslow's estimator) from a TRAINING set's risk scores and
event times, then combines it with any (e.g. held-out) risk score to
produce ``S(t|x) = exp(-H0(t) * exp(log_hazard))``.

Boundary rules: this module imports only ``numpy`` (and stdlib).
"""

from __future__ import annotations

import numpy as np


def breslow_baseline_hazard(
    log_hazard: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Breslow estimator of the cumulative baseline hazard.

    At each distinct event time ``t``, the hazard increment is
    ``d_t / R(t)``, where ``d_t`` is the number of events at ``t`` and
    ``R(t) = sum(exp(log_hazard))`` over the risk set (everyone with
    ``time >= t``). ``H0(t)`` is the running sum of these increments.

    Numerical stability: computed after subtracting a single global
    ``max(log_hazard)`` shift before exponentiating, so no intermediate
    sum can overflow; the shift is re-applied exactly (not an
    approximation) when forming the increments.

    Args:
        log_hazard: Risk score from the TRAINING set, shape ``(N,)`` or ``(N, 1)``.
        times: Observed (event or censoring) time, shape ``(N,)``.
        events: ``True`` where the event was observed, shape ``(N,)``.

    Returns:
        A tuple ``(event_times, cumulative_baseline_hazard)``, both sorted
        ascending by time and restricted to distinct times at which at
        least one event occurred.

    Raises:
        TypeError: If dtypes are wrong (``log_hazard``/``times`` not
            floating point, ``events`` not bool).
        ValueError: If shapes are inconsistent or no event is observed.
    """
    log_hazard = np.asarray(log_hazard)
    times = np.asarray(times)
    events = np.asarray(events)

    if log_hazard.ndim == 2 and log_hazard.shape[1] == 1:
        log_hazard = log_hazard.squeeze(-1)
    if log_hazard.ndim != 1:
        raise ValueError(f"log_hazard must have shape (N,) or (N, 1), got {log_hazard.shape}")
    if times.ndim != 1:
        raise ValueError(f"times must have shape (N,), got {times.shape}")
    if events.ndim != 1:
        raise ValueError(f"events must have shape (N,), got {events.shape}")
    if not (log_hazard.shape[0] == times.shape[0] == events.shape[0]):
        raise ValueError(
            f"log_hazard, times and events must share length, "
            f"got {log_hazard.shape[0]}, {times.shape[0]}, {events.shape[0]}"
        )
    if not np.issubdtype(log_hazard.dtype, np.floating):
        raise TypeError(f"log_hazard must be a floating-point array, got dtype {log_hazard.dtype}")
    if not np.issubdtype(times.dtype, np.floating):
        raise TypeError(f"times must be a floating-point array, got dtype {times.dtype}")
    if events.dtype != np.bool_:
        raise TypeError(f"events must be a bool array, got dtype {events.dtype}")
    if not np.any(events):
        raise ValueError("breslow_baseline_hazard requires at least one observed event; all samples are censored")

    order = np.argsort(times)[::-1]
    s_log_hazard = log_hazard[order]
    s_times = times[order]
    s_events = events[order]

    shift = s_log_hazard.max()
    exp_shifted = np.exp(s_log_hazard - shift)
    risk_cumsum_shifted = np.cumsum(exp_shifted)

    # Ties are adjacent after the descending sort; group boundaries are
    # exactly the positions where the time changes.
    is_group_start = np.empty(s_times.shape[0], dtype=bool)
    is_group_start[0] = True
    is_group_start[1:] = s_times[1:] != s_times[:-1]
    group_start_idx = np.nonzero(is_group_start)[0]
    group_id = np.cumsum(is_group_start) - 1
    # A tie group's risk-set sum is only complete at its LAST index (every
    # member, event or censored, sharing that time must be folded in).
    last_idx = np.concatenate([group_start_idx[1:] - 1, [s_times.shape[0] - 1]])

    group_times = s_times[group_start_idx]
    group_risk_shifted = risk_cumsum_shifted[last_idx]
    d_k = np.zeros(group_start_idx.shape[0])
    np.add.at(d_k, group_id, s_events.astype(np.float64))

    event_mask = d_k > 0
    event_times_desc = group_times[event_mask]
    # d_k / R_k = (d_k / group_risk_shifted) * exp(-shift), since
    # group_risk_shifted = R_k * exp(-shift).
    increments_desc = (d_k[event_mask] / group_risk_shifted[event_mask]) * np.exp(-shift)

    event_times = event_times_desc[::-1]
    cumulative_baseline_hazard = np.cumsum(increments_desc[::-1])

    return event_times, cumulative_baseline_hazard


def predict_survival_function(
    log_hazard: np.ndarray,
    event_times: np.ndarray,
    cumulative_baseline_hazard: np.ndarray,
    eval_times: np.ndarray,
) -> np.ndarray:
    """Predict absolute survival probabilities from a fitted baseline.

    ``S(t|x) = exp(-H0(t) * exp(log_hazard))``, where ``H0`` is the
    right-continuous step function defined by ``(event_times,
    cumulative_baseline_hazard)``: its value at ``t`` is carried forward
    from the last event time ``<= t``, and is ``0`` (so ``S = 1``) before
    the first event time.

    Args:
        log_hazard: Risk score to predict for (e.g. held-out), shape
            ``(N,)`` or ``(N, 1)``.
        event_times: Ascending distinct event times from
            :func:`breslow_baseline_hazard`, shape ``(K,)``.
        cumulative_baseline_hazard: Matching cumulative baseline hazard,
            shape ``(K,)``.
        eval_times: Times to evaluate survival at, shape ``(T,)``. May
            include times before the first event time.

    Returns:
        Survival probabilities, shape ``(N, T)``.

    Raises:
        TypeError: If dtypes are wrong (any array not floating point).
        ValueError: If shapes are inconsistent, or ``event_times`` is not
            sorted ascending.
    """
    log_hazard = np.asarray(log_hazard)
    event_times = np.asarray(event_times)
    cumulative_baseline_hazard = np.asarray(cumulative_baseline_hazard)
    eval_times = np.asarray(eval_times)

    if log_hazard.ndim == 2 and log_hazard.shape[1] == 1:
        log_hazard = log_hazard.squeeze(-1)
    if log_hazard.ndim != 1:
        raise ValueError(f"log_hazard must have shape (N,) or (N, 1), got {log_hazard.shape}")
    if event_times.ndim != 1:
        raise ValueError(f"event_times must have shape (K,), got {event_times.shape}")
    if cumulative_baseline_hazard.ndim != 1:
        raise ValueError(f"cumulative_baseline_hazard must have shape (K,), got {cumulative_baseline_hazard.shape}")
    if eval_times.ndim != 1:
        raise ValueError(f"eval_times must have shape (T,), got {eval_times.shape}")
    if event_times.shape[0] != cumulative_baseline_hazard.shape[0]:
        raise ValueError(
            f"event_times and cumulative_baseline_hazard must share length, "
            f"got {event_times.shape[0]}, {cumulative_baseline_hazard.shape[0]}"
        )
    for name, array in (
        ("log_hazard", log_hazard),
        ("event_times", event_times),
        ("cumulative_baseline_hazard", cumulative_baseline_hazard),
        ("eval_times", eval_times),
    ):
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(f"{name} must be a floating-point array, got dtype {array.dtype}")
    if event_times.shape[0] > 1 and np.any(np.diff(event_times) < 0):
        raise ValueError("event_times must be sorted ascending")

    # Index of the last event time <= each eval time; -1 means "before the
    # first event", where the baseline hazard is 0 (H0 not yet accrued).
    idx = np.searchsorted(event_times, eval_times, side="right") - 1
    baseline_at_eval = np.where(idx < 0, 0.0, cumulative_baseline_hazard[np.clip(idx, 0, None)])

    hazard_ratio = np.exp(log_hazard)
    return np.exp(-np.outer(hazard_ratio, baseline_at_eval))
