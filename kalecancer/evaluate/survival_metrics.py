"""Time-dependent AUC, integrated Brier score, and KM stratification.

These metrics live in ``kalecancer/evaluate/`` rather than
``kalecancer/survival/`` on purpose: unlike Harrell's C-index, they need
IPCW (inverse probability of censoring weighting) censoring weights,
which is exactly the kind of established, non-trivial statistical
machinery ``kalecancer/survival/`` is quarantined from depending on (see
that package's docstring) so it can migrate cleanly into PyKale core.
Here, outside the quarantine, we lean on ``scikit-survival`` instead of
reimplementing IPCW.

A consequence worth knowing: every ``eval_times`` grid passed to
:func:`time_dependent_auc` or :func:`integrated_brier` must lie strictly
inside the test set's observed follow-up range (min and max time),
otherwise the underlying ``scikit-survival`` call raises ``ValueError`` --
IPCW weights are undefined for times beyond what was actually observed.
"""

from __future__ import annotations

import numpy as np
from sksurv.compare import compare_survival
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator


def usable_eval_times(eval_times: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Keep only evaluation times strictly inside the observed follow-up.

    IPCW weights are undefined beyond what was observed, so horizons outside the
    range are dropped rather than extrapolated. An empty ``times`` describes no
    follow-up at all, so nothing is usable.
    """
    eval_times = np.asarray(eval_times, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if times.size == 0:
        return eval_times[:0]
    return eval_times[(eval_times > times.min()) & (eval_times < times.max())]


def _to_structured(times: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Build the ``(event: bool, time: float)`` structured array scikit-survival expects."""
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=bool)
    if times.shape != events.shape:
        raise ValueError(f"times and events must share shape, got {times.shape} and {events.shape}")

    structured = np.empty(times.shape[0], dtype=[("event", bool), ("time", np.float64)])
    structured["event"] = events
    structured["time"] = times
    return structured


def time_dependent_auc(
    train_times: np.ndarray,
    train_events: np.ndarray,
    test_times: np.ndarray,
    test_events: np.ndarray,
    risk: np.ndarray,
    eval_times: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Time-dependent (cumulative/dynamic) IPCW AUC.

    Thin wrapper around ``sksurv.metrics.cumulative_dynamic_auc``. The
    training set's event distribution is used only to fit the IPCW
    censoring weights; ``risk`` ranks the test set.

    Args:
        train_times: Training observed times, used for IPCW weighting.
        train_events: Training event indicators.
        test_times: Test observed times.
        test_events: Test event indicators.
        risk: Risk score for the test set, shape ``(N_test,)``.
        eval_times: Times to evaluate AUC at; must lie strictly inside the
            test set's observed follow-up range.

    Returns:
        A tuple ``(auc_per_time, mean_auc)``.
    """
    survival_train = _to_structured(train_times, train_events)
    survival_test = _to_structured(test_times, test_events)
    auc_per_time, mean_auc = cumulative_dynamic_auc(survival_train, survival_test, risk, eval_times)
    return auc_per_time, float(mean_auc)


def integrated_brier(
    train_times: np.ndarray,
    train_events: np.ndarray,
    test_times: np.ndarray,
    test_events: np.ndarray,
    survival_probs: np.ndarray,
    eval_times: np.ndarray,
) -> float:
    """Integrated Brier score (lower is better).

    Thin wrapper around ``sksurv.metrics.integrated_brier_score``.

    Args:
        train_times: Training observed times, used for IPCW weighting.
        train_events: Training event indicators.
        test_times: Test observed times.
        test_events: Test event indicators.
        survival_probs: Predicted survival probabilities for the test set,
            shape ``(N_test, len(eval_times))`` -- the matrix returned by
            ``kalecancer.evaluate.survival_metrics.predict_survival_function``.
        eval_times: Times the survival probabilities were evaluated at;
            must lie strictly inside the test set's observed follow-up range.

    Returns:
        The integrated Brier score.
    """
    survival_train = _to_structured(train_times, train_events)
    survival_test = _to_structured(test_times, test_events)
    return float(integrated_brier_score(survival_train, survival_test, survival_probs, eval_times))


def kaplan_meier_groups(
    risk: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    n_groups: int = 2,
) -> dict:
    """Split subjects into risk-quantile groups and compare their KM curves.

    Args:
        risk: Risk score, shape ``(N,)``.
        times: Observed times, shape ``(N,)``.
        events: Event indicators, shape ``(N,)``.
        n_groups: Number of equal-sized risk quantile groups.

    Returns:
        A dict with:

        * ``"group"``: integer array, shape ``(N,)``, 0 = lowest risk
          quantile through ``n_groups - 1`` = highest.
        * ``"km_curves"``: dict mapping group index to
          ``{"time": ..., "survival": ...}`` Kaplan-Meier curves.
        * ``"log_rank_p_value"``: p-value of the log-rank test
          (``sksurv.compare.compare_survival``) for a difference in
          survival across all groups.

    Raises:
        ValueError: If ``n_groups`` is below 2, or the risk scores cannot be
            divided into that many non-empty groups.
    """
    risk = np.asarray(risk)
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=bool)
    if n_groups < 2:
        raise ValueError(f"n_groups must be at least 2 to compare curves, got {n_groups}")

    quantile_edges = np.quantile(risk, np.linspace(0.0, 1.0, n_groups + 1))
    group = np.clip(np.digitize(risk, quantile_edges[1:-1], right=True), 0, n_groups - 1)

    # Quantile edges collapse when risk is near-constant or the cohort is smaller than
    # the requested number of groups, leaving a group with no members. The log-rank
    # test then fails deep inside scikit-survival, so it is caught here instead, where
    # the cause can be named.
    occupied = np.bincount(group, minlength=n_groups)
    if np.any(occupied == 0):
        raise ValueError(
            f"cannot form {n_groups} non-empty risk groups from {risk.shape[0]} subjects with "
            f"{np.unique(risk).size} distinct risk scores; group sizes would be {occupied.tolist()}"
        )

    structured = _to_structured(times, events)
    _, p_value = compare_survival(structured, group)

    km_curves = {}
    for g in range(n_groups):
        mask = group == g
        km_time, km_survival = kaplan_meier_estimator(events[mask], times[mask])
        km_curves[g] = {"time": km_time, "survival": km_survival}

    return {
        "group": group,
        "km_curves": km_curves,
        "log_rank_p_value": float(p_value),
    }


# --------------------------------------------------------------------------- #
# Discrimination
# --------------------------------------------------------------------------- #


def concordance_index(risk: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    """Harrell's concordance index (C-index).

    A pair ``(i, j)`` is comparable when subject ``i`` had an observed
    event and its time is strictly earlier than subject ``j``'s
    (``t_i < t_j``) -- ``j`` may itself be censored or not, its later time
    is known regardless. Among comparable pairs, a pair is concordant if
    the model ranked ``i`` (the earlier event) as higher risk than ``j``;
    ties in risk score count as half a concordant pair. A censored subject
    can only ever be the later member ``j`` of a pair, never the earlier
    reference ``i``.

    Computed via one ``(N, N)`` vectorised broadcast; no Python-level
    ``O(N^2)`` loop.

    Args:
        risk: Predicted risk score, higher means higher hazard, shape
            ``(N,)`` or ``(N, 1)``.
        times: Observed (event or censoring) time, shape ``(N,)``.
        events: ``True`` where the event was observed, ``False`` if
            censored, shape ``(N,)``.

    Returns:
        The concordance index in ``[0, 1]``.

    Raises:
        TypeError: If dtypes are wrong (``risk``/``times`` not floating
            point, ``events`` not bool).
        ValueError: If shapes are inconsistent or there is no comparable
            pair.
    """
    risk = np.asarray(risk)
    times = np.asarray(times)
    events = np.asarray(events)

    if risk.ndim == 2 and risk.shape[1] == 1:
        risk = risk.squeeze(-1)
    if risk.ndim != 1:
        raise ValueError(f"risk must have shape (N,) or (N, 1), got {risk.shape}")
    if times.ndim != 1:
        raise ValueError(f"times must have shape (N,), got {times.shape}")
    if events.ndim != 1:
        raise ValueError(f"events must have shape (N,), got {events.shape}")
    if not (risk.shape[0] == times.shape[0] == events.shape[0]):
        raise ValueError(
            f"risk, times and events must share length, got {risk.shape[0]}, {times.shape[0]}, {events.shape[0]}"
        )
    if not np.issubdtype(risk.dtype, np.floating):
        raise TypeError(f"risk must be a floating-point array, got dtype {risk.dtype}")
    if not np.issubdtype(times.dtype, np.floating):
        raise TypeError(f"times must be a floating-point array, got dtype {times.dtype}")
    if events.dtype != np.bool_:
        raise TypeError(f"events must be a bool array, got dtype {events.dtype}")

    # comparable[i, j]: subject i had an event and is strictly earlier than j.
    comparable = events[:, None] & (times[:, None] < times[None, :])
    n_comparable = comparable.sum()
    if n_comparable == 0:
        raise ValueError("concordance_index requires at least one comparable pair")

    concordant = (risk[:, None] > risk[None, :]).astype(np.float64)
    tied = (risk[:, None] == risk[None, :]).astype(np.float64) * 0.5
    score = np.where(comparable, concordant + tied, 0.0)

    return float(score.sum() / n_comparable)


# --------------------------------------------------------------------------- #
# The baseline hazard, and the survival curves it produces
# --------------------------------------------------------------------------- #
#
# A Cox model predicts a relative risk, not a probability. Turning one into the other
# needs a baseline hazard, which is estimated -- from the *training* split only, or
# the estimate leaks the split it is used to score.


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
