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
            ``kalecancer.survival.baseline.predict_survival_function``.
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
