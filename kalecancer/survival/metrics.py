"""Evaluation metrics for survival predictions.

Boundary rules: this module imports only ``numpy`` (and stdlib). Metrics
operate on plain arrays after training/inference, so there is no torch
dependency here.
"""

from __future__ import annotations

import numpy as np


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

"""Survival metrics that account for censoring.

Discrimination is measured with the concordance index; calibration with the Brier
score. Metrics that reweight for censoring (Uno's C-index, time-dependent AUC, the
Brier score) need an estimate of the censoring distribution. That estimate is taken
from the **training** set: deriving it from the data being scored would leak the test
outcome distribution into its own evaluation.
"""

from __future__ import annotations

import logging

import torch
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.ipcw import get_ipcw

from kalecancer.survival.baseline import BreslowBaselineHazard
from kalecancer.survival.loss import as_event_mask

logger = logging.getLogger(__name__)


def censoring_weights(
    train_event: torch.Tensor,
    train_time: torch.Tensor,
    new_time: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inverse probability of censoring weights from the training follow-up.

    Args:
        train_event: ``(n,)`` training event indicators.
        train_time: ``(n,)`` training times.
        new_time: Times at which to evaluate the weights. Defaults to ``train_time``.

    Returns:
        Weights aligned with ``new_time``.
    """
    return get_ipcw(
        as_event_mask(train_event),
        train_time.float(),
        new_time=None if new_time is None else new_time.float(),
    )


def concordance_index(
    risk: torch.Tensor,
    event: torch.Tensor,
    time: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> float:
    """Concordance index of a risk score.

    Args:
        risk: ``(n,)`` risk scores, higher meaning higher risk.
        event: ``(n,)`` event indicators.
        time: ``(n,)`` event or censoring times.
        weight: Optional censoring weights. Without them this is Harrell's C-index;
            with weights from :func:`censoring_weights` it is Uno's.

    Returns:
        Concordance in ``[0, 1]``; 0.5 is uninformative ranking.
    """
    return float(ConcordanceIndex()(risk, as_event_mask(event), time.float(), weight=weight))


def time_dependent_auc(
    risk: torch.Tensor,
    event: torch.Tensor,
    time: torch.Tensor,
    eval_times: torch.Tensor,
    weight: torch.Tensor | None = None,
    weight_eval_times: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cumulative/dynamic AUC at each of ``eval_times``."""
    return Auc()(
        risk,
        as_event_mask(event),
        time.float(),
        new_time=eval_times.float(),
        weight=weight,
        weight_new_time=weight_eval_times,
    )


def brier_score(
    survival_probabilities: torch.Tensor,
    event: torch.Tensor,
    time: torch.Tensor,
    eval_times: torch.Tensor,
    weight: torch.Tensor | None = None,
    weight_eval_times: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Brier score at each of ``eval_times`` and its integral.

    Args:
        survival_probabilities: ``(n, len(eval_times))`` predicted survival, e.g. from
            :meth:`~kalecancer.survival.baseline.BreslowBaselineHazard.survival_function`.
        event: ``(n,)`` event indicators.
        time: ``(n,)`` event or censoring times.
        eval_times: Times matching the columns of ``survival_probabilities``.
        weight: Optional censoring weights at ``time``.
        weight_eval_times: Optional censoring weights at ``eval_times``.

    Returns:
        Per-time Brier scores (lower is better) and the integrated Brier score.
    """
    metric = BrierScore()
    scores = metric(
        survival_probabilities,
        as_event_mask(event),
        time.float(),
        new_time=eval_times.float(),
        weight=weight,
        weight_new_time=weight_eval_times,
    )
    return scores, float(metric.integral())


def usable_eval_times(eval_times: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Keep only evaluation times covered by the observed follow-up.

    Time-dependent metrics are undefined beyond the last follow-up time, so horizons
    outside the observed range are dropped rather than extrapolated.
    """
    eval_times = eval_times.float().flatten()
    return eval_times[(eval_times > float(time.min())) & (eval_times < float(time.max()))]


def survival_metrics(
    risk: torch.Tensor,
    event: torch.Tensor,
    time: torch.Tensor,
    train_risk: torch.Tensor | None = None,
    train_event: torch.Tensor | None = None,
    train_time: torch.Tensor | None = None,
    eval_times: torch.Tensor | None = None,
) -> dict[str, float | dict[str, float]]:
    """Evaluate a set of risk predictions.

    Harrell's C-index is always reported. The censoring-weighted metrics are added
    only when training outcomes are supplied, since they need a leakage-free estimate
    of the censoring distribution; the Brier score additionally needs ``train_risk``
    to fit the baseline hazard.

    Args:
        risk: ``(n,)`` predicted risk scores for the evaluated split.
        event: ``(n,)`` event indicators for the evaluated split.
        time: ``(n,)`` times for the evaluated split.
        train_risk: ``(m,)`` predicted risks on the training split.
        train_event: ``(m,)`` training event indicators.
        train_time: ``(m,)`` training times.
        eval_times: Horizons for the time-dependent metrics, in the same unit as
            ``time``.

    Returns:
        Metric names mapped to values; time-dependent entries are keyed by horizon.
    """
    num_events = int(as_event_mask(event).sum())
    metrics: dict[str, float | dict[str, float]] = {
        "num_patients": float(len(risk)),
        "num_events": float(num_events),
    }

    # Every metric here is defined by comparisons against observed events, so a split
    # without any is reported as undefined rather than raising.
    if num_events == 0:
        logger.warning("no observed events in this split; survival metrics are undefined")
        return metrics

    metrics["c_index"] = concordance_index(risk, event, time)

    if train_event is None or train_time is None:
        return metrics
    if not bool(as_event_mask(train_event).any()):
        logger.warning("no observed events in the training split; skipping censoring-weighted metrics")
        return metrics

    weight = censoring_weights(train_event, train_time, new_time=time)
    metrics["c_index_ipcw"] = concordance_index(risk, event, time, weight=weight)

    if eval_times is None:
        return metrics

    horizons = usable_eval_times(eval_times, time)
    if horizons.numel() == 0:
        logger.warning(
            "none of the evaluation horizons %s fall inside the observed follow-up (%.0f to %.0f); "
            "skipping the time-dependent metrics",
            [float(t) for t in eval_times.flatten()],
            float(time.min()),
            float(time.max()),
        )
        return metrics

    weight_horizons = censoring_weights(train_event, train_time, new_time=horizons)
    auc = time_dependent_auc(risk, event, time, horizons, weight=weight, weight_eval_times=weight_horizons)
    metrics["auc"] = {f"{float(t):.0f}": float(value) for t, value in zip(horizons, auc, strict=True)}

    if train_risk is None:
        return metrics

    baseline = BreslowBaselineHazard().fit(train_risk, train_event, train_time)
    survival = baseline.survival_function(risk, horizons)
    scores, integral = brier_score(survival, event, time, horizons, weight=weight, weight_eval_times=weight_horizons)
    metrics["brier"] = {f"{float(t):.0f}": float(value) for t, value in zip(horizons, scores, strict=True)}
    metrics["integrated_brier_score"] = integral
    return metrics
