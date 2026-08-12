"""Time-to-event analysis: Cox head, losses, and metrics.

This submodule is destined to be refactored into PyKale core later. It MUST NOT
import anything cancer-specific — only ``torch``, ``numpy``, and ``pykale``.

Label contract used throughout: ``duration`` is the time to event or censoring and
``event`` is ``1`` when the event was observed, ``0`` when the patient was censored.
Risk scores are log partial hazards, where higher means higher risk.
"""

from kalecancer.survival.baseline import BreslowBaselineHazard
from kalecancer.survival.cox import CoxHead
from kalecancer.survival.loss import as_event_mask, cox_ph_loss, has_risk_set
from kalecancer.survival.metrics import (
    brier_score,
    censoring_weights,
    concordance_index,
    survival_metrics,
    time_dependent_auc,
    usable_eval_times,
)

__all__ = [
    "BreslowBaselineHazard",
    "CoxHead",
    "as_event_mask",
    "brier_score",
    "censoring_weights",
    "concordance_index",
    "cox_ph_loss",
    "has_risk_set",
    "survival_metrics",
    "time_dependent_auc",
    "usable_eval_times",
]
