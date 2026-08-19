"""Time-to-event analysis: Cox head, losses, and metrics."""

from .baseline import breslow_baseline_hazard, predict_survival_function
from .cox import CoxHead, neg_partial_log_likelihood
from .metrics import concordance_index
from .synthetic import SyntheticSurvival, make_synthetic_survival
from .trainer import fit_survival_model
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
    "CoxHead",
    "SyntheticSurvival",
    "breslow_baseline_hazard",
    "concordance_index",
    "fit_survival_model",
    "make_synthetic_survival",
    "neg_partial_log_likelihood",
    "predict_survival_function",
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
