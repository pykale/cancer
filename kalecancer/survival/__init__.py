"""Time-to-event analysis: Cox head, partial likelihood, baseline hazard, metrics.

This submodule is destined to be refactored into PyKale core later. It MUST NOT
import anything cancer-specific — only ``torch``, ``numpy``, and ``pykale``.
Censoring-weighted metrics, which need established survival-analysis machinery, live
in :mod:`kalecancer.evaluate.survival_metrics` instead.

Label contract used throughout: ``times`` is the time to event or censoring and
``events`` is ``True`` when the event was observed, ``False`` when censored. Risk
scores are log hazards, where higher means higher risk.
"""

from .baseline import breslow_baseline_hazard, predict_survival_function
from .cox import CoxHead, as_event_mask, has_risk_set, neg_partial_log_likelihood
from .metrics import concordance_index
from .survival_target import SurvivalTarget
from .synthetic import SyntheticSurvival, make_synthetic_survival
from .trainer import fit_survival_model

__all__ = [
    "CoxHead",
    "SurvivalTarget",
    "SyntheticSurvival",
    "as_event_mask",
    "breslow_baseline_hazard",
    "concordance_index",
    "fit_survival_model",
    "has_risk_set",
    "make_synthetic_survival",
    "neg_partial_log_likelihood",
    "predict_survival_function",
]
