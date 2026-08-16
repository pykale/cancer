"""Time-to-event analysis: Cox head, losses, and metrics.

This submodule is destined to be refactored into PyKale core later. It MUST NOT
import anything cancer-specific — only ``torch``, ``numpy``, and ``pykale``.
"""

from .baseline import breslow_baseline_hazard, predict_survival_function
from .cox import CoxHead, neg_partial_log_likelihood
from .metrics import concordance_index
from .synthetic import SyntheticSurvival, make_synthetic_survival
from .trainer import fit_survival_model

__all__ = [
    "CoxHead",
    "SyntheticSurvival",
    "breslow_baseline_hazard",
    "concordance_index",
    "fit_survival_model",
    "make_synthetic_survival",
    "neg_partial_log_likelihood",
    "predict_survival_function",
]
