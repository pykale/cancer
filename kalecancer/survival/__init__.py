"""Time-to-event analysis: Cox head, losses, and metrics.

This submodule is destined to be refactored into PyKale core later. It MUST NOT
import anything cancer-specific — only ``torch``, ``numpy``, and ``pykale``.
"""

from .cox import CoxHead, neg_partial_log_likelihood
from .metrics import concordance_index
from .synthetic import SyntheticSurvival, make_synthetic_survival

__all__ = [
    "CoxHead",
    "SyntheticSurvival",
    "concordance_index",
    "make_synthetic_survival",
    "neg_partial_log_likelihood",
]
