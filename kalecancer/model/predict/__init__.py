"""Task heads and the objectives they are trained with.

A head turns a fixed-width embedding into a score; a loss says whether that score was
right. They live together because neither is meaningful alone: the Cox head is
bias-free *because* of its partial likelihood, and the linear head keeps its bias
*because* cross-entropy is not shift-invariant.

Which pair a model gets is decided by its
:class:`~kalecancer.pipeline.task.PredictionTask`, so a new endpoint means a new
task, not a new trainer.
"""

from kalecancer.model.predict.heads import CoxHead, LinearHead
from kalecancer.model.predict.losses import (
    as_event_mask,
    binary_cross_entropy,
    has_risk_set,
    multimodal_bce_loss,
    multimodal_cox_loss,
    neg_partial_log_likelihood,
)

__all__ = [
    "CoxHead",
    "LinearHead",
    "as_event_mask",
    "binary_cross_entropy",
    "has_risk_set",
    "multimodal_bce_loss",
    "multimodal_cox_loss",
    "neg_partial_log_likelihood",
]
