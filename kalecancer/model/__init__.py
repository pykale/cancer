"""Model stage: modality embedders, multimodal fusion, task heads, and shared layers."""

from kalecancer.model.embed import (
    FUSION_METHODS,
    FUSION_STAGES,
    MultimodalFusion,
    MultimodalOutput,
    modality_dropout,
    multimodal_cox_loss,
)

__all__ = [
    "FUSION_METHODS",
    "FUSION_STAGES",
    "MultimodalFusion",
    "MultimodalOutput",
    "modality_dropout",
    "multimodal_cox_loss",
]
