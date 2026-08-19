"""Model stage: encoders, fusion, task heads, and shared layers."""

from kalecancer.model.multimodal import (
    FUSION_STRATEGIES,
    EarlyFusionSurvival,
    HybridFusionSurvival,
    LateFusionSurvival,
    MultimodalOutput,
    MultimodalSurvivalModel,
    build_multimodal_survival,
    modality_dropout,
    multimodal_cox_loss,
)

__all__ = [
    "FUSION_STRATEGIES",
    "EarlyFusionSurvival",
    "HybridFusionSurvival",
    "LateFusionSurvival",
    "MultimodalOutput",
    "MultimodalSurvivalModel",
    "build_multimodal_survival",
    "modality_dropout",
    "multimodal_cox_loss",
]
