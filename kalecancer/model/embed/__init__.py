"""Modality encoders and fusion wrappers."""

from kalecancer.model.embed.attention_mil import AttentionMIL, BagEncoder, GatedAttention
from kalecancer.model.embed.multimodal_fusion import (
    FUSION_METHODS,
    ConcatFusion,
    FusionBlock,
    LowRankFusion,
    ProductOfExpertsFusion,
    build_fusion,
)

__all__ = [
    "FUSION_METHODS",
    "AttentionMIL",
    "BagEncoder",
    "ConcatFusion",
    "FusionBlock",
    "GatedAttention",
    "LowRankFusion",
    "ProductOfExpertsFusion",
    "build_fusion",
]
