"""Modality embedders (one modality in, one vector per sample out) and the fusion wrappers that combine them."""

from kalecancer.model.embed.attention_mil import AttentionMIL, BagEncoder, GatedAttention
from kalecancer.model.embed.multimodal_fusion import (
    FUSION_METHODS,
    ConcatFusion,
    FusionBlock,
    LowRankFusion,
    ProductOfExpertsFusion,
    build_fusion,
)
from kalecancer.model.embed.protocols import Embedder
from kalecancer.model.embed.tabicl import TabICLEmbedder

__all__ = [
    "FUSION_METHODS",
    "AttentionMIL",
    "BagEncoder",
    "ConcatFusion",
    "Embedder",
    "FusionBlock",
    "GatedAttention",
    "LowRankFusion",
    "ProductOfExpertsFusion",
    "TabICLEmbedder",
    "build_fusion",
]
