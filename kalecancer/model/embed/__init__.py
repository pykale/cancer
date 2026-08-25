"""Modality embedders (one modality in, one vector per sample out) and the fusion wrappers that combine them."""

from kalecancer.model.embed.attention_mil import AttentionMIL, BagEncoder, GatedAttention
from kalecancer.model.embed.mlp import MLPEmbedder
from kalecancer.model.embed.multimodal_fusion import (
    FUSION_METHODS,
    FUSION_STAGES,
    ConcatFusion,
    FusionBlock,
    LowRankFusion,
    MultimodalFusion,
    MultimodalOutput,
    ProductOfExpertsFusion,
    build_fusion,
    modality_dropout,
    multimodal_bce_loss,
    multimodal_cox_loss,
)
from kalecancer.model.embed.protocols import Embedder
from kalecancer.model.embed.tabicl import TabICLEmbedder

__all__ = [
    "FUSION_METHODS",
    "FUSION_STAGES",
    "AttentionMIL",
    "BagEncoder",
    "ConcatFusion",
    "Embedder",
    "FusionBlock",
    "GatedAttention",
    "LowRankFusion",
    "MLPEmbedder",
    "MultimodalFusion",
    "MultimodalOutput",
    "ProductOfExpertsFusion",
    "TabICLEmbedder",
    "build_fusion",
    "modality_dropout",
    "multimodal_bce_loss",
    "multimodal_cox_loss",
]
