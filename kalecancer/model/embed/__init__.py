"""Modality embedders, and the fusion that combines them.

An embedder is one modality in, one vector per patient out. The blocks doing the
transformation live in :mod:`kalecancer.model.layers`; what is added here is the
contract fusion relies on. Fusion then combines several such vectors into one,
whatever their modalities were.
"""

from kalecancer.model.embed.encoders import BagEncoder, MLPEmbedder
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
)
from kalecancer.model.embed.tabicl import TabICLEmbedder

__all__ = [
    "FUSION_METHODS",
    "FUSION_STAGES",
    "BagEncoder",
    "ConcatFusion",
    "FusionBlock",
    "LowRankFusion",
    "MLPEmbedder",
    "MultimodalFusion",
    "MultimodalOutput",
    "ProductOfExpertsFusion",
    "TabICLEmbedder",
    "build_fusion",
    "modality_dropout",
]
