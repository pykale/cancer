"""The model stage, in three parts.

``layers`` holds blocks that transform tensors, ``embed`` adapts them to the contract
fusion relies on and combines several modalities into one representation, and
``predict`` turns that representation into a score with the loss that scores it.

A class belongs to exactly one: a block that knows no modality is a layer, a thing
with ``out_dim`` is an embedder, a thing emitting one number per patient is a head.
"""

from kalecancer.model.embed import FUSION_METHODS, FUSION_STAGES, MultimodalFusion, MultimodalOutput
from kalecancer.model.layers import MLP, AttentionMIL, GatedAttention
from kalecancer.model.predict import CoxHead, LinearHead

__all__ = [
    "AttentionMIL",
    "CoxHead",
    "FUSION_METHODS",
    "FUSION_STAGES",
    "GatedAttention",
    "LinearHead",
    "MLP",
    "MultimodalFusion",
    "MultimodalOutput",
]
