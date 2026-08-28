"""Shared neural network building blocks.

Pieces small enough to be reused by more than one model, and independent enough to
be tested with ``torch.randn``. A block here transforms tensors and knows nothing
about modalities, cohorts or endpoints -- which is what separates it from
:mod:`kalecancer.model.embed`, where a block is adapted to the embedder contract.
"""

from kalecancer.model.layers.attention import AttentionMIL, GatedAttention
from kalecancer.model.layers.mlp import MLP

__all__ = ["MLP", "AttentionMIL", "GatedAttention"]
