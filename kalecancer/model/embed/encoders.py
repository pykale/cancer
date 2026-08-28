"""Layers adapted to the embedder contract.

An embedder is one modality in, one vector per patient out. The transformation is a
block from :mod:`kalecancer.model.layers`; what is added here is the contract fusion
relies on -- a declared ``out_dim``, a ``needs_full_batch`` flag, and a ``forward``
that accepts the mask a ragged modality carries and a fixed-width one ignores.

Keeping the two apart is what lets a block be reused outside this pipeline, and what
lets an embedder be swapped without touching the block it wraps. See
:mod:`kalecancer.utils.protocols` for the contract itself.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from kalecancer.model.layers import MLP, AttentionMIL


class MLPEmbedder(MLP):
    """A feed-forward stack, satisfying the embedder contract.

    The counterpart of :class:`BagEncoder` for data that is already one vector per
    patient -- a clinical table, or a frozen foundation-model representation that now
    needs projecting into the shared fusion space.

    Args:
        in_dim: Width of the incoming features.
        out_dim: Width of the representation.
        hidden_dims: Widths of any hidden layers. Empty gives a single linear layer.
        dropout: Dropout applied after each hidden activation.
    """

    #: Rows are independent, so mini-batching changes nothing.
    needs_full_batch = False

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
    ) -> None:
        super().__init__(in_dim, out_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.out_dim = out_dim

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Embed ``(batch, in_dim)`` features. ``mask`` is accepted and ignored."""
        return super().forward(x)


class BagEncoder(nn.Module):
    """Adapt a bag-pooling module to the :class:`~kalecancer.model.embed.Embedder` interface.

    Multimodal fusion expects each modality to yield one vector per patient, whereas
    :class:`AttentionMIL` consumes a list of variable-length bags and also returns
    attention. This wrapper keeps the last attention weights available for
    interpretation while presenting the plain embedder signature fusion needs.

    Args:
        mil: The bag-pooling module to wrap.
    """

    #: Bags are pooled independently, so mini-batching changes nothing.
    needs_full_batch = False

    def __init__(self, mil: AttentionMIL) -> None:
        super().__init__()
        self.mil = mil
        self.out_dim = mil.output_dim
        self.last_attention: list[torch.Tensor] = []

    @property
    def output_dim(self) -> int:
        """Alias of :attr:`out_dim`, matching :class:`AttentionMIL`."""
        return self.out_dim

    def forward(self, bags: list[torch.Tensor]) -> torch.Tensor:
        embeddings, attention = self.mil.forward_bags(bags)
        # Detached: holding these on the module would otherwise keep each bag's
        # activations alive between forward passes.
        self.last_attention = [weights.detach() for weights in attention]
        return embeddings
