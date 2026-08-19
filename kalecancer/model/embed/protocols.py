"""What the rest of the model requires of a modality embedder.

Imports nothing from ``kalecancer``: an embedder takes tensors and returns tensors,
and knows nothing about cohorts, views or targets. That is what makes two of them
interchangeable, and what makes one testable with ``torch.randn``.

Whatever a particular embedder needs beyond the features -- TabICL's in-context
examples, a tile encoder's checkpoint -- it is given at construction, by the caller.
"""

from __future__ import annotations

from typing import Protocol

from torch import Tensor


class Embedder(Protocol):
    """One modality in, one vector per sample out.

    Implementations are ``nn.Module``s, so device moves, ``state_dict``, freezing and
    ``nn.ModuleDict`` come from torch rather than from anything defined here.
    """

    #: Width of the representation. Known at construction, because the projection
    #: layer that consumes it is built in the same breath.
    out_dim: int

    #: Whether a whole split should go through in one call rather than in mini-batches.
    #: Why varies by embedder -- ``TabICLEmbedder`` re-embeds its entire context on
    #: every call -- so a training loop should honour the flag and run these at
    #: ``batch_size="full"`` rather than reason about the cause.
    needs_full_batch: bool

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Embed a batch.

        Args:
            x (Tensor): ``(B, ...)`` features for one modality. ``(B, d)`` for a
                table, ``(B, n_tiles, d)`` for a bag of tiles.
            mask (Tensor | None): ``(B, n)`` marking real entries of a padded bag.
                Fixed-width modalities accept and ignore it, so one call site serves
                every embedder.

        Returns:
            Tensor: ``(B, out_dim)``.
        """
        ...
