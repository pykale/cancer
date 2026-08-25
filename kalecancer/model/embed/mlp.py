"""A small feed-forward embedder for fixed-width modalities.

The counterpart of :class:`~kalecancer.model.embed.AttentionMIL` for data that is
already one vector per sample -- a clinical table, or a frozen representation from a
foundation model that now needs projecting into a shared fusion space.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


class MLPEmbedder(nn.Module):
    """Project a fixed-width modality, satisfying the :class:`Embedder` protocol.

    Args:
        in_dim: Width of the incoming features.
        out_dim: Width of the representation.
        hidden_dims: Widths of any hidden layers. Empty gives a single linear layer.
        dropout: Dropout applied after each hidden activation.

    Raises:
        ValueError: If any layer width is not positive.
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
        super().__init__()
        widths = [in_dim, *hidden_dims, out_dim]
        if any(width < 1 for width in widths):
            raise ValueError(f"every layer width must be positive, got {widths}")
        self.out_dim = out_dim

        layers: list[nn.Module] = []
        width = in_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Embed a batch of ``(batch, in_dim)`` features. ``mask`` is accepted and ignored."""
        return self.net(x)
