"""A feed-forward stack.

The plainest building block here: linear layers with an activation and dropout
between them, and nothing else. Adapting it to the embedder contract -- declaring an
output width, accepting the mask a bag modality needs -- is
:class:`~kalecancer.model.embed.MLPEmbedder`'s job, not this one's.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


class MLP(nn.Module):
    """Linear layers with ReLU and dropout between them.

    Args:
        in_dim: Width of the incoming features.
        out_dim: Width of the output.
        hidden_dims: Widths of any hidden layers. Empty gives a single linear layer.
        dropout: Dropout applied after each hidden activation.

    Raises:
        ValueError: If any layer width is not positive.
    """

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

        layers: list[nn.Module] = []
        width = in_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Transform ``(batch, in_dim)`` features into ``(batch, out_dim)``."""
        return self.net(x)
