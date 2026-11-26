"""Heads: ``nn.Module`` layers from an embedding to raw outputs, which an endpoint can take as its ``head``."""

from __future__ import annotations

from kale.utils.initialize_nn import bias_init, xavier_init
from torch import Tensor, nn


class LinearClassifier(nn.Module):
    """Linear layer producing class scores, with Xavier-initialized weights and zero bias.

    Moved from ``kale.predict.decode`` in PyKale (MIT license), where MOGONET used it as its per-omics classifier.

    Args:
        in_dim: Width of each input sample.
        out_dim: Width of each output sample.
        bias: Whether the layer learns an additive bias.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the weights with Xavier normal and the bias with zeros."""
        self.fc.apply(xavier_init)
        self.fc.apply(bias_init)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the linear layer to ``(batch, in_dim)`` input, giving ``(batch, out_dim)`` scores."""
        return self.fc(x)
