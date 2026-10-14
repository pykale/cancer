"""Encoders that need no optional extra: modules that turn a modality's inputs into one embedding per patient."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from kalecancer.model.encode.base_encoders import AttentionEncoder


class MLP(nn.Module):
    """Linear → ReLU → Dropout for each hidden width, then a final Linear.

    Args:
        in_dim: Width of the input.
        hidden_dims: Width of each hidden layer. There must be at least one.
        out_dim: Width of the output.
        dropout: Dropout probability after each hidden layer.
    """

    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, dropout: float):
        super().__init__()
        if not hidden_dims:
            raise ValueError(
                "MLP needs at least one hidden layer; use torch.nn.Linear for a single projection "
                "(its dropout would have nothing to act on)"
            )
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.dropout = dropout
        widths = [in_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for width_in, width_out in zip(widths[:-1], widths[1:], strict=True):
            layers += [nn.Linear(width_in, width_out), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(widths[-1], out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Map an ``(n, in_dim)`` batch to ``(n, out_dim)``."""
        return self.layers(x)


class ABMIL(AttentionEncoder):
    """Gated attention-based multiple-instance pooling (Ilse et al., 2018).

    Takes a list of bags, one ``(N_i, in_dim)`` tensor per patient, and returns ``(n, hidden_dim)``.

    Args:
        in_dim: Width of each instance's features.
        hidden_dim: Width each instance is embedded to, which is also the output width.
        attention_dim: Width of the layers that compute the attention scores.
        dropout: Dropout probability after the instance embedding.
    """

    def __init__(self, in_dim: int, hidden_dim: int, attention_dim: int, dropout: float):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.out_dim = hidden_dim
        self.embed = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attention_value = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Tanh())
        self.attention_gate = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Sigmoid())
        self.attention_score = nn.Linear(attention_dim, 1)

    def _pool(self, bag: Tensor) -> tuple[Tensor, Tensor]:
        if bag.ndim != 2 or bag.shape[1] != self.in_dim or bag.shape[0] == 0:
            raise ValueError(f"ABMIL expects non-empty (N, {self.in_dim}) bags, got {tuple(bag.shape)}")
        instances = self.embed(bag)
        scores = self.attention_score(self.attention_value(instances) * self.attention_gate(instances)).squeeze(-1)
        # fp16 softmax over thousands of patches underflows to zeros
        weights = torch.softmax(scores.float(), dim=0)
        return weights.to(instances.dtype) @ instances, weights

    def forward(self, bags: list[Tensor]) -> Tensor:
        """Pool each bag into the attention-weighted mean of its embedded instances."""
        if len(bags) == 0:
            raise ValueError("ABMIL received no bags")
        return torch.stack([self._pool(bag)[0] for bag in bags])

    def attention(self, bags: list[Tensor]) -> list[Tensor]:
        """Attention weights per bag, row-aligned with the bag's instances; each sums to 1."""
        return [self._pool(bag)[1] for bag in bags]
