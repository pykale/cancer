"""Attention scoring over a set of instances.

Two blocks: :class:`GatedAttention` scores instances, and :class:`AttentionMIL`
pools a bag of them into one vector using those scores. Neither knows what an
instance is -- a slide patch, a cell, a time step -- which is what makes them layers
rather than models. Adapting the pooler to the embedder contract is
:class:`~kalecancer.model.embed.BagEncoder`.

PyKale's ``kale.embed.attention`` covers cross-modal attention between latent
representations (``BANLayer``) rather than within-bag scoring, so the two do not
overlap.
"""

from __future__ import annotations

import torch
from torch import nn


class GatedAttention(nn.Module):
    """Attention scoring over instances.

    Args:
        input_dim: Dimension of each instance embedding.
        attention_dim: Width of the attention hidden layer.
        gated: Use the gated variant (``tanh`` branch modulated by a ``sigmoid``
            branch). The non-gated variant keeps only the ``tanh`` branch.
    """

    def __init__(self, input_dim: int, attention_dim: int = 128, gated: bool = True) -> None:
        super().__init__()
        self.gated = gated
        self.value = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Sigmoid()) if gated else None
        self.score = nn.Linear(attention_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score instances.

        Args:
            x: ``(num_instances, input_dim)`` instance embeddings.

        Returns:
            ``(num_instances,)`` unnormalised attention logits.
        """
        hidden = self.value(x)
        if self.gate is not None:
            hidden = hidden * self.gate(x)
        return self.score(hidden).squeeze(-1)


class AttentionMIL(nn.Module):
    """Pool a bag of patch embeddings into one representation.

    Args:
        input_dim: Dimension of the precomputed patch features (1024 for UNI).
        hidden_dim: Width of the projected instance embedding and of the output.
        attention_dim: Width of the attention hidden layer.
        dropout: Dropout applied after the projection.
        gated: Use gated attention.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        dropout: float = 0.25,
        gated: bool = True,
    ) -> None:
        super().__init__()
        self.output_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = GatedAttention(hidden_dim, attention_dim=attention_dim, gated=gated)

    def forward(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool one bag.

        Args:
            bag: ``(num_patches, input_dim)`` patch embeddings.

        Returns:
            ``(embedding, attention)`` where ``embedding`` is ``(hidden_dim,)`` and
            ``attention`` is ``(num_patches,)`` summing to 1, aligned index-for-index
            with ``bag``.

        Raises:
            ValueError: If ``bag`` is not a non-empty 2D tensor.
        """
        if bag.ndim != 2:
            raise ValueError(f"expected a 2D bag (num_patches, input_dim), got shape {tuple(bag.shape)}")
        if bag.shape[0] == 0:
            raise ValueError("cannot pool an empty bag")

        instances = self.projection(bag)
        attention = torch.softmax(self.attention(instances), dim=0)
        embedding = attention @ instances
        return embedding, attention

    def forward_bags(self, bags: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Pool a batch of variable-length bags.

        Args:
            bags: Per-patient ``(num_patches, input_dim)`` tensors; lengths may differ.

        Returns:
            ``(embeddings, attentions)`` where ``embeddings`` is ``(batch, hidden_dim)``
            and ``attentions`` holds one weight vector per bag.

        Raises:
            ValueError: If ``bags`` is empty.
        """
        # len(), not truthiness: a tensor of several values has no boolean value.
        if len(bags) == 0:
            raise ValueError("cannot pool an empty batch of bags")
        pooled = [self.forward(bag) for bag in bags]
        embeddings = torch.stack([embedding for embedding, _ in pooled])
        return embeddings, [attention for _, attention in pooled]
