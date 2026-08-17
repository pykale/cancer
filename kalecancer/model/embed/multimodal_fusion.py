"""Latent-level fusion of per-modality representations.

Each block takes one latent vector per modality and returns a single fused
representation of ``output_dim``. Because every block exposes the same interface and
the same output width, the fusion method is swappable from configuration without
touching the encoders or the task head.

Absent modalities are first-class: a boolean modality mask is carried through, and
each block degrades in a way appropriate to its mechanism.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from kale.embed.multimodal_fusion import Concat, ProductOfExperts
from torch import nn

#: Log-variance assigned to an absent expert, making its precision negligible.
ABSENT_LOG_VAR = 20.0


class FusionBlock(nn.Module):
    """Base class for latent fusion.

    Args:
        input_dims: Latent dimension of each modality, in a fixed order.
        output_dim: Width of the fused representation.
    """

    def __init__(self, input_dims: Sequence[int], output_dim: int) -> None:
        super().__init__()
        if len(input_dims) < 2:
            raise ValueError(f"fusion needs at least two modalities, got {len(input_dims)}")
        self.input_dims = list(input_dims)
        self.output_dim = output_dim

    def forward(self, latents: list[torch.Tensor], mask: torch.Tensor | None = None) -> torch.Tensor:
        """Fuse per-modality latents.

        Args:
            latents: One ``(batch, input_dims[i])`` tensor per modality.
            mask: ``(batch, num_modalities)`` indicator, 1 where the modality is
                present. ``None`` treats every modality as present.

        Returns:
            ``(batch, output_dim)`` fused representation.
        """
        raise NotImplementedError

    def _check(self, latents: list[torch.Tensor], mask: torch.Tensor | None) -> torch.Tensor:
        """Validate inputs and return a usable mask."""
        if len(latents) != len(self.input_dims):
            raise ValueError(f"expected {len(self.input_dims)} modalities, got {len(latents)}")
        for index, (latent, dim) in enumerate(zip(latents, self.input_dims, strict=True)):
            if latent.shape[-1] != dim:
                raise ValueError(f"modality {index} has dimension {latent.shape[-1]}, expected {dim}")

        batch_size = latents[0].shape[0]
        if mask is None:
            return torch.ones(batch_size, len(latents), device=latents[0].device)
        if mask.shape != (batch_size, len(latents)):
            raise ValueError(f"mask must be ({batch_size}, {len(latents)}), got {tuple(mask.shape)}")
        return mask.to(latents[0].device).float()


class _PlaceholderMixin(nn.Module):
    """Substitutes a learned embedding wherever a modality is absent.

    Zero-padding an absent modality is indistinguishable from a genuine all-zero
    latent, so a learned placeholder is used instead.
    """

    def _init_placeholders(self, input_dims: Sequence[int]) -> None:
        self.placeholders = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for dim in input_dims])

    def _substitute(self, latents: list[torch.Tensor], mask: torch.Tensor) -> list[torch.Tensor]:
        present = mask.unsqueeze(-1)
        return [
            latent * present[:, index] + placeholder * (1.0 - present[:, index])
            for index, (latent, placeholder) in enumerate(zip(latents, self.placeholders, strict=True))
        ]


class ConcatFusion(FusionBlock, _PlaceholderMixin):
    """Concatenate latents, then project to ``output_dim``.

    The simplest fusion baseline. Wraps :class:`kale.embed.multimodal_fusion.Concat`
    and projects so the output width matches every other block.
    """

    def __init__(self, input_dims: Sequence[int], output_dim: int) -> None:
        super().__init__(input_dims, output_dim)
        self._init_placeholders(input_dims)
        self.concat = Concat()
        self.projection = nn.Linear(sum(self.input_dims), output_dim)

    def forward(self, latents: list[torch.Tensor], mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = self._check(latents, mask)
        return self.projection(self.concat(self._substitute(latents, mask)))


class ProductOfExpertsFusion(FusionBlock):
    """Fuse modalities as a product of Gaussian experts.

    Each modality proposes a Gaussian over the shared latent space; the product is
    available in closed form via :class:`kale.embed.multimodal_fusion.ProductOfExperts`.
    An absent modality is given negligible precision, so it drops out of the product
    without retraining - the reason this is the preferred path when modalities go
    missing at inference time.

    A fixed prior expert is included, following the multimodal-VAE convention, so a
    sample missing *every* modality still yields a defined result rather than 0/0.

    Args:
        input_dims: Latent dimension of each modality.
        output_dim: Width of the shared latent space.
        use_prior: Include the prior expert.
    """

    def __init__(self, input_dims: Sequence[int], output_dim: int, use_prior: bool = True) -> None:
        super().__init__(input_dims, output_dim)
        self.use_prior = use_prior
        self.to_mean = nn.ModuleList([nn.Linear(dim, output_dim) for dim in self.input_dims])
        self.to_log_var = nn.ModuleList([nn.Linear(dim, output_dim) for dim in self.input_dims])
        self.product = ProductOfExperts()

    def forward(self, latents: list[torch.Tensor], mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = self._check(latents, mask)

        means = torch.stack([layer(latent) for layer, latent in zip(self.to_mean, latents, strict=True)])
        log_vars = torch.stack([layer(latent) for layer, latent in zip(self.to_log_var, latents, strict=True)])

        absent = (1.0 - mask).t().unsqueeze(-1)
        means = means * (1.0 - absent)
        log_vars = log_vars * (1.0 - absent) + ABSENT_LOG_VAR * absent

        if self.use_prior:
            prior_shape = (1, means.shape[1], means.shape[2])
            means = torch.cat([means, torch.zeros(prior_shape, device=means.device)])
            log_vars = torch.cat([log_vars, torch.zeros(prior_shape, device=log_vars.device)])

        fused_mean, _ = self.product(means, log_vars)
        return fused_mean


class LowRankFusion(FusionBlock, _PlaceholderMixin):
    """Low-rank multimodal tensor fusion (Liu et al., 2018).

    Models multiplicative interactions between modalities while keeping the parameter
    count linear in the number of modalities, by factorising the interaction tensor.

    Implemented here rather than reusing ``kale.embed.multimodal_fusion.LowRankTensorFusion``,
    whose factors are held in a plain list of device-moved tensors: they are not
    registered as module parameters, so ``.parameters()`` is empty and an optimiser
    never updates them, and its device is fixed to ``cuda:0``.

    Args:
        input_dims: Latent dimension of each modality.
        output_dim: Width of the fused representation.
        rank: Rank of the factorisation; higher captures richer interactions.
    """

    def __init__(self, input_dims: Sequence[int], output_dim: int, rank: int = 4) -> None:
        super().__init__(input_dims, output_dim)
        self._init_placeholders(input_dims)
        self.rank = rank
        self.factors = nn.ParameterList(
            [nn.Parameter(torch.empty(rank, dim + 1, output_dim)) for dim in self.input_dims]
        )
        for factor in self.factors:
            nn.init.xavier_normal_(factor)
        self.fusion_weights = nn.Parameter(torch.empty(1, rank))
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, latents: list[torch.Tensor], mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = self._check(latents, mask)
        latents = self._substitute(latents, mask)

        fused = torch.ones(1, device=latents[0].device)
        for latent, factor in zip(latents, self.factors, strict=True):
            ones = torch.ones(latent.shape[0], 1, dtype=latent.dtype, device=latent.device)
            fused = fused * torch.matmul(torch.cat([ones, latent], dim=1), factor)

        return torch.matmul(self.fusion_weights, fused.permute(1, 0, 2)).squeeze(1) + self.fusion_bias


FUSION_METHODS: dict[str, type[FusionBlock]] = {
    "concat": ConcatFusion,
    "poe": ProductOfExpertsFusion,
    "lowrank": LowRankFusion,
}


def build_fusion(method: str, input_dims: Sequence[int], output_dim: int, **kwargs) -> FusionBlock:
    """Construct a fusion block by name.

    Args:
        method: One of :data:`FUSION_METHODS`.
        input_dims: Latent dimension of each modality.
        output_dim: Width of the fused representation.
        **kwargs: Method-specific options, e.g. ``rank`` for ``"lowrank"``.

    Raises:
        KeyError: If ``method`` is not registered.
    """
    if method not in FUSION_METHODS:
        raise KeyError(f"unknown fusion method {method!r}; available: {sorted(FUSION_METHODS)}")
    return FUSION_METHODS[method](input_dims, output_dim, **kwargs)
