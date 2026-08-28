"""Multimodal fusion: where modalities meet, and how they are combined.

Two independent choices, so either can change from configuration alone:

**stage** -- *where* fusion happens, see :class:`MultimodalFusion`.

**method** -- *how* the vectors are combined, see :data:`FUSION_METHODS`. Each block
takes one latent per modality and returns a single representation of ``output_dim``,
so swapping the method leaves the embedders and the task head untouched.

The losses that score a :class:`MultimodalOutput` live in
:mod:`kalecancer.model.predict.losses`, beside the heads that produced it.

Absent modalities are first-class: a boolean modality mask is carried through, and
each block degrades in a way appropriate to its mechanism.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

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


FUSION_STAGES = ("early", "intermediate", "late", "hybrid")


@dataclass
class MultimodalOutput:
    """What a fusion model returns.

    Attributes:
        prediction: The model output, from the task head.
        representation: Fused features, for stages that produce them. ``None`` for
            late fusion, which never forms a joint representation.
        modality_predictions: Per-modality outputs, from late fusion's heads or
            hybrid fusion's auxiliary heads. Empty otherwise.
    """

    prediction: torch.Tensor
    representation: torch.Tensor | None = None
    modality_predictions: dict[str, torch.Tensor] = field(default_factory=dict)


def modality_dropout(mask: torch.Tensor, probability: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Randomly mark present modalities absent, simulating missing data.

    At least one modality is always kept, so no sample is left with nothing to
    predict from.

    Args:
        mask: ``(batch, num_modalities)`` indicator, 1 where present.
        probability: Chance of dropping each present modality.
        generator: Optional RNG for reproducibility.

    Returns:
        A new mask; the input is not modified.
    """
    if probability <= 0:
        return mask

    device = mask.device if generator is None else generator.device
    keep = (torch.rand(mask.shape, device=device, generator=generator) >= probability).float()
    dropped = mask * keep.to(mask.device)
    empty = dropped.sum(dim=1) == 0
    if bool(empty.any()):
        dropped[empty, mask[empty].argmax(dim=1)] = 1.0
    return dropped


def _embedder_dim(name: str, embedder: nn.Module) -> int:
    dim = getattr(embedder, "out_dim", None)
    if dim is None:
        raise AttributeError(f"embedder {name!r} must expose 'out_dim'; see kalecancer.model.embed.Embedder")
    return int(dim)


class MultimodalFusion(nn.Module):
    """Combine several modalities at a configurable stage.

    The stage decides *where* modalities meet; the method decides *how* their vectors
    are combined. The two are independent, so a configuration can change either alone.

    ================  ========================================================
    ``early``         Fusion before modality-specific encoding. **Unavailable**:
                      this pipeline starts from already-extracted features, so
                      no raw input remains to fuse. Kept as a named stage so a
                      raw-input pipeline can add it without changing callers.
    ``intermediate``  Encode each modality, fuse the features, predict once.
                      The main stage, and where ``method`` applies.
    ``late``          Encode and predict per modality, combine the predictions.
    ``hybrid``        An intermediate trunk plus a head on each modality, whose
                      outputs give auxiliary supervision and, optionally, votes.
    ================  ========================================================

    Every modality is projected to ``fusion_dim`` first, so embedders of differing
    widths interoperate and every head sees the same input size.

    Args:
        embedders: One module per modality, each exposing ``out_dim`` and mapping its
            input to ``(batch, out_dim)``.
        head_factory: Called with a width to build a task head, e.g.
            :class:`~kalecancer.model.predict.losses.CoxHead`. Injecting it leaves prediction
            and loss logic where it already lives.
        stage: One of :data:`FUSION_STAGES`.
        method: One of :data:`FUSION_METHODS`; used by ``intermediate`` and ``hybrid``.
        fusion_dim: Width every modality is projected to, and of the fused vector.
        auxiliary_heads: Hybrid only; attach a head to each modality.
        combine_predictions: Hybrid only; blend the per-modality predictions into the
            output instead of using the fused trunk alone.
        modality_dropout: Probability of dropping each present modality in training.
        **method_kwargs: Method-specific options, e.g. ``rank`` for ``"lowrank"``.

    Raises:
        NotImplementedError: If ``stage`` is ``"early"``.
        ValueError: If ``stage`` is unknown or fewer than two modalities are given.
    """

    def __init__(
        self,
        embedders: Mapping[str, nn.Module],
        head_factory: Callable[[int], nn.Module],
        stage: str = "intermediate",
        method: str = "concat",
        fusion_dim: int = 256,
        auxiliary_heads: bool = True,
        combine_predictions: bool = False,
        modality_dropout: float = 0.0,
        **method_kwargs,
    ) -> None:
        super().__init__()
        if stage == "early":
            raise NotImplementedError(
                "early fusion combines raw modalities before any modality-specific "
                "encoding, which this pipeline cannot do because it starts from "
                "extracted features; use 'intermediate' to fuse those features"
            )
        if stage not in FUSION_STAGES:
            raise ValueError(f"unknown fusion stage {stage!r}; available: {FUSION_STAGES}")
        if not embedders:
            raise ValueError("at least one modality is required")

        self.modalities = list(embedders)
        self.stage = stage
        self.fusion_dim = fusion_dim
        self.modality_dropout = modality_dropout
        self.combine_predictions = combine_predictions

        # Declared optional up front: which of these a model has depends on its
        # stage, and mypy otherwise fixes each type from whichever branch runs first.
        self.fusion: nn.Module | None
        self.head: nn.Module | None
        self.modality_heads: nn.ModuleDict | None
        self.weights: nn.Parameter | None

        self.embedders = nn.ModuleDict(dict(embedders))
        self.projections = nn.ModuleDict(
            {name: nn.Linear(_embedder_dim(name, embedders[name]), fusion_dim) for name in self.modalities}
        )

        if stage == "late":
            self.fusion = None
            self.head = None
            self.modality_heads = nn.ModuleDict({name: head_factory(fusion_dim) for name in self.modalities})
            self.weights = nn.Parameter(torch.zeros(len(self.modalities)))
            return

        # With one modality there is nothing to fuse: the projection is the
        # representation, which lets a unimodal baseline share this code path.
        self.fusion = (
            build_fusion(method, [fusion_dim] * len(self.modalities), fusion_dim, **method_kwargs)
            if len(self.modalities) > 1
            else None
        )
        self.head = head_factory(self.fusion.output_dim if self.fusion else fusion_dim)
        self.modality_heads = (
            nn.ModuleDict({name: head_factory(fusion_dim) for name in self.modalities})
            if stage == "hybrid" and auxiliary_heads
            else None
        )
        # Only created where it is used: an unused parameter would still be decayed
        # by the optimiser and would trip distributed training's unused-parameter check.
        # The fused trunk always votes; each modality votes only when present.
        self.weights = nn.Parameter(torch.zeros(1 + len(self.modalities))) if combine_predictions else None

    def forward(
        self,
        modalities: Mapping[str, torch.Tensor],
        present: Mapping[str, torch.Tensor] | None = None,
    ) -> MultimodalOutput:
        """Embed, fuse and predict.

        Args:
            modalities: One entry per modality, keyed by name. Each is whatever that
                modality's embedder accepts.
            present: ``(batch,)`` boolean per modality, as carried by
                :class:`~kalecancer.loaddata.multimodal_access.PatientBatch`. ``None`` treats
                every modality as present.

        Raises:
            KeyError: If a configured modality is absent from ``modalities``.
        """
        missing = set(self.modalities) - set(modalities)
        if missing:
            raise KeyError(f"missing input for modalities {sorted(missing)}")

        embeddings = [self.projections[name](self.embedders[name](modalities[name])) for name in self.modalities]
        mask = self._resolve_mask(present, embeddings[0])

        if self.stage == "late":
            # Both are built for every late model; only other stages leave them unset.
            assert self.modality_heads is not None and self.weights is not None
            predictions = {
                name: self.modality_heads[name](embedding)
                for name, embedding in zip(self.modalities, embeddings, strict=True)
            }
            stacked = torch.stack([predictions[name] for name in self.modalities])
            return MultimodalOutput(
                prediction=self._combine(stacked, mask.t(), self.weights),
                modality_predictions=predictions,
            )

        representation = self.fusion(embeddings, mask) if self.fusion else embeddings[0]
        assert self.head is not None
        fused = self.head(representation)
        if self.modality_heads is None:
            return MultimodalOutput(prediction=fused, representation=representation)

        predictions = {
            name: self.modality_heads[name](embedding)
            for name, embedding in zip(self.modalities, embeddings, strict=True)
        }
        if not self.combine_predictions:
            return MultimodalOutput(
                prediction=fused,
                representation=representation,
                modality_predictions=predictions,
            )

        # combine_predictions is what creates the vote weights, and we are inside it.
        assert self.weights is not None
        stacked = torch.stack([fused] + [predictions[name] for name in self.modalities])
        available = torch.cat([torch.ones(mask.shape[0], 1, device=mask.device), mask], dim=1)
        return MultimodalOutput(
            prediction=self._combine(stacked, available.t(), self.weights),
            representation=representation,
            modality_predictions=predictions,
        )

    def _resolve_mask(self, present: Mapping[str, torch.Tensor] | None, reference: torch.Tensor) -> torch.Tensor:
        batch_size, device = reference.shape[0], reference.device
        if present is None:
            mask = torch.ones(batch_size, len(self.modalities), device=device)
        else:
            mask = torch.stack([present[name].to(device).float() for name in self.modalities], dim=1)
        return modality_dropout(mask, self.modality_dropout) if self.training else mask

    @staticmethod
    def _combine(stacked: torch.Tensor, availability: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Weighted mean over the leading axis, ignoring unavailable contributors."""
        combined = torch.softmax(weights, dim=0).unsqueeze(1) * availability
        combined = combined / combined.sum(dim=0, keepdim=True).clamp_min(torch.finfo(combined.dtype).eps)
        while combined.dim() < stacked.dim():
            combined = combined.unsqueeze(-1)
        return (stacked * combined).sum(dim=0)
