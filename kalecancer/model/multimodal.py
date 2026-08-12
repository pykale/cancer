"""Early, late, and hybrid multimodal fusion for survival prediction.

The three strategies differ in *where* modalities meet:

======  =========================================================  ==================
early   features concatenated at the input, one joint encoder      narrow use
late    independent per-modality models, predictions combined      baseline to beat
hybrid  per-modality encoders, latent fusion, plus auxiliary       richest supervision
        per-modality heads
======  =========================================================  ==================

All three share one interface and emit a patient-level log partial hazard, so the
strategy is swappable from configuration. Encoders are injected rather than built
here, so any modality that can produce a ``(batch, latent_dim)`` vector plugs in -
attention MIL over slide patches, an MLP over clinical variables, a 3D CT encoder.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from kalecancer.model.embed.multimodal_fusion import FusionBlock, build_fusion
from kalecancer.survival.cox import CoxHead
from kalecancer.survival.loss import cox_ph_loss


@dataclass
class MultimodalOutput:
    """Prediction of a multimodal survival model.

    Attributes:
        risk: ``(batch,)`` fused log partial hazard, higher meaning higher risk.
        modality_risk: Per-modality risks, for late fusion and for the auxiliary
            heads of hybrid fusion. Empty for early fusion, which has no
            modality-specific prediction.
    """

    risk: torch.Tensor
    modality_risk: dict[str, torch.Tensor] = field(default_factory=dict)


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

    keep = (torch.rand(mask.shape, device=mask.device, generator=generator) >= probability).float()
    dropped = mask * keep
    # Restore one modality wherever dropout removed them all.
    empty = dropped.sum(dim=1) == 0
    if bool(empty.any()):
        restore = mask[empty].argmax(dim=1)
        dropped[empty, restore] = 1.0
    return dropped


class MultimodalSurvivalModel(nn.Module):
    """Base class for multimodal survival models.

    Args:
        modalities: Modality names, in a fixed order that the modality mask follows.
        modality_dropout: Probability of dropping each present modality during
            training, improving robustness to missing modalities at inference.
    """

    def __init__(self, modalities: Sequence[str], modality_dropout: float = 0.0) -> None:
        super().__init__()
        if len(modalities) < 2:
            raise ValueError(f"multimodal fusion needs at least two modalities, got {list(modalities)}")
        self.modalities = list(modalities)
        self.modality_dropout = modality_dropout

    def forward(self, inputs: Mapping[str, torch.Tensor], mask: torch.Tensor | None = None) -> MultimodalOutput:
        """Predict patient risk.

        Args:
            inputs: One entry per modality, keyed by name.
            mask: ``(batch, num_modalities)`` indicator, 1 where present, in
                ``modalities`` order. ``None`` treats every modality as present.
        """
        raise NotImplementedError

    def _ordered(self, inputs: Mapping[str, torch.Tensor]) -> list[torch.Tensor]:
        missing = set(self.modalities) - set(inputs)
        if missing:
            raise KeyError(f"missing input for modalit(y/ies) {sorted(missing)}")
        return [inputs[name] for name in self.modalities]

    def _resolve_mask(self, batch_size: int, mask: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        resolved = (
            torch.ones(batch_size, len(self.modalities), device=device) if mask is None else mask.to(device).float()
        )
        return modality_dropout(resolved, self.modality_dropout) if self.training else resolved


class EarlyFusionSurvival(MultimodalSurvivalModel):
    """Concatenate modality features at the input, then encode jointly.

    The joint encoder can model raw cross-modal interactions, but every modality must
    already be a fixed-length vector of comparable scale - a gigapixel slide has to be
    pooled first. This is why early fusion suits commensurable inputs and is a narrow
    choice for imaging plus tabular data.

    Args:
        input_dims: Feature width of each modality, keyed by name.
        hidden_dim: Width of the joint encoder.
        dropout: Dropout inside the joint encoder.
        modality_dropout: Probability of dropping each present modality in training.
    """

    def __init__(
        self,
        input_dims: Mapping[str, int],
        hidden_dim: int = 128,
        dropout: float = 0.25,
        modality_dropout: float = 0.0,
    ) -> None:
        super().__init__(list(input_dims), modality_dropout)
        self.input_dims = dict(input_dims)
        self.placeholders = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.input_dims[name])) for name in self.modalities]
        )
        self.encoder = nn.Sequential(
            nn.Linear(sum(self.input_dims.values()), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = CoxHead(hidden_dim)

    def forward(self, inputs: Mapping[str, torch.Tensor], mask: torch.Tensor | None = None) -> MultimodalOutput:
        features = self._ordered(inputs)
        mask = self._resolve_mask(features[0].shape[0], mask, features[0].device)

        present = mask.unsqueeze(-1)
        filled = [
            feature * present[:, index] + placeholder * (1.0 - present[:, index])
            for index, (feature, placeholder) in enumerate(zip(features, self.placeholders, strict=True))
        ]
        return MultimodalOutput(risk=self.head(self.encoder(torch.cat(filled, dim=1))))


class LateFusionSurvival(MultimodalSurvivalModel):
    """Predict from each modality independently, then combine the risks.

    Every modality gets its own encoder and Cox head, so each prediction is
    interpretable on its own. Cross-modal interactions are never modelled during
    representation learning, which makes this the baseline that fusion has to beat.

    Risks are combined as a weighted mean over the *present* modalities, so a missing
    modality simply does not vote.

    Args:
        encoders: One encoder per modality, each mapping its input to ``(batch, dim)``.
        latent_dims: Output width of each encoder, keyed by modality.
        learn_weights: Learn the per-modality combination weights instead of using a
            uniform average.
        modality_dropout: Probability of dropping each present modality in training.
    """

    def __init__(
        self,
        encoders: Mapping[str, nn.Module],
        latent_dims: Mapping[str, int],
        learn_weights: bool = True,
        modality_dropout: float = 0.0,
    ) -> None:
        super().__init__(list(encoders), modality_dropout)
        self.encoders = nn.ModuleDict(dict(encoders))
        self.heads = nn.ModuleDict({name: CoxHead(latent_dims[name]) for name in self.modalities})
        self.weights = nn.Parameter(torch.zeros(len(self.modalities)), requires_grad=learn_weights)

    def forward(self, inputs: Mapping[str, torch.Tensor], mask: torch.Tensor | None = None) -> MultimodalOutput:
        self._ordered(inputs)
        modality_risk = {name: self.heads[name](self.encoders[name](inputs[name])) for name in self.modalities}
        risks = torch.stack([modality_risk[name] for name in self.modalities], dim=1)
        mask = self._resolve_mask(risks.shape[0], mask, risks.device)

        weights = torch.softmax(self.weights, dim=0).unsqueeze(0) * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        return MultimodalOutput(risk=(risks * weights).sum(dim=1), modality_risk=modality_risk)


class HybridFusionSurvival(MultimodalSurvivalModel):
    """Fuse per-modality latents, and keep an auxiliary head on each modality.

    The shared head sees a fused representation, so cross-modal interactions are
    learned; the auxiliary heads keep each encoder individually supervised, which
    both stabilises training when one modality is much easier and preserves a
    per-modality risk for interpretation.

    Args:
        encoders: One encoder per modality, each mapping its input to ``(batch, dim)``.
        latent_dims: Output width of each encoder, keyed by modality.
        fusion: Fusion method name, see
            :data:`~kalecancer.model.embed.multimodal_fusion.FUSION_METHODS`, or a
            prebuilt :class:`~kalecancer.model.embed.multimodal_fusion.FusionBlock`.
        fused_dim: Width of the fused representation.
        auxiliary_heads: Attach a Cox head to each modality latent.
        modality_dropout: Probability of dropping each present modality in training.
        **fusion_kwargs: Method-specific fusion options, e.g. ``rank``.
    """

    def __init__(
        self,
        encoders: Mapping[str, nn.Module],
        latent_dims: Mapping[str, int],
        fusion: str | FusionBlock = "concat",
        fused_dim: int = 64,
        auxiliary_heads: bool = True,
        modality_dropout: float = 0.0,
        **fusion_kwargs,
    ) -> None:
        super().__init__(list(encoders), modality_dropout)
        self.encoders = nn.ModuleDict(dict(encoders))

        ordered_dims = [latent_dims[name] for name in self.modalities]
        self.fusion = (
            fusion
            if isinstance(fusion, FusionBlock)
            else build_fusion(fusion, ordered_dims, fused_dim, **fusion_kwargs)
        )
        self.head = CoxHead(self.fusion.output_dim)
        self.auxiliary = (
            nn.ModuleDict({name: CoxHead(latent_dims[name]) for name in self.modalities}) if auxiliary_heads else None
        )

    def forward(self, inputs: Mapping[str, torch.Tensor], mask: torch.Tensor | None = None) -> MultimodalOutput:
        self._ordered(inputs)
        latents = [self.encoders[name](inputs[name]) for name in self.modalities]
        mask = self._resolve_mask(latents[0].shape[0], mask, latents[0].device)

        risk = self.head(self.fusion(latents, mask))
        modality_risk = (
            {name: self.auxiliary[name](latent) for name, latent in zip(self.modalities, latents, strict=True)}
            if self.auxiliary is not None
            else {}
        )
        return MultimodalOutput(risk=risk, modality_risk=modality_risk)


def multimodal_cox_loss(
    output: MultimodalOutput,
    event: torch.Tensor,
    duration: torch.Tensor,
    auxiliary_weight: float = 0.0,
    ties_method: str = "efron",
) -> torch.Tensor:
    """Cox loss on the fused risk, optionally supervising each modality too.

    The auxiliary term keeps every encoder learning even when one modality dominates
    the fused prediction. It is averaged over modalities so its scale does not depend
    on how many there are.

    Args:
        output: Prediction from a multimodal survival model.
        event: ``(batch,)`` indicator, 1 observed and 0 censored.
        duration: ``(batch,)`` event or censoring times.
        auxiliary_weight: Weight on the mean per-modality loss. 0 disables it.
        ties_method: Tie handling passed to the Cox loss.
    """
    loss = cox_ph_loss(output.risk, event, duration, ties_method=ties_method)
    if auxiliary_weight <= 0 or not output.modality_risk:
        return loss

    auxiliary = torch.stack(
        [cox_ph_loss(risk, event, duration, ties_method=ties_method) for risk in output.modality_risk.values()]
    )
    return loss + auxiliary_weight * auxiliary.mean()


FUSION_STRATEGIES: dict[str, type[MultimodalSurvivalModel]] = {
    "early": EarlyFusionSurvival,
    "late": LateFusionSurvival,
    "hybrid": HybridFusionSurvival,
}


def build_multimodal_survival(strategy: str, **kwargs) -> MultimodalSurvivalModel:
    """Construct a multimodal survival model by strategy name.

    Args:
        strategy: One of :data:`FUSION_STRATEGIES`.
        **kwargs: Passed to the strategy. ``"early"`` takes ``input_dims``; ``"late"``
            and ``"hybrid"`` take ``encoders`` and ``latent_dims``.

    Raises:
        KeyError: If ``strategy`` is not registered.
    """
    if strategy not in FUSION_STRATEGIES:
        raise KeyError(f"unknown fusion strategy {strategy!r}; available: {sorted(FUSION_STRATEGIES)}")
    return FUSION_STRATEGIES[strategy](**kwargs)
