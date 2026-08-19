"""Early, late, and hybrid multimodal fusion for survival prediction.

The three strategies differ in *what* is combined:

======  ==========================================================================
early   **features**: each modality is encoded, the features are fused, and a
        single shared head predicts from the fused representation
late    **decisions**: each modality is encoded and predicts independently, and
        the risks are combined
hybrid  **both**: an early-fusion trunk supplies the prediction while late-style
        per-modality heads keep each encoder supervised and interpretable
======  ==========================================================================

Early fusion is feature-level, not raw-input-level: modalities are encoded first,
because a gigapixel slide and a clinical table share no common input space. What
makes it *early* is that the fusion happens before any prediction is made, so the
shared head learns from cross-modal structure rather than from separate verdicts.

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
from kalecancer.survival.cox import CoxHead, as_event_mask, neg_partial_log_likelihood


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

    # A generator is bound to its own device, so draw there and move the result.
    device = mask.device if generator is None else generator.device
    keep = (torch.rand(mask.shape, device=device, generator=generator) >= probability).float()
    dropped = mask * keep.to(mask.device)
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
    """Encode each modality, fuse the features, then predict once.

    Fusion happens before any prediction, so the shared head learns from cross-modal
    structure. The fusion operator is swappable - ``"concat"`` for a plain
    feature-level baseline, ``"poe"`` when modalities may be missing, ``"lowrank"``
    for multiplicative interactions.

    Encoders are trained end to end with the survival loss. To reproduce strict early
    fusion over pre-extracted features, freeze the encoders; this is already the case
    for WSI features, which come from a frozen foundation model.

    Args:
        encoders: One encoder per modality, each mapping its input to ``(batch, dim)``.
        latent_dims: Output width of each encoder, keyed by modality.
        fusion: Fusion method name, see
            :data:`~kalecancer.model.embed.multimodal_fusion.FUSION_METHODS`, or a
            prebuilt :class:`~kalecancer.model.embed.multimodal_fusion.FusionBlock`.
        fused_dim: Width of the fused representation.
        modality_dropout: Probability of dropping each present modality in training.
        **fusion_kwargs: Method-specific fusion options, e.g. ``rank``.
    """

    def __init__(
        self,
        encoders: Mapping[str, nn.Module],
        latent_dims: Mapping[str, int],
        fusion: str | FusionBlock = "concat",
        fused_dim: int = 64,
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

    def _encode(
        self, inputs: Mapping[str, torch.Tensor], mask: torch.Tensor | None
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        self._ordered(inputs)
        latents = [self.encoders[name](inputs[name]) for name in self.modalities]
        return latents, self._resolve_mask(latents[0].shape[0], mask, latents[0].device)

    def forward(self, inputs: Mapping[str, torch.Tensor], mask: torch.Tensor | None = None) -> MultimodalOutput:
        latents, mask = self._encode(inputs, mask)
        return MultimodalOutput(risk=self.head(self.fusion(latents, mask)).squeeze(-1))


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
        modality_risk = {
            name: self.heads[name](self.encoders[name](inputs[name])).squeeze(-1) for name in self.modalities
        }
        risks = torch.stack([modality_risk[name] for name in self.modalities], dim=1)
        mask = self._resolve_mask(risks.shape[0], mask, risks.device)

        weights = torch.softmax(self.weights, dim=0).unsqueeze(0) * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        return MultimodalOutput(risk=(risks * weights).sum(dim=1), modality_risk=modality_risk)


class HybridFusionSurvival(EarlyFusionSurvival):
    """Combine early and late fusion: a fused trunk plus per-modality heads.

    Extends :class:`EarlyFusionSurvival` with the late-fusion ingredient - a Cox head
    on every modality latent. The fused trunk learns cross-modal structure while the
    per-modality heads keep each encoder directly supervised, which stabilises
    training when one modality is much easier to learn from and preserves a
    per-modality risk for interpretation.

    By default the fused trunk alone makes the prediction and the per-modality heads
    act as auxiliary supervision via
    :func:`multimodal_cox_loss`. Set ``combine_risks`` to also merge the decisions,
    giving a combination at both the feature and the decision level.

    Args:
        encoders: One encoder per modality, each mapping its input to ``(batch, dim)``.
        latent_dims: Output width of each encoder, keyed by modality.
        fusion: Fusion method name or a prebuilt fusion block.
        fused_dim: Width of the fused representation.
        auxiliary_heads: Attach a Cox head to each modality latent.
        combine_risks: Blend the fused risk with the per-modality risks instead of
            using the fused risk alone. Requires ``auxiliary_heads``.
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
        combine_risks: bool = False,
        modality_dropout: float = 0.0,
        **fusion_kwargs,
    ) -> None:
        super().__init__(
            encoders,
            latent_dims,
            fusion=fusion,
            fused_dim=fused_dim,
            modality_dropout=modality_dropout,
            **fusion_kwargs,
        )
        if combine_risks and not auxiliary_heads:
            raise ValueError("combine_risks needs auxiliary_heads to produce per-modality risks")

        self.combine_risks = combine_risks
        self.auxiliary = (
            nn.ModuleDict({name: CoxHead(latent_dims[name]) for name in self.modalities}) if auxiliary_heads else None
        )
        # One weight for the fused trunk, then one per modality.
        self.risk_weights = nn.Parameter(torch.zeros(1 + len(self.modalities)))

    def forward(self, inputs: Mapping[str, torch.Tensor], mask: torch.Tensor | None = None) -> MultimodalOutput:
        latents, mask = self._encode(inputs, mask)
        fused_risk = self.head(self.fusion(latents, mask)).squeeze(-1)

        if self.auxiliary is None:
            return MultimodalOutput(risk=fused_risk)

        modality_risk = {
            name: self.auxiliary[name](latent).squeeze(-1)
            for name, latent in zip(self.modalities, latents, strict=True)
        }
        if not self.combine_risks:
            return MultimodalOutput(risk=fused_risk, modality_risk=modality_risk)

        risks = torch.stack([fused_risk] + [modality_risk[name] for name in self.modalities], dim=1)
        # The fused trunk always votes; each modality votes only when present.
        available = torch.cat([torch.ones(mask.shape[0], 1, device=mask.device), mask], dim=1)
        weights = torch.softmax(self.risk_weights, dim=0).unsqueeze(0) * available
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        return MultimodalOutput(risk=(risks * weights).sum(dim=1), modality_risk=modality_risk)


def multimodal_cox_loss(
    output: MultimodalOutput,
    event: torch.Tensor,
    duration: torch.Tensor,
    auxiliary_weight: float = 0.0,
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
    """
    loss = neg_partial_log_likelihood(output.risk, duration, as_event_mask(event))
    if auxiliary_weight <= 0 or not output.modality_risk:
        return loss

    auxiliary = torch.stack(
        [neg_partial_log_likelihood(risk, duration, as_event_mask(event)) for risk in output.modality_risk.values()]
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
