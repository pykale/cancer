"""Tests for early, late, and hybrid multimodal survival strategies."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from kalecancer.model import (
    FUSION_STRATEGIES,
    EarlyFusionSurvival,
    HybridFusionSurvival,
    LateFusionSurvival,
    build_multimodal_survival,
    modality_dropout,
    multimodal_cox_loss,
)
from kalecancer.model.embed import ConcatFusion

BATCH = 8
RAW_DIMS = {"wsi": 64, "tab": 10}
LATENT_DIMS = {"wsi": 32, "tab": 16}


@pytest.fixture
def inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {name: torch.randn(BATCH, dim) for name, dim in RAW_DIMS.items()}


@pytest.fixture
def encoders() -> dict[str, nn.Module]:
    torch.manual_seed(0)
    return {name: nn.Linear(RAW_DIMS[name], LATENT_DIMS[name]) for name in RAW_DIMS}


@pytest.fixture
def outcome() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    return (torch.rand(BATCH) < 0.5).int(), torch.rand(BATCH) * 1000 + 50


def build(strategy: str, encoders: dict[str, nn.Module], **kwargs):
    if strategy == "early":
        return build_multimodal_survival("early", input_dims=RAW_DIMS, **kwargs)
    return build_multimodal_survival(strategy, encoders=encoders, latent_dims=LATENT_DIMS, **kwargs)


@pytest.mark.parametrize("strategy", ["early", "late", "hybrid"])
def test_every_strategy_emits_one_risk_per_patient(strategy: str, encoders, inputs) -> None:
    output = build(strategy, encoders)(inputs)

    assert output.risk.shape == (BATCH,)
    assert bool(torch.isfinite(output.risk).all())


@pytest.mark.parametrize("strategy", ["early", "late", "hybrid"])
def test_every_strategy_accepts_a_modality_mask(strategy: str, encoders, inputs) -> None:
    mask = torch.tensor([[1.0, 0.0]] * (BATCH // 2) + [[1.0, 1.0]] * (BATCH // 2))

    output = build(strategy, encoders)(inputs, mask)

    assert output.risk.shape == (BATCH,)
    assert bool(torch.isfinite(output.risk).all())


@pytest.mark.parametrize("strategy", ["early", "late", "hybrid"])
def test_gradients_reach_every_parameter_path(strategy: str, encoders, inputs, outcome) -> None:
    model = build(strategy, encoders)
    event, duration = outcome

    multimodal_cox_loss(model(inputs), event, duration).backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_registry_covers_the_documented_strategies() -> None:
    assert set(FUSION_STRATEGIES) == {"early", "late", "hybrid"}


def test_registry_rejects_an_unknown_strategy() -> None:
    with pytest.raises(KeyError, match="unknown fusion strategy"):
        build_multimodal_survival("mid", input_dims=RAW_DIMS)


@pytest.mark.parametrize("strategy", ["early", "late", "hybrid"])
def test_missing_modality_input_is_rejected(strategy: str, encoders, inputs) -> None:
    with pytest.raises(KeyError, match="missing input"):
        build(strategy, encoders)({"wsi": inputs["wsi"]})


def test_a_single_modality_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least two modalities"):
        EarlyFusionSurvival(input_dims={"wsi": 64})


def test_early_fusion_has_no_per_modality_prediction(encoders, inputs) -> None:
    """Early fusion mixes modalities before encoding, so no modality has its own risk."""
    output = build("early", encoders)(inputs)

    assert output.modality_risk == {}


def test_early_fusion_uses_every_modality(encoders, inputs) -> None:
    model = build("early", encoders).eval()

    changed = dict(inputs)
    changed["tab"] = torch.randn(BATCH, RAW_DIMS["tab"])

    assert not torch.allclose(model(inputs).risk, model(changed).risk)


def test_late_fusion_exposes_an_independent_risk_per_modality(encoders, inputs) -> None:
    output = build("late", encoders)(inputs)

    assert set(output.modality_risk) == set(RAW_DIMS)
    assert all(risk.shape == (BATCH,) for risk in output.modality_risk.values())


def test_late_fusion_ignores_a_modality_that_is_absent(encoders, inputs) -> None:
    """An absent modality must not vote in the combined risk."""
    model = build("late", encoders).eval()
    mask = torch.tensor([[1.0, 0.0]] * BATCH)

    output = model(inputs, mask)

    assert torch.allclose(output.risk, output.modality_risk["wsi"], atol=1e-5)


def test_late_fusion_with_uniform_weights_averages_the_modalities(encoders, inputs) -> None:
    model = LateFusionSurvival(encoders, LATENT_DIMS, learn_weights=False).eval()

    output = model(inputs)
    expected = torch.stack(list(output.modality_risk.values())).mean(dim=0)

    assert torch.allclose(output.risk, expected, atol=1e-5)


def test_late_fusion_weights_are_frozen_when_not_learned(encoders) -> None:
    model = LateFusionSurvival(encoders, LATENT_DIMS, learn_weights=False)

    assert not model.weights.requires_grad


def test_hybrid_fusion_supervises_each_modality(encoders, inputs) -> None:
    output = build("hybrid", encoders)(inputs)

    assert set(output.modality_risk) == set(RAW_DIMS)


def test_hybrid_auxiliary_heads_can_be_disabled(encoders, inputs) -> None:
    model = HybridFusionSurvival(encoders, LATENT_DIMS, auxiliary_heads=False)

    assert model(inputs).modality_risk == {}


@pytest.mark.parametrize(("method", "kwargs"), [("concat", {}), ("poe", {}), ("lowrank", {"rank": 3})])
def test_hybrid_fusion_method_is_swappable(method: str, kwargs: dict, encoders, inputs) -> None:
    """Changing only the fusion name must keep the same interface, as config swapping needs."""
    model = HybridFusionSurvival(encoders, LATENT_DIMS, fusion=method, fused_dim=24, **kwargs)

    assert model(inputs).risk.shape == (BATCH,)


def test_hybrid_accepts_a_prebuilt_fusion_block(encoders, inputs) -> None:
    block = ConcatFusion([LATENT_DIMS["wsi"], LATENT_DIMS["tab"]], 48)

    model = HybridFusionSurvival(encoders, LATENT_DIMS, fusion=block)

    assert model.fusion is block
    assert model(inputs).risk.shape == (BATCH,)


def test_hybrid_fused_risk_differs_from_its_auxiliary_risks(encoders, inputs) -> None:
    output = build("hybrid", encoders)(inputs)

    for risk in output.modality_risk.values():
        assert not torch.allclose(output.risk, risk)


def test_modality_dropout_never_leaves_a_sample_empty() -> None:
    dropped = modality_dropout(torch.ones(200, 3), probability=0.9)

    assert int(dropped.sum(dim=1).min()) >= 1


def test_modality_dropout_removes_some_modalities() -> None:
    dropped = modality_dropout(torch.ones(200, 3), probability=0.5)

    assert float(dropped.mean()) < 1.0


def test_modality_dropout_is_a_no_op_at_zero_probability() -> None:
    mask = torch.ones(4, 2)

    assert modality_dropout(mask, probability=0.0) is mask


def test_modality_dropout_never_revives_an_absent_modality() -> None:
    mask = torch.tensor([[1.0, 0.0]] * 50)

    dropped = modality_dropout(mask, probability=0.5)

    assert float(dropped[:, 1].sum()) == 0.0


def test_modality_dropout_only_applies_while_training(encoders, inputs) -> None:
    model = build("hybrid", encoders, modality_dropout=0.9).eval()

    assert torch.allclose(model(inputs).risk, model(inputs).risk)


def test_auxiliary_loss_changes_the_objective(encoders, inputs, outcome) -> None:
    event, duration = outcome
    output = build("hybrid", encoders)(inputs)

    without = multimodal_cox_loss(output, event, duration, auxiliary_weight=0.0)
    with_auxiliary = multimodal_cox_loss(output, event, duration, auxiliary_weight=0.5)

    assert not torch.isclose(without, with_auxiliary)


def test_auxiliary_loss_is_ignored_without_modality_risks(encoders, inputs, outcome) -> None:
    event, duration = outcome
    output = build("early", encoders)(inputs)

    assert torch.isclose(
        multimodal_cox_loss(output, event, duration, auxiliary_weight=0.0),
        multimodal_cox_loss(output, event, duration, auxiliary_weight=0.9),
    )


def test_loss_is_finite_and_differentiable(encoders, inputs, outcome) -> None:
    event, duration = outcome
    model = build("hybrid", encoders)

    loss = multimodal_cox_loss(model(inputs), event, duration, auxiliary_weight=0.3)
    loss.backward()

    assert torch.isfinite(loss)
