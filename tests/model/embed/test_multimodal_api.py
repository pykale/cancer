"""Tests for the multimodal fusion API: stage and method are independent."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from kalecancer.model.embed import FUSION_STAGES, ConcatFusion, MultimodalFusion
from kalecancer.model.predict import CoxHead
from kalecancer.model.predict.losses import multimodal_cox_loss

BATCH = 8
RAW_DIMS = {"tabular": 10, "imaging": 64}
EMBED_DIMS = {"tabular": 16, "imaging": 32}
FUSION_DIM = 24
STAGES = ["intermediate", "late", "hybrid"]
METHODS = [("concat", {}), ("poe", {}), ("lowrank", {"rank": 3})]


class ToyEmbedder(nn.Module):
    """Stands in for TabICLEmbedder or BagEncoder: exposes out_dim, returns (B, out_dim)."""

    needs_full_batch = False

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@pytest.fixture
def embedders() -> dict[str, nn.Module]:
    torch.manual_seed(0)
    return {name: ToyEmbedder(RAW_DIMS[name], EMBED_DIMS[name]) for name in RAW_DIMS}


@pytest.fixture
def modalities() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {name: torch.randn(BATCH, dim) for name, dim in RAW_DIMS.items()}


@pytest.fixture
def outcome() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    return torch.rand(BATCH) * 1000 + 50, (torch.rand(BATCH) < 0.5).int()


def build(embedders, stage="intermediate", **kwargs) -> MultimodalFusion:
    return MultimodalFusion(embedders, CoxHead, stage=stage, fusion_dim=FUSION_DIM, **kwargs)


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_emits_one_prediction_per_sample(stage, embedders, modalities) -> None:
    output = build(embedders, stage)(modalities)

    assert output.prediction.shape[0] == BATCH
    assert bool(torch.isfinite(output.prediction).all())


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize(("method", "kwargs"), METHODS)
def test_stage_and_method_vary_independently(stage, method, kwargs, embedders, modalities) -> None:
    """The point of the API: any stage works with any method, with no code change."""
    model = build(embedders, stage, method=method, **kwargs)

    assert model(modalities).prediction.shape[0] == BATCH


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_accepts_a_present_mask(stage, embedders, modalities) -> None:
    present = {
        "tabular": torch.ones(BATCH, dtype=torch.bool),
        "imaging": torch.tensor([True, False] * (BATCH // 2)),
    }

    output = build(embedders, stage)(modalities, present)

    assert bool(torch.isfinite(output.prediction).all())


@pytest.mark.parametrize("stage", STAGES)
def test_gradients_reach_every_embedder(stage, embedders, modalities, outcome) -> None:
    times, events = outcome
    model = build(embedders, stage)

    multimodal_cox_loss(model(modalities), times, events).backward()

    for name in RAW_DIMS:
        gradient = model.embedders[name].net.weight.grad
        assert gradient is not None and float(gradient.abs().sum()) > 0


def test_early_stage_is_named_but_not_available(embedders) -> None:
    """Raw-input fusion has no meaning once features are already extracted."""
    with pytest.raises(NotImplementedError, match="before any modality-specific"):
        build(embedders, "early")


def test_early_remains_a_documented_stage() -> None:
    assert FUSION_STAGES == ("early", "intermediate", "late", "hybrid")


def test_unknown_stage_is_rejected(embedders) -> None:
    with pytest.raises(ValueError, match="unknown fusion stage"):
        build(embedders, "middle")


def test_a_single_modality_needs_no_fusion(embedders, modalities) -> None:
    """A unimodal baseline shares this code path: there is simply nothing to fuse."""
    model = build({"tabular": embedders["tabular"]})

    output = model({"tabular": modalities["tabular"]})

    assert model.fusion is None
    assert output.representation.shape == (BATCH, FUSION_DIM)
    assert output.prediction.shape[0] == BATCH


def test_no_modality_at_all_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one modality"):
        build({})


def test_missing_modality_input_is_rejected(embedders, modalities) -> None:
    with pytest.raises(KeyError, match="missing input"):
        build(embedders)({"tabular": modalities["tabular"]})


def test_embedder_without_out_dim_is_rejected(embedders) -> None:
    with pytest.raises(AttributeError, match="must expose 'out_dim'"):
        build({"tabular": embedders["tabular"], "imaging": nn.Linear(4, 4)})


def test_projections_bring_modalities_to_one_width(embedders) -> None:
    """Differing embedder widths must not constrain the fusion method."""
    model = build(embedders)

    assert model.projections["tabular"].in_features == EMBED_DIMS["tabular"]
    assert model.projections["imaging"].in_features == EMBED_DIMS["imaging"]
    assert {p.out_features for p in model.projections.values()} == {FUSION_DIM}


def test_intermediate_fuses_features_before_predicting(embedders, modalities) -> None:
    output = build(embedders, "intermediate")(modalities)

    assert output.representation is not None
    assert output.representation.shape == (BATCH, FUSION_DIM)
    assert output.modality_predictions == {}


def test_late_predicts_per_modality_and_forms_no_joint_representation(embedders, modalities) -> None:
    output = build(embedders, "late")(modalities)

    assert output.representation is None
    assert set(output.modality_predictions) == set(RAW_DIMS)


def test_late_ignores_a_modality_that_is_absent(embedders, modalities) -> None:
    model = build(embedders, "late").eval()
    present = {
        "tabular": torch.ones(BATCH, dtype=torch.bool),
        "imaging": torch.zeros(BATCH, dtype=torch.bool),
    }

    output = model(modalities, present)

    assert torch.allclose(output.prediction, output.modality_predictions["tabular"], atol=1e-5)


def test_hybrid_carries_both_a_trunk_and_per_modality_heads(embedders, modalities) -> None:
    output = build(embedders, "hybrid")(modalities)

    assert output.representation is not None
    assert set(output.modality_predictions) == set(RAW_DIMS)


def test_hybrid_defaults_to_the_fused_trunk(embedders, modalities) -> None:
    """Without combine_predictions the modality heads are supervision only."""
    model = build(embedders, "hybrid").eval()

    output = model(modalities)

    assert model.head is not None
    assert torch.allclose(output.prediction, model.head(output.representation), atol=1e-6)


def test_hybrid_can_combine_predictions_as_well_as_features(embedders, modalities) -> None:
    trunk_only = build(embedders, "hybrid").eval()
    combined = build(embedders, "hybrid", combine_predictions=True).eval()
    # strict=False: the blend weights exist only on the combining model.
    combined.load_state_dict(trunk_only.state_dict(), strict=False)

    assert not torch.allclose(combined(modalities).prediction, trunk_only(modalities).prediction)


def test_hybrid_auxiliary_heads_can_be_disabled(embedders, modalities) -> None:
    output = build(embedders, "hybrid", auxiliary_heads=False)(modalities)

    assert output.modality_predictions == {}


def test_the_head_factory_decides_the_task(embedders, modalities) -> None:
    """Prediction logic stays injected, so the fusion API knows nothing about tasks."""
    model = MultimodalFusion(
        embedders,
        lambda width: nn.Linear(width, 3),
        stage="late",
        fusion_dim=FUSION_DIM,
    )

    assert model(modalities).prediction.shape == (BATCH, 3)


def test_auxiliary_loss_changes_the_objective(embedders, modalities, outcome) -> None:
    times, events = outcome
    output = build(embedders, "hybrid")(modalities)

    assert not torch.isclose(
        multimodal_cox_loss(output, times, events, auxiliary_weight=0.0),
        multimodal_cox_loss(output, times, events, auxiliary_weight=0.5),
    )


def test_auxiliary_loss_is_ignored_without_modality_predictions(embedders, modalities, outcome) -> None:
    times, events = outcome
    output = build(embedders, "intermediate")(modalities)

    assert torch.isclose(
        multimodal_cox_loss(output, times, events, auxiliary_weight=0.0),
        multimodal_cox_loss(output, times, events, auxiliary_weight=0.9),
    )


def test_modality_dropout_only_applies_while_training(embedders, modalities) -> None:
    model = build(embedders, "hybrid", modality_dropout=0.9).eval()

    assert torch.allclose(model(modalities).prediction, model(modalities).prediction)


def test_a_prebuilt_fusion_block_is_not_required(embedders, modalities) -> None:
    """The block is built from the method name, but its type is the documented one."""
    model = build(embedders, "intermediate", method="concat")

    assert isinstance(model.fusion, ConcatFusion)
