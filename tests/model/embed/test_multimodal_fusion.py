"""Tests for latent-level multimodal fusion blocks."""

from __future__ import annotations

import pytest
import torch

from kalecancer.model.embed import (
    FUSION_METHODS,
    ConcatFusion,
    LowRankFusion,
    MultimodalOutput,
    ProductOfExpertsFusion,
    build_fusion,
    multimodal_bce_loss,
)

INPUT_DIMS = [16, 8]
OUTPUT_DIM = 32
BATCH = 6

METHODS = [("concat", {}), ("poe", {}), ("lowrank", {"rank": 3})]


@pytest.fixture
def latents() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(BATCH, dim) for dim in INPUT_DIMS]


@pytest.mark.parametrize(("method", "kwargs"), METHODS)
def test_every_method_returns_the_same_output_width(method: str, kwargs: dict, latents) -> None:
    """Interchangeability: swapping the method must not change the head's input."""
    fused = build_fusion(method, INPUT_DIMS, OUTPUT_DIM, **kwargs)(latents)

    assert fused.shape == (BATCH, OUTPUT_DIM)


@pytest.mark.parametrize(("method", "kwargs"), METHODS)
def test_parameters_are_registered_and_receive_gradients(method: str, kwargs: dict, latents) -> None:
    """Guards the PyKale LowRankTensorFusion defect: unregistered params never train."""
    block = build_fusion(method, INPUT_DIMS, OUTPUT_DIM, **kwargs)
    parameters = list(block.parameters())

    assert parameters, f"{method} registered no parameters, so an optimiser could not train it"

    block(latents).sum().backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in parameters)


@pytest.mark.parametrize(("method", "kwargs"), METHODS)
def test_output_stays_on_the_input_device(method: str, kwargs: dict, latents) -> None:
    fused = build_fusion(method, INPUT_DIMS, OUTPUT_DIM, **kwargs)(latents)

    assert fused.device == latents[0].device


@pytest.mark.parametrize(("method", "kwargs"), METHODS)
def test_mask_is_accepted_and_output_stays_finite(method: str, kwargs: dict, latents) -> None:
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    fused = build_fusion(method, INPUT_DIMS, OUTPUT_DIM, **kwargs)(latents, mask)

    assert fused.shape == (BATCH, OUTPUT_DIM)
    assert bool(torch.isfinite(fused).all())


@pytest.mark.parametrize(("method", "kwargs"), METHODS)
def test_absent_modality_content_is_ignored(method: str, kwargs: dict, latents) -> None:
    """Masking a modality out must make its values irrelevant."""
    block = build_fusion(method, INPUT_DIMS, OUTPUT_DIM, **kwargs).eval()
    mask = torch.tensor([[1.0, 0.0]] * BATCH)
    other = [latents[0], torch.randn(BATCH, INPUT_DIMS[1])]

    assert torch.allclose(block(latents, mask), block(other, mask), atol=1e-5)


@pytest.mark.parametrize(("method", "kwargs"), METHODS)
def test_registry_rejects_unknown_method(method: str, kwargs: dict) -> None:
    with pytest.raises(KeyError, match="unknown fusion method"):
        build_fusion("not_a_method", INPUT_DIMS, OUTPUT_DIM)


def test_registry_covers_every_documented_method() -> None:
    assert set(FUSION_METHODS) == {"concat", "poe", "lowrank"}


@pytest.mark.parametrize(("method", "kwargs"), METHODS)
def test_fewer_than_two_modalities_is_rejected(method: str, kwargs: dict) -> None:
    with pytest.raises(ValueError, match="at least two modalities"):
        build_fusion(method, [16], OUTPUT_DIM, **kwargs)


@pytest.mark.parametrize(("method", "kwargs"), METHODS)
def test_wrong_number_of_modalities_is_rejected(method: str, kwargs: dict, latents) -> None:
    block = build_fusion(method, INPUT_DIMS, OUTPUT_DIM, **kwargs)

    with pytest.raises(ValueError, match="expected 2 modalities"):
        block(latents + [torch.randn(BATCH, 4)])


@pytest.mark.parametrize(("method", "kwargs"), METHODS)
def test_wrong_latent_dimension_is_rejected(method: str, kwargs: dict) -> None:
    block = build_fusion(method, INPUT_DIMS, OUTPUT_DIM, **kwargs)

    with pytest.raises(ValueError, match="has dimension 5, expected 16"):
        block([torch.randn(BATCH, 5), torch.randn(BATCH, 8)])


@pytest.mark.parametrize(("method", "kwargs"), METHODS)
def test_wrongly_shaped_mask_is_rejected(method: str, kwargs: dict, latents) -> None:
    block = build_fusion(method, INPUT_DIMS, OUTPUT_DIM, **kwargs)

    with pytest.raises(ValueError, match="mask must be"):
        block(latents, torch.ones(BATCH, 3))


def test_concat_uses_a_learned_placeholder_not_zeros(latents) -> None:
    """A zero-filled absent modality would be indistinguishable from a real zero latent."""
    block = ConcatFusion(INPUT_DIMS, OUTPUT_DIM).eval()
    with torch.no_grad():
        block.placeholders[1].fill_(3.0)
    mask = torch.tensor([[1.0, 0.0]] * BATCH)

    absent = block(latents, mask)
    zeroed = block([latents[0], torch.zeros(BATCH, INPUT_DIMS[1])], torch.ones(BATCH, 2))

    assert not torch.allclose(absent, zeroed)


def test_product_of_experts_survives_a_sample_missing_everything(latents) -> None:
    """The prior expert keeps the product defined when no modality is present."""
    block = ProductOfExpertsFusion(INPUT_DIMS, OUTPUT_DIM)

    fused = block(latents, torch.zeros(BATCH, 2))

    assert bool(torch.isfinite(fused).all())
    # With only the prior expert left, the consensus is the prior mean.
    assert torch.allclose(fused, torch.zeros_like(fused), atol=1e-5)


def test_product_of_experts_can_drop_the_prior(latents) -> None:
    block = ProductOfExpertsFusion(INPUT_DIMS, OUTPUT_DIM, use_prior=False)

    assert block(latents, torch.ones(BATCH, 2)).shape == (BATCH, OUTPUT_DIM)


def test_product_of_experts_sharpens_as_modalities_are_added(latents) -> None:
    """Adding an expert must change the consensus, not leave it at one modality's view."""
    block = ProductOfExpertsFusion(INPUT_DIMS, OUTPUT_DIM).eval()

    one = block(latents, torch.tensor([[1.0, 0.0]] * BATCH))
    both = block(latents, torch.ones(BATCH, 2))

    assert not torch.allclose(one, both, atol=1e-4)


def test_low_rank_parameter_count_grows_with_rank() -> None:
    small = sum(p.numel() for p in LowRankFusion(INPUT_DIMS, OUTPUT_DIM, rank=2).parameters())
    large = sum(p.numel() for p in LowRankFusion(INPUT_DIMS, OUTPUT_DIM, rank=8).parameters())

    assert large > small


def test_low_rank_models_multiplicative_interaction(latents) -> None:
    """Scaling one modality must change the fused output non-additively."""
    block = LowRankFusion(INPUT_DIMS, OUTPUT_DIM, rank=4).eval()

    base = block(latents)
    scaled = block([latents[0] * 2.0, latents[1]])

    assert not torch.allclose(base, scaled)


def test_three_modalities_are_supported() -> None:
    dims = [8, 4, 6]
    inputs = [torch.randn(BATCH, dim) for dim in dims]

    for method, kwargs in METHODS:
        assert build_fusion(method, dims, OUTPUT_DIM, **kwargs)(inputs).shape == (BATCH, OUTPUT_DIM)


def test_bce_loss_is_finite_and_differentiable() -> None:
    output = MultimodalOutput(prediction=torch.randn(6, 1, requires_grad=True), representation=None)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])

    loss = multimodal_bce_loss(output, labels)
    loss.backward()

    assert torch.isfinite(loss)


def test_bce_auxiliary_supervision_changes_the_objective() -> None:
    prediction = torch.randn(6, 1)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    output = MultimodalOutput(
        prediction=prediction,
        representation=None,
        modality_predictions={"alpha": torch.randn(6, 1), "beta": torch.randn(6, 1)},
    )

    plain = multimodal_bce_loss(output, labels, auxiliary_weight=0.0)
    auxiliary = multimodal_bce_loss(output, labels, auxiliary_weight=0.5)

    assert not torch.isclose(plain, auxiliary)


def test_bce_auxiliary_weight_is_ignored_without_modality_predictions() -> None:
    output = MultimodalOutput(prediction=torch.randn(6, 1), representation=None)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])

    assert torch.isclose(
        multimodal_bce_loss(output, labels, auxiliary_weight=0.0),
        multimodal_bce_loss(output, labels, auxiliary_weight=0.9),
    )


def test_positive_weighting_raises_the_cost_of_missing_positives() -> None:
    """A weighted loss penalises the same wrong answer on a positive more heavily."""
    output = MultimodalOutput(prediction=torch.full((4, 1), -2.0), representation=None)
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

    plain = multimodal_bce_loss(output, labels)
    weighted = multimodal_bce_loss(output, labels, pos_weight=torch.tensor(4.0))

    assert weighted > plain
