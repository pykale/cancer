"""Tests for attention MIL pooling."""

from __future__ import annotations

import pytest
import torch

from kalecancer.model.embed import AttentionMIL, BagEncoder

INPUT_DIM = 16
HIDDEN_DIM = 8


@pytest.fixture
def model() -> AttentionMIL:
    torch.manual_seed(0)
    return AttentionMIL(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, attention_dim=4, dropout=0.0).eval()


def test_pooling_returns_one_representation_per_bag(model: AttentionMIL) -> None:
    embedding, attention = model(torch.randn(25, INPUT_DIM))

    assert embedding.shape == (HIDDEN_DIM,)
    assert attention.shape == (25,)


def test_attention_has_one_weight_per_instance(model: AttentionMIL) -> None:
    for num_patches in (1, 7, 300):
        _, attention = model(torch.randn(num_patches, INPUT_DIM))
        assert attention.shape == (num_patches,)


def test_attention_weights_form_a_distribution(model: AttentionMIL) -> None:
    _, attention = model(torch.randn(40, INPUT_DIM))

    assert torch.isclose(attention.sum(), torch.tensor(1.0), atol=1e-5)
    assert bool((attention >= 0).all())


def test_pooling_is_permutation_equivariant(model: AttentionMIL) -> None:
    """The representation is order-invariant while attention follows its instance."""
    bag = torch.randn(20, INPUT_DIM)
    permutation = torch.randperm(20)

    embedding, attention = model(bag)
    permuted_embedding, permuted_attention = model(bag[permutation])

    assert torch.allclose(embedding, permuted_embedding, atol=1e-5)
    assert torch.allclose(attention[permutation], permuted_attention, atol=1e-5)


def test_forward_bags_handles_variable_lengths(model: AttentionMIL) -> None:
    bags = [torch.randn(n, INPUT_DIM) for n in (5, 31, 12)]

    embeddings, attentions = model.forward_bags(bags)

    assert embeddings.shape == (3, HIDDEN_DIM)
    assert [attention.shape[0] for attention in attentions] == [5, 31, 12]


def test_non_gated_variant_produces_the_same_shapes() -> None:
    model = AttentionMIL(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, gated=False).eval()

    embedding, attention = model(torch.randn(9, INPUT_DIM))

    assert embedding.shape == (HIDDEN_DIM,)
    assert attention.shape == (9,)


def test_empty_bag_is_rejected(model: AttentionMIL) -> None:
    with pytest.raises(ValueError, match="empty bag"):
        model(torch.zeros(0, INPUT_DIM))


def test_wrongly_shaped_bag_is_rejected(model: AttentionMIL) -> None:
    with pytest.raises(ValueError, match="expected a 2D bag"):
        model(torch.randn(2, 5, INPUT_DIM))


def test_bag_encoder_presents_the_plain_encoder_interface(model: AttentionMIL) -> None:
    """Fusion expects one vector per patient, not the MIL (embedding, attention) pair."""
    encoder = BagEncoder(model)

    embeddings = encoder([torch.randn(n, INPUT_DIM) for n in (5, 11)])

    assert embeddings.shape == (2, HIDDEN_DIM)
    assert encoder.output_dim == HIDDEN_DIM


def test_bag_encoder_keeps_attention_available_for_interpretation(model: AttentionMIL) -> None:
    encoder = BagEncoder(model)

    encoder([torch.randn(n, INPUT_DIM) for n in (5, 11)])

    assert [len(attention) for attention in encoder.last_attention] == [5, 11]


def test_bag_encoder_registers_the_wrapped_parameters(model: AttentionMIL) -> None:
    encoder = BagEncoder(model)

    assert sum(p.numel() for p in encoder.parameters()) == sum(p.numel() for p in model.parameters())


def test_bag_encoder_detaches_attention_from_the_graph(model: AttentionMIL) -> None:
    """Holding graph-linked attention would retain each bag's activations."""
    encoder = BagEncoder(model)

    encoder([torch.randn(n, INPUT_DIM) for n in (5, 11)]).sum().backward()

    assert all(not a.requires_grad and a.grad_fn is None for a in encoder.last_attention)
