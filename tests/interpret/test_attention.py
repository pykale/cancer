"""Tests for attention interpretation.

Attention is only useful if each weight stays attached to the patch it came from, so
most of these check alignment rather than values: a permuted, truncated or
mismatched join is the failure that would silently produce a plausible-looking but
wrong heatmap.
"""

from __future__ import annotations

import pytest
import torch

from kalecancer.interpret import attention_records, bag_attention, batch_records, top_k_patches
from kalecancer.loaddata.multimodal_access import PatientBatch
from kalecancer.model.embed import BagEncoder, MLPEmbedder
from kalecancer.model.layers import AttentionMIL

FEATURE_DIM = 8
MODALITY = "imaging"


def make_batch(sizes: tuple[int, ...] = (5, 9), slides: int = 1) -> PatientBatch:
    """A bag batch carrying the provenance an attention export joins to."""
    bags = [torch.randn(n, FEATURE_DIM) for n in sizes]
    return PatientBatch(
        patient_id=[f"{i:03d}" for i in range(len(sizes))],
        modalities={MODALITY: bags},
        present={MODALITY: torch.ones(len(sizes), dtype=torch.bool)},
        metadata={
            "coords": [torch.arange(n * 2).reshape(n, 2) for n in sizes],
            "slide_index": [torch.zeros(n, dtype=torch.long) for n in sizes],
            "slide_ids": [tuple(f"slide{s}" for s in range(slides)) for _ in sizes],
        },
    )


class _TrainerStub:
    """The minimum surface the attention helpers read: embedders and a predict call."""

    def __init__(self, embedders) -> None:
        self.embedders = embedders
        self.seen = 0

    def predict(self, batch):
        self.seen += 1
        for name, embedder in self.embedders.items():
            embedder(batch.modalities[name])
        return None


def bag_model() -> _TrainerStub:
    return _TrainerStub({MODALITY: BagEncoder(AttentionMIL(input_dim=FEATURE_DIM, hidden_dim=8, attention_dim=4))})


# --------------------------------------------------------------------------- #
# attention_records
# --------------------------------------------------------------------------- #


def test_every_patch_gets_a_record() -> None:
    records = attention_records(
        "001", torch.rand(5), torch.arange(10).reshape(5, 2), torch.zeros(5, dtype=torch.long), ("s",)
    )

    assert len(records) == 5
    assert set(records[0]) == {"patient_id", "slide_id", "x", "y", "attention"}


def test_each_weight_keeps_its_own_coordinate() -> None:
    """The join that matters: permuting the weights must move them with their patches."""
    coords = torch.arange(10).reshape(5, 2)
    slide_index = torch.zeros(5, dtype=torch.long)
    attention = torch.linspace(0.0, 1.0, 5)

    original = {
        (r["x"], r["y"]): r["attention"] for r in attention_records("001", attention, coords, slide_index, ("s",))
    }
    permutation = torch.tensor([4, 0, 3, 1, 2])
    permuted = attention_records("001", attention[permutation], coords, slide_index, ("s",))

    assert [r["attention"] for r in permuted] == attention[permutation].tolist()
    assert original != {(r["x"], r["y"]): r["attention"] for r in permuted}


def test_a_patch_names_the_slide_it_came_from() -> None:
    slide_index = torch.tensor([0, 0, 1, 1, 1])
    records = attention_records("001", torch.rand(5), torch.arange(10).reshape(5, 2), slide_index, ("a", "b"))

    assert [r["slide_id"] for r in records] == ["a", "a", "b", "b", "b"]


def test_a_length_mismatch_is_refused() -> None:
    """Silently truncating would place weights on the wrong patches."""
    with pytest.raises(ValueError, match="stay aligned"):
        attention_records(
            "001", torch.rand(6), torch.arange(10).reshape(5, 2), torch.zeros(5, dtype=torch.long), ("s",)
        )


# --------------------------------------------------------------------------- #
# top_k_patches
# --------------------------------------------------------------------------- #


def test_top_k_returns_the_most_attended_first() -> None:
    records = attention_records(
        "001", torch.linspace(0.0, 1.0, 5), torch.arange(10).reshape(5, 2), torch.zeros(5, dtype=torch.long), ("s",)
    )

    top = top_k_patches(records, k=2)

    assert [r["attention"] for r in top] == sorted((r["attention"] for r in records), reverse=True)[:2]


def test_a_negative_k_is_refused() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        top_k_patches([], k=-1)


# --------------------------------------------------------------------------- #
# bag_attention and batch_records
# --------------------------------------------------------------------------- #


def test_weights_are_paired_with_the_patients_they_came_from() -> None:
    batch = make_batch()

    weights = bag_attention(bag_model(), batch, MODALITY)

    assert list(weights) == ["000", "001"]
    assert [len(w) for w in weights.values()] == [5, 9]
    assert all(torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-5) for w in weights.values())


def test_an_unknown_modality_is_refused() -> None:
    with pytest.raises(KeyError, match="no modality"):
        bag_attention(bag_model(), make_batch(), "clinical")


def test_an_embedder_without_attention_is_refused() -> None:
    """An MLP records nothing, and saying so beats returning an empty map."""
    batch = PatientBatch(
        patient_id=["001"],
        modalities={"clinical": torch.randn(1, 4)},
        present={"clinical": torch.ones(1, dtype=torch.bool)},
    )

    with pytest.raises(AttributeError, match="records no attention"):
        bag_attention(_TrainerStub({"clinical": MLPEmbedder(4, 8)}), batch, "clinical")


def test_batch_records_join_weights_to_every_patients_patches() -> None:
    batch = make_batch(sizes=(5, 9))

    records = batch_records(bag_model(), batch, MODALITY)

    assert list(records) == ["000", "001"]
    assert [len(r) for r in records.values()] == [5, 9]


def test_a_batch_without_provenance_is_refused() -> None:
    """Attention with no coordinates cannot be placed on a slide; say so."""
    batch = make_batch()
    batch.metadata = {}

    with pytest.raises(KeyError, match="missing"):
        batch_records(bag_model(), batch, MODALITY)
