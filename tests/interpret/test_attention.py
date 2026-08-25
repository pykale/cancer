"""Tests that attention stays aligned with patch coordinates."""

from __future__ import annotations

import pytest
import torch

from kalecancer.interpret import attention_records, multimodal_attention, top_k_patches
from kalecancer.loaddata import WSIFeatureDataset
from kalecancer.loaddata.sample import PatientBatch
from kalecancer.model.embed import AttentionMIL, BagEncoder, MLPEmbedder
from tests.conftest import FEATURE_DIM


@pytest.fixture
def sample(cohort):
    return WSIFeatureDataset(cohort, expected_dim=FEATURE_DIM)[0]


def test_one_record_per_patch_carrying_its_coordinate(sample) -> None:
    attention = torch.rand(len(sample["coords"]))

    records = attention_records(sample, attention)

    assert len(records) == len(sample["coords"])
    for index, record in enumerate(records):
        assert record["x"] == int(sample["coords"][index, 0])
        assert record["y"] == int(sample["coords"][index, 1])
        assert record["attention"] == pytest.approx(float(attention[index]))


def test_records_identify_the_patient_and_source_slide(cohort) -> None:
    dataset = WSIFeatureDataset(cohort, expected_dim=FEATURE_DIM)
    multi_slide = next(
        sample for sample in map(dataset.__getitem__, range(len(dataset))) if sample["group_id"] == "003"
    )

    records = attention_records(multi_slide, torch.rand(len(multi_slide["coords"])))

    assert {record["patient_id"] for record in records} == {"003"}
    # Patches map back to whichever of the patient's slides they came from.
    assert [record["slide_id"] for record in records] == ["PrimaryTumor_HE_003"] * 7 + ["PrimaryTumor_HE_003_a"] * 5


def test_attention_follows_its_patch_when_instances_are_reordered(sample) -> None:
    """Permuting the bag must permute the coordinate/attention pairing identically."""
    attention = torch.rand(len(sample["coords"]))
    original = {(record["x"], record["y"]): record["attention"] for record in attention_records(sample, attention)}

    permutation = torch.randperm(len(attention))
    sample["features"] = sample["features"][permutation]
    sample["coords"] = sample["coords"][permutation]
    sample["slide_index"] = sample["slide_index"][permutation]
    permuted = attention_records(sample, attention[permutation])

    assert {(record["x"], record["y"]): record["attention"] for record in permuted} == original


def test_misaligned_attention_is_rejected(sample) -> None:
    with pytest.raises(ValueError, match="must stay aligned"):
        attention_records(sample, torch.rand(len(sample["coords"]) + 1))


def test_top_k_returns_the_most_attended_patches_first(sample) -> None:
    records = attention_records(sample, torch.linspace(0.0, 1.0, len(sample["coords"])))

    top = top_k_patches(records, k=3)

    assert len(top) == 3
    assert [record["attention"] for record in top] == sorted((record["attention"] for record in top), reverse=True)
    # The largest weight was assigned to the final patch.
    assert top[0]["x"] == int(sample["coords"][-1, 0])


def test_a_negative_top_k_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        top_k_patches([{"attention": 0.5}], k=-1)


class _FusedStub:
    """The minimum surface multimodal_attention reads: embedders and a predict call."""

    def __init__(self, embedders) -> None:
        self.model = type("Fusion", (), {"embedders": embedders})()
        self.seen = 0

    def predict_logits(self, batch):
        self.seen += 1
        return None


def test_multimodal_attention_pairs_weights_with_patients() -> None:
    encoder = BagEncoder(AttentionMIL(input_dim=FEATURE_DIM, hidden_dim=8, attention_dim=4))
    bags = [torch.randn(5, FEATURE_DIM), torch.randn(9, FEATURE_DIM)]
    encoder(bags)
    batch = PatientBatch(patient_id=["001", "002"], modalities={"imaging": bags}, present={})

    weights = multimodal_attention(_FusedStub({"imaging": encoder}), batch, "imaging")

    assert list(weights) == ["001", "002"]
    assert [len(w) for w in weights.values()] == [5, 9]
    assert all(torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-5) for w in weights.values())


def test_an_unknown_modality_is_refused() -> None:
    encoder = BagEncoder(AttentionMIL(input_dim=FEATURE_DIM, hidden_dim=8, attention_dim=4))
    batch = PatientBatch(patient_id=["001"], modalities={}, present={})

    with pytest.raises(KeyError, match="no modality"):
        multimodal_attention(_FusedStub({"imaging": encoder}), batch, "clinical")


def test_an_embedder_without_attention_is_refused() -> None:
    batch = PatientBatch(patient_id=["001"], modalities={}, present={})

    with pytest.raises(AttributeError, match="records no attention"):
        multimodal_attention(_FusedStub({"clinical": MLPEmbedder(4, 8)}), batch, "clinical")
