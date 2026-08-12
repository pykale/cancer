"""Tests that attention stays aligned with patch coordinates."""

from __future__ import annotations

import pytest
import torch

from kalecancer.interpret import attention_records, top_k_patches
from kalecancer.loaddata import WSIFeatureBagDataset
from tests.conftest import FEATURE_DIM


@pytest.fixture
def sample(cohort_bags):
    return WSIFeatureBagDataset(cohort_bags, expected_dim=FEATURE_DIM)[0]


def test_one_record_per_patch_carrying_its_coordinate(sample) -> None:
    attention = torch.rand(len(sample.coords))

    records = attention_records(sample, attention)

    assert len(records) == len(sample.coords)
    for index, record in enumerate(records):
        assert record["x"] == int(sample.coords[index, 0])
        assert record["y"] == int(sample.coords[index, 1])
        assert record["attention"] == pytest.approx(float(attention[index]))


def test_records_identify_the_patient_and_source_slide(cohort_bags) -> None:
    multi_slide = next(
        WSIFeatureBagDataset(cohort_bags, expected_dim=FEATURE_DIM)[i]
        for i in range(len(cohort_bags))
        if cohort_bags[i].patient_id == "003"
    )

    records = attention_records(multi_slide, torch.rand(len(multi_slide.coords)))

    assert {record["patient_id"] for record in records} == {"003"}
    # Patches map back to whichever of the patient's slides they came from.
    assert [record["slide_id"] for record in records] == ["PrimaryTumor_HE_003"] * 7 + ["PrimaryTumor_HE_003_a"] * 5


def test_attention_follows_its_patch_when_instances_are_reordered(sample) -> None:
    """Permuting the bag must permute the coordinate/attention pairing identically."""
    attention = torch.rand(len(sample.coords))
    original = {(record["x"], record["y"]): record["attention"] for record in attention_records(sample, attention)}

    permutation = torch.randperm(len(attention))
    sample.features = sample.features[permutation]
    sample.coords = sample.coords[permutation]
    sample.slide_index = sample.slide_index[permutation]
    permuted = attention_records(sample, attention[permutation])

    assert {(record["x"], record["y"]): record["attention"] for record in permuted} == original


def test_misaligned_attention_is_rejected(sample) -> None:
    with pytest.raises(ValueError, match="must stay aligned"):
        attention_records(sample, torch.rand(len(sample.coords) + 1))


def test_top_k_returns_the_most_attended_patches_first(sample) -> None:
    records = attention_records(sample, torch.linspace(0.0, 1.0, len(sample.coords)))

    top = top_k_patches(records, k=3)

    assert len(top) == 3
    assert [record["attention"] for record in top] == sorted((record["attention"] for record in top), reverse=True)
    # The largest weight was assigned to the final patch.
    assert top[0]["x"] == int(sample.coords[-1, 0])
