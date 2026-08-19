"""Tests for grouping cohort rows into patch bags."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from kalecancer.loaddata import InvalidFeatureFileError, WSIFeatureDataset, collate_bags
from tests.conftest import FEATURE_DIM

SAMPLE_KEYS = {"features", "coords", "slide_index", "group_id", "slide_ids", "duration", "event"}


def test_sample_is_a_named_mapping(cohort) -> None:
    sample = WSIFeatureDataset(cohort, expected_dim=FEATURE_DIM)[0]

    assert set(sample) == SAMPLE_KEYS


def test_one_bag_per_group_not_per_row(cohort) -> None:
    dataset = WSIFeatureDataset(cohort, group_key="patient_id", expected_dim=FEATURE_DIM)

    assert len(dataset) == cohort["patient_id"].nunique() < len(cohort)


def test_rows_of_a_group_are_pooled_into_one_bag(cohort) -> None:
    dataset = WSIFeatureDataset(cohort, expected_dim=FEATURE_DIM)
    sample = next(dataset[i] for i in range(len(dataset)) if dataset[i]["group_id"] == "003")

    # Patient 003's two slides hold 7 and 5 patches.
    assert sample["features"].shape == (12, FEATURE_DIM)
    assert sample["coords"].shape == (12, 2)
    assert sample["slide_ids"] == ("PrimaryTumor_HE_003", "PrimaryTumor_HE_003_a")
    assert sample["slide_index"].tolist() == [0] * 7 + [1] * 5


def test_grouping_column_is_configurable(cohort) -> None:
    """Any column can define a bag; the dataset does not assume patients."""
    dataset = WSIFeatureDataset(cohort, group_key="slide_id", expected_dim=FEATURE_DIM)

    assert len(dataset) == len(cohort)
    assert all(len(dataset[i]["slide_ids"]) == 1 for i in range(len(dataset)))


def test_bags_keep_their_own_length(cohort) -> None:
    dataset = WSIFeatureDataset(cohort, expected_dim=FEATURE_DIM)

    assert [dataset[i]["features"].shape[0] for i in range(len(dataset))] == [10, 15, 12]


def test_labels_travel_with_the_bag(cohort) -> None:
    sample = WSIFeatureDataset(cohort, expected_dim=FEATURE_DIM)[0]

    assert sample["group_id"] == "001"
    assert float(sample["duration"]) == 500.0
    assert float(sample["event"]) == 1.0


def test_max_patches_subsamples_every_aligned_array(cohort) -> None:
    dataset = WSIFeatureDataset(cohort, expected_dim=FEATURE_DIM, max_patches=4, seed=0)

    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["features"].shape[0] == 4
        assert sample["coords"].shape[0] == 4
        assert sample["slide_index"].shape[0] == 4


def test_subsampling_is_reproducible_for_a_given_seed(cohort) -> None:
    first = WSIFeatureDataset(cohort, expected_dim=FEATURE_DIM, max_patches=4, seed=7)[1]
    second = WSIFeatureDataset(cohort, expected_dim=FEATURE_DIM, max_patches=4, seed=7)[1]

    assert torch.equal(first["coords"], second["coords"])


def test_bags_shorter_than_the_cap_are_untouched(cohort) -> None:
    dataset = WSIFeatureDataset(cohort, expected_dim=FEATURE_DIM, max_patches=1000)

    assert dataset[0]["features"].shape[0] == 10


def test_collate_keeps_bags_unpadded_and_stacks_labels(cohort) -> None:
    dataset = WSIFeatureDataset(cohort, expected_dim=FEATURE_DIM)

    batch = collate_bags([dataset[0], dataset[1], dataset[2]])

    assert [sample["features"].shape[0] for sample in batch["samples"]] == [10, 15, 12]
    assert batch["duration"].shape == (3,)
    assert batch["event"].tolist() == [1.0, 0.0, 1.0]


def test_dataloader_yields_collated_batches(cohort) -> None:
    dataset = WSIFeatureDataset(cohort, expected_dim=FEATURE_DIM)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_bags, num_workers=0)

    assert [len(batch["samples"]) for batch in loader] == [2, 1]


def test_unexpected_feature_dimension_is_rejected(cohort) -> None:
    dataset = WSIFeatureDataset(cohort, expected_dim=1024)

    with pytest.raises(InvalidFeatureFileError, match="expected feature dimension 1024"):
        _ = dataset[0]


def test_missing_columns_are_reported(cohort) -> None:
    with pytest.raises(KeyError, match="missing column"):
        WSIFeatureDataset(cohort.drop(columns=["duration"]))


def test_invalid_max_patches_is_rejected(cohort) -> None:
    with pytest.raises(ValueError, match="must be a positive number"):
        WSIFeatureDataset(cohort, max_patches=0)
