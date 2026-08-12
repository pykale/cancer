"""Tests for patient-level bag construction and collation."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from kalecancer.loaddata import WSIFeatureBagDataset, collate_bags
from tests.conftest import FEATURE_DIM


def test_sample_pools_all_slides_of_one_patient(cohort_bags) -> None:
    dataset = WSIFeatureBagDataset(cohort_bags, expected_dim=FEATURE_DIM)
    sample = next(s for s in (dataset[i] for i in range(len(dataset))) if s.patient_id == "003")

    # Patient 003's two slides hold 7 and 5 patches.
    assert sample.features.shape == (12, FEATURE_DIM)
    assert sample.coords.shape == (12, 2)
    assert sample.slide_ids == ("PrimaryTumor_HE_003", "PrimaryTumor_HE_003_a")
    assert sample.slide_index.tolist() == [0] * 7 + [1] * 5


def test_bags_keep_their_own_length(cohort_bags) -> None:
    dataset = WSIFeatureBagDataset(cohort_bags, expected_dim=FEATURE_DIM)

    assert [dataset[i].features.shape[0] for i in range(len(dataset))] == [10, 15, 12]


def test_survival_labels_travel_with_the_bag(cohort_bags) -> None:
    dataset = WSIFeatureBagDataset(cohort_bags, expected_dim=FEATURE_DIM)
    sample = dataset[0]

    assert sample.patient_id == "001"
    assert float(sample.duration) == 500.0
    assert float(sample.event) == 1.0


def test_max_patches_subsamples_features_coords_and_slide_index_together(cohort_bags) -> None:
    dataset = WSIFeatureBagDataset(cohort_bags, expected_dim=FEATURE_DIM, max_patches=4, seed=0)

    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample.features.shape[0] == 4
        assert sample.coords.shape[0] == 4
        assert sample.slide_index.shape[0] == 4


def test_subsampling_is_reproducible_for_a_given_seed(cohort_bags) -> None:
    first = WSIFeatureBagDataset(cohort_bags, expected_dim=FEATURE_DIM, max_patches=4, seed=7)[1]
    second = WSIFeatureBagDataset(cohort_bags, expected_dim=FEATURE_DIM, max_patches=4, seed=7)[1]

    assert torch.equal(first.coords, second.coords)


def test_bags_shorter_than_the_cap_are_untouched(cohort_bags) -> None:
    dataset = WSIFeatureBagDataset(cohort_bags, expected_dim=FEATURE_DIM, max_patches=1000)

    assert dataset[0].features.shape[0] == 10


def test_collate_keeps_bags_unpadded_and_stacks_labels(cohort_bags) -> None:
    dataset = WSIFeatureBagDataset(cohort_bags, expected_dim=FEATURE_DIM)
    batch = collate_bags([dataset[0], dataset[1], dataset[2]])

    assert len(batch) == 3
    assert [sample.features.shape[0] for sample in batch.samples] == [10, 15, 12]
    assert batch.duration.shape == (3,)
    assert batch.event.tolist() == [1.0, 0.0, 1.0]


def test_dataloader_yields_collated_batches(cohort_bags) -> None:
    dataset = WSIFeatureBagDataset(cohort_bags, expected_dim=FEATURE_DIM)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_bags, num_workers=0)

    batches = list(loader)

    assert [len(batch) for batch in batches] == [2, 1]


def test_dataset_rejects_unexpected_feature_dimension(cohort_bags) -> None:
    from kalecancer.loaddata import InvalidFeatureFileError

    dataset = WSIFeatureBagDataset(cohort_bags, expected_dim=1024)

    with pytest.raises(InvalidFeatureFileError, match="expected feature dimension 1024"):
        _ = dataset[0]
