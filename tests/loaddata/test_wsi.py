"""Tests for reading patch bags into a cohort.

A bag is one modality among any number, so these build the same
:class:`~kalecancer.loaddata.MultimodalDataset` an experiment would, with a single
:class:`~kalecancer.loaddata.FeatureBagSource`.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from kalecancer.loaddata import (
    ColumnTarget,
    FeatureBagSource,
    InvalidFeatureFileError,
    MultimodalDataset,
    collate_ragged,
)
from tests.conftest import FEATURE_DIM

MODALITY = "wsi"
METADATA_KEYS = {"coords", "slide_index", "slide_ids"}


def bags_of(cohort, group_key: str = "patient_id") -> dict[str, list]:
    """The slide-level cohort table collapsed to one entry per group."""
    return {group: list(rows["path"]) for group, rows in cohort.groupby(group_key)}


def make_dataset(cohort, group_key: str = "patient_id", max_patches: int | None = None, seed: int = 0):
    paths = bags_of(cohort, group_key)
    source = FeatureBagSource(paths, feature_dim=FEATURE_DIM, max_patches=max_patches, seed=seed, with_coordinates=True)
    target = ColumnTarget(
        cohort.drop_duplicates(group_key), columns={"time": "duration", "event": "event"}, id_column=group_key
    )
    return MultimodalDataset(sorted(paths), {MODALITY: source}, target=target)


def test_sample_is_a_patient_sample(cohort) -> None:
    """The same type every other source yields, which is what lets one trainer serve them."""
    sample = make_dataset(cohort)[0]

    assert set(sample.modalities) == {MODALITY}
    assert set(sample.target) == {"time", "event"}
    assert set(sample.metadata) == METADATA_KEYS


def test_one_bag_per_group_not_per_row(cohort) -> None:
    dataset = make_dataset(cohort)

    assert len(dataset) == cohort["patient_id"].nunique() < len(cohort)


def test_rows_of_a_group_are_pooled_into_one_bag(cohort) -> None:
    dataset = make_dataset(cohort)
    sample = next(dataset[i] for i in range(len(dataset)) if dataset[i].patient_id == "003")

    # Patient 003's two slides hold 7 and 5 patches.
    assert sample.modalities[MODALITY].shape == (12, FEATURE_DIM)
    assert sample.metadata["coords"].shape == (12, 2)
    assert sample.metadata["slide_ids"] == ("PrimaryTumor_HE_003", "PrimaryTumor_HE_003_a")
    assert sample.metadata["slide_index"].tolist() == [0] * 7 + [1] * 5


def test_grouping_column_is_configurable(cohort) -> None:
    """Any column can define a bag; nothing here assumes patients."""
    dataset = make_dataset(cohort, group_key="slide_id")

    assert len(dataset) == len(cohort)
    assert all(len(dataset[i].metadata["slide_ids"]) == 1 for i in range(len(dataset)))


def test_bags_keep_their_own_length(cohort) -> None:
    dataset = make_dataset(cohort)

    assert [dataset[i].modalities[MODALITY].shape[0] for i in range(len(dataset))] == [10, 15, 12]


def test_labels_travel_with_the_bag(cohort) -> None:
    sample = make_dataset(cohort)[0]

    assert sample.patient_id == "001"
    assert float(sample.target["time"]) == 500.0
    assert float(sample.target["event"]) == 1.0


def test_max_patches_subsamples_every_aligned_array(cohort) -> None:
    dataset = make_dataset(cohort, max_patches=4)

    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample.modalities[MODALITY].shape[0] == 4
        assert sample.metadata["coords"].shape[0] == 4
        assert sample.metadata["slide_index"].shape[0] == 4


def test_subsampling_is_reproducible_for_a_given_seed(cohort) -> None:
    first = make_dataset(cohort, max_patches=4, seed=7)[1]
    second = make_dataset(cohort, max_patches=4, seed=7)[1]

    assert torch.equal(first.metadata["coords"], second.metadata["coords"])


def test_bags_shorter_than_the_cap_are_untouched(cohort) -> None:
    dataset = make_dataset(cohort, max_patches=1000)

    assert dataset[0].modalities[MODALITY].shape[0] == 10


def test_collate_keeps_bags_unpadded_and_stacks_labels(cohort) -> None:
    dataset = make_dataset(cohort)

    batch = collate_ragged([dataset[0], dataset[1], dataset[2]])

    assert [bag.shape[0] for bag in batch.modalities[MODALITY]] == [10, 15, 12]
    assert batch.target["time"].shape == (3,)
    assert batch.target["event"].tolist() == [1.0, 0.0, 1.0]
    assert [len(coords) for coords in batch.metadata["coords"]] == [10, 15, 12]


def test_dataloader_yields_collated_batches(cohort) -> None:
    loader = DataLoader(make_dataset(cohort), batch_size=2, collate_fn=collate_ragged, num_workers=0)

    assert [len(batch) for batch in loader] == [2, 1]


def test_collating_an_empty_batch_is_refused() -> None:
    with pytest.raises(ValueError, match="empty list of samples"):
        collate_ragged([])


def test_unexpected_feature_dimension_is_rejected(cohort) -> None:
    """A width mismatch means the wrong encoder, and must not be padded over."""
    source = FeatureBagSource(bags_of(cohort), feature_dim=FEATURE_DIM + 1)
    dataset = MultimodalDataset(sorted(bags_of(cohort)), {MODALITY: source})

    with pytest.raises(InvalidFeatureFileError):
        _ = dataset[0]


def test_a_patient_with_no_slides_is_marked_absent(cohort) -> None:
    """Missing data is ordinary: a placeholder keeps the batch rectangular."""
    source = FeatureBagSource(bags_of(cohort), feature_dim=FEATURE_DIM)
    dataset = MultimodalDataset(["nobody"], {MODALITY: source})

    sample = dataset[0]

    assert not bool(sample.present[MODALITY])
    assert sample.modalities[MODALITY].shape == (1, FEATURE_DIM)
