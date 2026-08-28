"""Tests for assembling a cohort from any combination of modality sources.

The point of the design is that the number and kind of modalities are data, not
code, so most of these are the same assertions over different dictionaries:
tabular-only, imaging-only, two tabular, two imaging, and a mixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from kalecancer.loaddata import (
    ColumnTarget,
    FeatureBagSource,
    ModalitySource,
    MultimodalDataset,
    VectorSource,
    collate_ragged,
)
from tests.conftest import FEATURE_DIM, write_bag

PATIENTS = ["001", "002", "003"]


@pytest.fixture
def vectors() -> dict[str, torch.Tensor]:
    return {patient: torch.randn(6) for patient in PATIENTS}


@pytest.fixture
def bags(tmp_path) -> dict[str, list]:
    """Per-patient feature files, of deliberately different lengths."""
    paths = {}
    for count, patient in zip((4, 7, 5), PATIENTS, strict=True):
        path = tmp_path / f"slide_{patient}.h5"
        write_bag(path, num_patches=count, feature_dim=FEATURE_DIM)
        paths[patient] = [path]
    return paths


@pytest.fixture
def target() -> ColumnTarget:
    frame = pd.DataFrame({"patient_id": PATIENTS, "duration": [100.0, 200.0, 300.0], "event": [1.0, 0.0, 1.0]})
    return ColumnTarget(frame, columns={"time": "duration", "event": "event"})


# --------------------------------------------------------------------------- #
# Arbitrary combinations
# --------------------------------------------------------------------------- #


def test_a_single_tabular_modality(vectors) -> None:
    dataset = MultimodalDataset(PATIENTS, {"clinical": VectorSource(vectors)})

    assert set(dataset[0].modalities) == {"clinical"}


def test_two_tabular_modalities(vectors) -> None:
    """Clinical plus blood, say: nothing about the dataset says they differ."""
    dataset = MultimodalDataset(PATIENTS, {"clinical": VectorSource(vectors), "blood": VectorSource(vectors)})

    assert set(dataset[0].modalities) == {"clinical", "blood"}


def test_two_imaging_modalities(bags) -> None:
    """Two slide regions, which a design built around one bag could not express."""
    dataset = MultimodalDataset(
        PATIENTS,
        {
            "primary": FeatureBagSource(bags, feature_dim=FEATURE_DIM),
            "lymph_node": FeatureBagSource(bags, feature_dim=FEATURE_DIM),
        },
    )

    sample = dataset[0]
    assert set(sample.modalities) == {"primary", "lymph_node"}
    assert sample.modalities["primary"].shape == (4, FEATURE_DIM)


def test_imaging_and_tabular_together(vectors, bags) -> None:
    dataset = MultimodalDataset(
        PATIENTS, {"clinical": VectorSource(vectors), "wsi": FeatureBagSource(bags, feature_dim=FEATURE_DIM)}
    )

    sample = dataset[0]
    assert sample.modalities["clinical"].shape == (6,)
    assert sample.modalities["wsi"].shape == (4, FEATURE_DIM)


def test_four_modalities_of_both_kinds(vectors, bags) -> None:
    """Adding a modality is adding a dictionary entry, not an API."""
    dataset = MultimodalDataset(
        PATIENTS,
        {
            "clinical": VectorSource(vectors),
            "blood": VectorSource(vectors),
            "primary": FeatureBagSource(bags, feature_dim=FEATURE_DIM),
            "lymph_node": FeatureBagSource(bags, feature_dim=FEATURE_DIM),
        },
    )

    assert set(dataset[0].modalities) == {"clinical", "blood", "primary", "lymph_node"}


def test_a_cohort_needs_at_least_one_source() -> None:
    with pytest.raises(ValueError, match="at least one modality source"):
        MultimodalDataset(PATIENTS, {})


# --------------------------------------------------------------------------- #
# Missing modalities
# --------------------------------------------------------------------------- #


def test_a_missing_value_is_marked_absent_and_placeheld(vectors, bags) -> None:
    """Rectangular batches without pretending the zeros are evidence."""
    partial = {patient: vectors[patient] for patient in PATIENTS[:1]}
    dataset = MultimodalDataset(
        PATIENTS, {"clinical": VectorSource(partial, width=6), "wsi": FeatureBagSource(bags, feature_dim=FEATURE_DIM)}
    )

    present, absent = dataset[0], dataset[1]

    assert bool(present.present["clinical"])
    assert not bool(absent.present["clinical"])
    assert torch.equal(absent.modalities["clinical"], torch.zeros(6))
    assert bool(absent.present["wsi"]), "the other modality is unaffected"


def test_an_absent_bag_still_has_a_poolable_shape(vectors) -> None:
    dataset = MultimodalDataset(PATIENTS, {"wsi": FeatureBagSource({}, feature_dim=FEATURE_DIM)})

    sample = dataset[0]

    assert not bool(sample.present["wsi"])
    assert sample.modalities["wsi"].shape == (1, FEATURE_DIM)


def test_an_empty_vector_source_needs_a_declared_width() -> None:
    """There is no shape to infer, and guessing one would batch wrongly."""
    with pytest.raises(ValueError, match="width must be given"):
        VectorSource({})


# --------------------------------------------------------------------------- #
# Targets and provenance
# --------------------------------------------------------------------------- #


def test_the_target_travels_with_the_patient(vectors, target) -> None:
    dataset = MultimodalDataset(PATIENTS, {"clinical": VectorSource(vectors)}, target=target)

    assert float(dataset[1].target["time"]) == 200.0
    assert float(dataset[1].target["event"]) == 0.0


def test_an_unlabelled_cohort_carries_no_target(vectors) -> None:
    dataset = MultimodalDataset(PATIENTS, {"clinical": VectorSource(vectors)})

    assert dataset[0].target == {}


def test_column_target_renames_explicitly() -> None:
    """The cohort column and the batch key are different vocabularies."""
    frame = pd.DataFrame({"patient_id": PATIENTS, "recurred": [1.0, 0.0, 1.0]})

    target = ColumnTarget(frame, columns={"label": "recurred"})

    assert set(target.for_("001")) == {"label"}
    assert float(target.for_("001")["label"]) == 1.0


def test_column_target_refuses_a_missing_column() -> None:
    frame = pd.DataFrame({"patient_id": PATIENTS})

    with pytest.raises(KeyError, match="no column"):
        ColumnTarget(frame, columns={"label": "absent"})


def test_column_target_refuses_duplicate_identifiers() -> None:
    """A repeated id means two patients share a row, which silently mislabels one."""
    frame = pd.DataFrame({"patient_id": ["001", "001"], "event": [1.0, 0.0]})

    with pytest.raises(KeyError, match="must be unique"):
        ColumnTarget(frame, columns={"event": "event"})


def test_provenance_is_carried_only_when_asked_for(bags) -> None:
    without = MultimodalDataset(PATIENTS, {"wsi": FeatureBagSource(bags, feature_dim=FEATURE_DIM)})
    with_coords = MultimodalDataset(
        PATIENTS, {"wsi": FeatureBagSource(bags, feature_dim=FEATURE_DIM, with_coordinates=True)}
    )

    assert without[0].metadata == {}
    assert set(with_coords[0].metadata) == {"coords", "slide_index", "slide_ids"}


def test_metadata_from_selects_which_sources_contribute(vectors, bags) -> None:
    """Recording provenance re-reads a bag, so a caller can decline to pay for it."""
    sources = {
        "clinical": VectorSource(vectors),
        "wsi": FeatureBagSource(bags, feature_dim=FEATURE_DIM, with_coordinates=True),
    }

    dataset = MultimodalDataset(PATIENTS, sources, metadata_from=["clinical"])

    assert dataset[0].metadata == {}


def test_metadata_from_refuses_an_unknown_source(vectors) -> None:
    with pytest.raises(ValueError, match="not sources"):
        MultimodalDataset(PATIENTS, {"clinical": VectorSource(vectors)}, metadata_from=["imaging"])


# --------------------------------------------------------------------------- #
# Collation
# --------------------------------------------------------------------------- #


def test_a_mixed_batch_stacks_vectors_and_leaves_bags_ragged(vectors, bags, target) -> None:
    dataset = MultimodalDataset(
        PATIENTS,
        {"clinical": VectorSource(vectors), "wsi": FeatureBagSource(bags, feature_dim=FEATURE_DIM)},
        target=target,
    )

    batch = collate_ragged([dataset[i] for i in range(len(dataset))])

    clinical, wsi = batch.modalities["clinical"], batch.modalities["wsi"]
    assert isinstance(clinical, torch.Tensor), "a flat vector stacks"
    assert isinstance(wsi, list), "a bag stays ragged"
    assert clinical.shape == (3, 6)
    assert [bag.shape[0] for bag in wsi] == [4, 7, 5]
    assert batch.target["time"].tolist() == [100.0, 200.0, 300.0]


def test_subsampling_is_reproducible_for_a_given_seed(bags) -> None:
    def first_bag(seed: int) -> torch.Tensor:
        source = FeatureBagSource(bags, feature_dim=FEATURE_DIM, max_patches=3, seed=seed)
        bag = MultimodalDataset(PATIENTS, {"wsi": source})[1].modalities["wsi"]
        assert isinstance(bag, torch.Tensor)
        return bag

    assert torch.equal(first_bag(7), first_bag(7))
    assert not torch.equal(first_bag(7), first_bag(8))


def test_a_custom_source_needs_only_two_methods(vectors) -> None:
    """The extension point: anything answering per patient is a modality."""

    class Constant(ModalitySource):
        def get(self, identifier: str, index: int) -> torch.Tensor:
            return torch.full((3,), float(identifier))

        def placeholder(self) -> torch.Tensor:
            return torch.zeros(3)

    dataset = MultimodalDataset(PATIENTS, {"clinical": VectorSource(vectors), "made_up": Constant()})

    assert dataset[2].modalities["made_up"].tolist() == [3.0, 3.0, 3.0]


def test_repr_names_the_modalities(vectors, bags) -> None:
    dataset = MultimodalDataset(
        PATIENTS, {"clinical": VectorSource(vectors), "wsi": FeatureBagSource(bags, feature_dim=FEATURE_DIM)}
    )

    assert "3 patients" in repr(dataset)
    assert "clinical=VectorSource" in repr(dataset)


def test_bags_of_several_slides_are_pooled_with_their_origin(tmp_path) -> None:
    """A patient's slides become one bag, each patch remembering which slide."""
    first, second = tmp_path / "a.h5", tmp_path / "b.h5"
    write_bag(first, num_patches=3, feature_dim=FEATURE_DIM)
    write_bag(second, num_patches=2, feature_dim=FEATURE_DIM)
    source = FeatureBagSource({"001": [first, second]}, feature_dim=FEATURE_DIM, with_coordinates=True)

    sample = MultimodalDataset(["001"], {"wsi": source})[0]

    assert sample.modalities["wsi"].shape == (5, FEATURE_DIM)
    assert sample.metadata["slide_index"].tolist() == [0, 0, 0, 1, 1]
    assert sample.metadata["slide_ids"] == ("a", "b")
    assert np.asarray(sample.metadata["coords"]).shape == (5, 2)
