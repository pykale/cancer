"""Tests for HDF5 patch-bag discovery, parsing, and validation."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from kalecancer.loaddata import (
    InvalidFeatureFileError,
    SlideIdentifierError,
    inspect_feature_bag,
    parse_patient_id,
    read_feature_bag,
    slide_table,
)
from tests.conftest import FEATURE_DIM, write_bag


@pytest.mark.parametrize(
    ("slide_id", "expected"),
    [
        ("PrimaryTumor_HE_001", "001"),
        ("PrimaryTumor_HE_036_a", "036"),
        ("LymphNode_HE_763", "763"),
    ],
)
def test_parse_patient_id_handles_multi_slide_suffix(slide_id: str, expected: str) -> None:
    assert parse_patient_id(slide_id) == expected


def test_parse_patient_id_rejects_unparseable_name() -> None:
    with pytest.raises(SlideIdentifierError, match="cannot parse a patient id"):
        parse_patient_id("slide_without_number")


def test_slide_table_walks_nested_directories(feature_root: Path) -> None:
    slides = slide_table(feature_root)

    assert not slides.attrs["unparsed"]
    assert list(slides.columns) == ["patient_id", "slide_id", "path"]
    assert list(slides["slide_id"]) == [
        "PrimaryTumor_HE_001",
        "PrimaryTumor_HE_002",
        "PrimaryTumor_HE_003",
        "PrimaryTumor_HE_003_a",
    ]
    assert list(slides["patient_id"]) == ["001", "002", "003", "003"]


def test_slide_table_reports_unparseable_files(feature_root: Path) -> None:
    write_bag(feature_root / "Larynx" / "h5_files" / "no_identifier.h5")

    slides = slide_table(feature_root)

    assert len(slides) == 4
    assert [Path(entry["path"]).name for entry in slides.attrs["unparsed"]] == ["no_identifier.h5"]


def test_slide_table_is_empty_for_a_directory_without_features(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    slides = slide_table(empty)

    assert slides.empty
    assert list(slides.columns) == ["patient_id", "slide_id", "path"]


def test_slide_table_requires_existing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        slide_table(tmp_path / "absent")


def test_read_feature_bag_returns_aligned_features_and_coords(feature_root: Path) -> None:
    features, coords = read_feature_bag(feature_root / "Larynx" / "h5_files" / "PrimaryTumor_HE_001.h5")

    assert features.shape == (10, FEATURE_DIM)
    assert coords.shape == (10, 2)
    assert features.dtype == np.float32


def test_read_feature_bag_subsets_features_and_coords_together(feature_root: Path) -> None:
    path = feature_root / "Larynx" / "h5_files" / "PrimaryTumor_HE_001.h5"
    full_features, full_coords = read_feature_bag(path)
    indices = np.array([7, 2, 5])

    features, coords = read_feature_bag(path, indices=indices)

    assert features.shape == (3, FEATURE_DIM)
    # Selection is sorted internally; rows must stay paired with their coordinates.
    for row, index in enumerate(sorted(indices)):
        assert np.array_equal(features[row], full_features[index])
        assert np.array_equal(coords[row], full_coords[index])


def test_read_feature_bag_rejects_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "no_coords.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=np.zeros((3, FEATURE_DIM), dtype=np.float32))

    with pytest.raises(InvalidFeatureFileError, match="missing required key"):
        read_feature_bag(path)


def test_read_feature_bag_rejects_empty_features(tmp_path: Path) -> None:
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=np.zeros((0, FEATURE_DIM), dtype=np.float32))
        handle.create_dataset("coords", data=np.zeros((0, 2), dtype=np.int32))

    with pytest.raises(InvalidFeatureFileError, match="is empty"):
        read_feature_bag(path)


def test_read_feature_bag_rejects_coord_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "mismatch.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=np.zeros((5, FEATURE_DIM), dtype=np.float32))
        handle.create_dataset("coords", data=np.zeros((3, 2), dtype=np.int32))

    with pytest.raises(InvalidFeatureFileError, match="patch alignment is required"):
        read_feature_bag(path)


def test_read_feature_bag_rejects_unexpected_feature_dim(feature_root: Path) -> None:
    path = feature_root / "Larynx" / "h5_files" / "PrimaryTumor_HE_001.h5"

    with pytest.raises(InvalidFeatureFileError, match="expected feature dimension 1024"):
        read_feature_bag(path, expected_dim=1024)


def test_inspect_feature_bag_returns_shape_without_reading_data(feature_root: Path) -> None:
    assert inspect_feature_bag(feature_root / "Larynx" / "h5_files" / "PrimaryTumor_HE_002.h5") == (15, FEATURE_DIM)


def test_coordinates_must_be_two_dimensional(tmp_path: Path) -> None:
    path = tmp_path / "PrimaryTumor_HE_009.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=np.zeros((4, FEATURE_DIM), dtype=np.float32))
        handle.create_dataset("coords", data=np.zeros(4, dtype=np.int32))

    with pytest.raises(InvalidFeatureFileError, match="coords"):
        inspect_feature_bag(path)


def test_patch_indices_outside_the_bag_are_reported_with_the_file(tmp_path: Path) -> None:
    path = write_bag(tmp_path / "PrimaryTumor_HE_010.h5", num_patches=4)

    with pytest.raises(InvalidFeatureFileError, match=r"\[0, 4\)"):
        read_feature_bag(path, indices=np.array([0, 99]))


def test_negative_patch_indices_are_refused(tmp_path: Path) -> None:
    path = write_bag(tmp_path / "PrimaryTumor_HE_011.h5", num_patches=4)

    with pytest.raises(InvalidFeatureFileError):
        read_feature_bag(path, indices=np.array([-1, 2]))
