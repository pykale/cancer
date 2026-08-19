"""Tests for joining slide features to clinical survival labels."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from kalecancer.loaddata import COHORT_COLUMNS, build_cohort
from tests.conftest import FEATURE_DIM, write_bag


def test_cohort_has_one_row_per_matched_slide(feature_root: Path, clinical_path: Path) -> None:
    cohort = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)

    assert list(cohort.columns) == COHORT_COLUMNS
    assert len(cohort) == 4
    assert cohort["patient_id"].nunique() == 3


def test_a_patients_slides_share_one_label(feature_root: Path, clinical_path: Path) -> None:
    cohort = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)
    multi_slide = cohort[cohort["patient_id"] == "003"]

    assert list(multi_slide["slide_id"]) == ["PrimaryTumor_HE_003", "PrimaryTumor_HE_003_a"]
    assert multi_slide["duration"].nunique() == 1
    assert set(multi_slide["event"]) == {1}


def test_rows_are_sorted_by_identifier(feature_root: Path, clinical_path: Path) -> None:
    cohort = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)

    assert list(cohort["slide_id"]) == sorted(cohort["slide_id"])


def test_labels_are_carried_onto_each_slide(feature_root: Path, clinical_path: Path) -> None:
    cohort = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)
    labels = cohort.drop_duplicates("patient_id").set_index("patient_id")

    assert labels.loc["001", "duration"] == 500.0
    assert labels.loc["001", "event"] == 1
    assert labels.loc["002", "event"] == 0


def test_patients_without_features_are_excluded(feature_root: Path, clinical_path: Path) -> None:
    cohort = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)

    assert "004" not in set(cohort["patient_id"])


def test_slides_without_a_label_are_excluded(feature_root: Path, clinical_path: Path) -> None:
    write_bag(feature_root / "Larynx" / "h5_files" / "PrimaryTumor_HE_999.h5")

    cohort = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)

    assert "999" not in set(cohort["patient_id"])


def test_invalid_feature_files_are_recorded_not_dropped_silently(feature_root: Path, clinical_path: Path) -> None:
    broken = feature_root / "Larynx" / "h5_files" / "PrimaryTumor_HE_005.h5"
    with h5py.File(broken, "w") as handle:
        handle.create_dataset("features", data=np.zeros((4, FEATURE_DIM), dtype=np.float32))

    cohort = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)

    assert len(cohort.attrs["invalid_features"]) == 1
    assert "missing required key" in cohort.attrs["invalid_features"][0]["reason"]
    assert cohort["patient_id"].nunique() == 3


def test_feature_validation_can_be_skipped(feature_root: Path, clinical_path: Path) -> None:
    with h5py.File(feature_root / "Larynx" / "h5_files" / "PrimaryTumor_HE_005.h5", "w") as handle:
        handle.create_dataset("features", data=np.zeros((4, FEATURE_DIM), dtype=np.float32))

    cohort = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM, validate_features=False)

    assert cohort.attrs["invalid_features"] == []


def test_unparsed_filenames_are_recorded(feature_root: Path, clinical_path: Path) -> None:
    write_bag(feature_root / "Larynx" / "h5_files" / "no_identifier.h5")

    cohort = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)

    assert [Path(entry["path"]).name for entry in cohort.attrs["unparsed_files"]] == ["no_identifier.h5"]


def test_provenance_is_attached_for_reporting(feature_root: Path, clinical_path: Path) -> None:
    cohort = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)

    assert cohort.attrs["endpoint"] == "OS"
    assert cohort.attrs["num_clinical_patients"] == 4
    assert len(cohort.attrs["slides"]) == 4
