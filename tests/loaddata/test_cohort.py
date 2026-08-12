"""Tests for patient-level matching of slides to survival labels."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from kalecancer.loaddata import build_cohort
from tests.conftest import FEATURE_DIM, write_bag


def test_matches_patients_and_pools_their_slides(feature_root: Path, clinical_path: Path) -> None:
    bags, summary = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)

    assert [bag.patient_id for bag in bags] == ["001", "002", "003"]
    assert summary.num_matched_patients == 3
    assert summary.num_matched_slides == 4
    # Patient 003 contributes two slides to a single bag.
    assert summary.patients_with_multiple_slides == {"003": 2}
    assert summary.num_events == 2
    assert summary.num_censored == 1


def test_slides_of_a_patient_are_ordered_deterministically(feature_root: Path, clinical_path: Path) -> None:
    bags, _ = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)
    multi_slide = next(bag for bag in bags if bag.patient_id == "003")

    assert [slide.slide_id for slide in multi_slide.slides] == [
        "PrimaryTumor_HE_003",
        "PrimaryTumor_HE_003_a",
    ]


def test_survival_labels_are_attached_to_the_bag(feature_root: Path, clinical_path: Path) -> None:
    bags, _ = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)

    assert {bag.patient_id: (bag.duration, bag.event) for bag in bags} == {
        "001": (500.0, 1),
        "002": (1200.0, 0),
        "003": (900.0, 1),
    }


def test_records_both_sides_of_the_mismatch(feature_root: Path, clinical_path: Path) -> None:
    _, summary = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)

    assert summary.patients_without_wsi == ["004"]
    assert summary.unmatched_wsi_patients == []
    assert summary.num_clinical_patients == 4
    assert summary.num_wsi_patients == 3


def test_slides_without_a_clinical_record_are_reported(feature_root: Path, clinical_path: Path) -> None:
    write_bag(feature_root / "Larynx" / "h5_files" / "PrimaryTumor_HE_999.h5")

    bags, summary = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)

    assert summary.unmatched_wsi_patients == ["999"]
    assert "999" not in {bag.patient_id for bag in bags}


def test_invalid_feature_files_are_reported_not_dropped_silently(feature_root: Path, clinical_path: Path) -> None:
    broken = feature_root / "Larynx" / "h5_files" / "PrimaryTumor_HE_005.h5"
    with h5py.File(broken, "w") as handle:
        handle.create_dataset("features", data=np.zeros((4, FEATURE_DIM), dtype=np.float32))

    bags, summary = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)

    assert len(summary.invalid_feature_files) == 1
    assert "missing required key" in summary.invalid_feature_files[0]["reason"]
    # The invalid file is excluded but the rest of the cohort still loads.
    assert [bag.patient_id for bag in bags] == ["001", "002", "003"]


def test_feature_validation_can_be_skipped(feature_root: Path, clinical_path: Path) -> None:
    broken = feature_root / "Larynx" / "h5_files" / "PrimaryTumor_HE_005.h5"
    with h5py.File(broken, "w") as handle:
        handle.create_dataset("features", data=np.zeros((4, FEATURE_DIM), dtype=np.float32))

    _, summary = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM, validate_features=False)

    assert summary.invalid_feature_files == []


def test_summary_is_json_serialisable(feature_root: Path, clinical_path: Path) -> None:
    import json

    _, summary = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)

    assert json.loads(json.dumps(summary.as_dict()))["endpoint"] == "OS"
