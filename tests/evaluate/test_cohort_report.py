"""Tests for cohort and split reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from kalecancer.evaluate import cohort_summary, split_summary
from tests.conftest import FEATURE_DIM, OS_ENDPOINT, write_bag


def test_summary_counts_groups_and_slides(cohort) -> None:
    summary = cohort_summary(cohort)

    assert summary["num_matched_groups"] == 3
    assert summary["num_matched_slides"] == 4
    assert summary["num_events"] == 2
    assert summary["num_censored"] == 1


def test_summary_reports_both_sides_of_the_mismatch(cohort, feature_root: Path, clinical_path: Path) -> None:
    from kalecancer.loaddata import build_cohort

    write_bag(feature_root / "Larynx" / "h5_files" / "PrimaryTumor_HE_999.h5")
    summary = cohort_summary(build_cohort(feature_root, clinical_path, endpoint=OS_ENDPOINT, expected_dim=FEATURE_DIM))

    assert summary["labelled_without_features"] == ["004"]
    assert summary["features_without_label"] == ["999"]


def test_summary_lists_groups_with_several_samples(cohort) -> None:
    assert cohort_summary(cohort)["groups_with_multiple_slides"] == {"003": 2}


def test_summary_is_json_serialisable(cohort) -> None:
    assert json.loads(json.dumps(cohort_summary(cohort)))["endpoint"] == "OS"


def test_summary_handles_an_empty_cohort() -> None:
    empty = pd.DataFrame(columns=["patient_id", "slide_id", "path", "duration", "event"])

    summary = cohort_summary(empty)

    assert summary["num_matched_groups"] == 0
    assert summary["num_events"] == 0


def test_split_summary_describes_each_split(cohort) -> None:
    split = {"train": np.array([0, 1]), "val": np.array([2]), "test": np.array([3])}

    summary = split_summary(cohort, split)

    assert set(summary) == {"train", "val", "test"}
    assert summary["train"]["num_slides"] == 2
    assert all(0.0 <= entry["event_rate"] <= 1.0 for entry in summary.values())


def test_split_summary_counts_groups_not_rows(cohort) -> None:
    """Patient 003 contributes two slides but is one group."""
    indices = np.flatnonzero(cohort["patient_id"] == "003")

    summary = split_summary(cohort, {"test": indices})

    assert summary["test"]["num_slides"] == 2
    assert summary["test"]["num_groups"] == 1
