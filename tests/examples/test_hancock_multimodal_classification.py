"""Tests for the HANCOCK outcome-classification example.

Network-free: the structured tables are built in memory rather than fetched.
"""

from __future__ import annotations

import pandas as pd
import pytest

from examples.hancock import official_split
from examples.hancock_multimodal_classification.cohorts import LABEL, labelled_cohort
from examples.hancock_multimodal_classification.config import get_cfg_defaults
from examples.hancock_multimodal_classification.tabular import STAGE_RANKS, hematology_parameters


@pytest.fixture
def cfg():
    return get_cfg_defaults()


@pytest.fixture
def reference_ranges() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["Hematology", "Hematology", "Hematology", "Routine", "Hematology"],
            "LOINC_name": ["Leukocytes", "Basophils %", "Hemoglobin", "Sodium", "Platelets"],
            "normal_female_max": [10.0, 2.0, 16.0, 145.0, None],
        }
    )


def test_percentage_differentials_and_other_groups_are_excluded(reference_ranges) -> None:
    """Percentages are determined by the absolute counts, and only blood counts qualify."""
    assert hematology_parameters(reference_ranges) == ["Hemoglobin", "Leukocytes"]


def test_stage_ranks_are_ordered_and_exclude_the_unassessed() -> None:
    """``TX``/``NX`` mean the stage was never assessed, which is not a mild stage."""
    assert "TX" not in STAGE_RANKS["pT_stage"]
    assert "NX" not in STAGE_RANKS["pN_stage"]
    assert list(STAGE_RANKS["pT_stage"].values()) == sorted(STAGE_RANKS["pT_stage"].values())


@pytest.fixture
def clinical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["001", "002", "003", "004", "005"],
            "survival_status": ["living", "deceased", "deceased", "living", "deceased"],
            "survival_status_with_cause": [
                "living",
                "deceased tumor specific",
                "deceased not tumor specific",
                "living",
                "deceased",
            ],
            "recurrence": ["no", "yes", "no", "no", "yes"],
            "days_to_recurrence": [None, 300.0, None, None, 2000.0],
            "days_to_last_information": [4000, 300, 900, 400, 2000],
        }
    )


def test_non_tumour_deaths_are_excluded_not_counted_as_survivors(clinical, cfg) -> None:
    cohort = labelled_cohort(clinical, cfg, "survival_status")

    assert "003" not in set(cohort["patient_id"])
    assert dict(zip(cohort["patient_id"], cohort[LABEL])) == {"001": 0, "002": 1, "004": 0, "005": 1}


def test_recurrence_follows_the_published_code_not_the_paper_prose(clinical, cfg) -> None:
    """Patient 004 is the discrepancy, and it is deliberate.

    They are alive and recurrence-free but have only 400 days of follow-up. The
    paper's prose excludes such patients for insufficient follow-up; the published
    code keeps them, via a trailing ``or survival_status == "living"`` disjunct that
    roughly doubles the negative class. We follow the code, because that is what
    produced the published numbers. Do not "fix" this to match the prose without
    changing every recurrence figure with it.
    """
    cohort = labelled_cohort(clinical, cfg, "recurrence")

    assert "004" in set(cohort["patient_id"])
    assert dict(zip(cohort["patient_id"], cohort[LABEL]))["004"] == 0


def test_a_recurrence_after_the_horizon_is_dropped(clinical, cfg) -> None:
    """Patient 005 recurred at 2000 days, which is not a three-year recurrence."""
    cohort = labelled_cohort(clinical, cfg, "recurrence")

    assert "005" not in set(cohort["patient_id"])


def test_an_unknown_target_is_refused(clinical, cfg) -> None:
    with pytest.raises(KeyError, match="unknown target"):
        labelled_cohort(clinical, cfg, "metastasis")


def test_official_split_keeps_identifiers_padded(tmp_path) -> None:
    """Padding lost here makes every downstream join silently match nothing."""
    path = tmp_path / "dataset_split_in.json"
    path.write_text(
        pd.DataFrame(
            {"patient_id": ["001", "002", "010"], "dataset": ["training", "test", "training"]}
        ).to_json(orient="records"),
        encoding="utf-8",
    )

    split = official_split(path)

    assert split["training"] == ["001", "010"]
    assert split["test"] == ["002"]
