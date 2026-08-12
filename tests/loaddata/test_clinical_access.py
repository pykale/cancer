"""Tests for clinical survival labels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalecancer.loaddata import ClinicalDataError, build_survival_records, load_clinical_records


def test_overall_survival_normalises_event_indicator() -> None:
    records = [
        {"patient_id": "001", "days_to_last_information": 500, "survival_status": "deceased"},
        {"patient_id": "002", "days_to_last_information": 800, "survival_status": "living"},
    ]

    survival, excluded = build_survival_records(records, endpoint="OS")

    assert survival["001"].event == 1
    assert survival["002"].event == 0
    assert survival["001"].duration == 500.0
    assert not any(excluded.values())


def test_patient_ids_keep_zero_padding() -> None:
    survival, _ = build_survival_records(
        [{"patient_id": "007", "days_to_last_information": 100, "survival_status": "living"}]
    )

    assert list(survival) == ["007"]


def test_status_matching_ignores_case_and_padding() -> None:
    survival, _ = build_survival_records(
        [{"patient_id": "001", "days_to_last_information": 100, "survival_status": " Deceased "}]
    )

    assert survival["001"].event == 1


def test_disease_specific_survival_excludes_unknown_cause() -> None:
    records = [
        {"patient_id": "001", "days_to_last_information": 500, "survival_status_with_cause": "deceased tumor specific"},
        {
            "patient_id": "002",
            "days_to_last_information": 600,
            "survival_status_with_cause": "deceased not tumor specific",
        },
        {"patient_id": "003", "days_to_last_information": 700, "survival_status_with_cause": "deceased"},
    ]

    survival, excluded = build_survival_records(records, endpoint="DSS")

    assert survival["001"].event == 1
    # Died of another cause: censored for a disease-specific endpoint.
    assert survival["002"].event == 0
    # Cause unknown: cannot be classified, so excluded rather than assumed censored.
    assert "003" not in survival
    assert excluded["unknown_status"] == ["003"]


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        ({"patient_id": "001", "survival_status": "living"}, "missing_time"),
        ({"patient_id": "001", "days_to_last_information": -5, "survival_status": "living"}, "invalid_time"),
        ({"patient_id": "001", "days_to_last_information": 0, "survival_status": "living"}, "invalid_time"),
        ({"patient_id": "001", "days_to_last_information": 100}, "missing_status"),
    ],
)
def test_invalid_survival_records_are_excluded_with_a_reason(record: dict, reason: str) -> None:
    survival, excluded = build_survival_records([record])

    assert not survival
    assert excluded[reason] == ["001"]


def test_duplicate_patient_ids_are_rejected() -> None:
    records = [
        {"patient_id": "001", "days_to_last_information": 100, "survival_status": "living"},
        {"patient_id": "001", "days_to_last_information": 200, "survival_status": "deceased"},
    ]

    with pytest.raises(ClinicalDataError, match="duplicate patient id"):
        build_survival_records(records)


def test_record_without_an_identifier_is_rejected() -> None:
    with pytest.raises(ClinicalDataError, match="without 'patient_id'"):
        build_survival_records([{"days_to_last_information": 100, "survival_status": "living"}])


def test_unknown_endpoint_is_rejected() -> None:
    with pytest.raises(ClinicalDataError, match="unknown endpoint"):
        build_survival_records([], endpoint="PFS")


def test_load_clinical_records_reads_a_list_of_patients(clinical_path: Path) -> None:
    records = load_clinical_records(clinical_path)

    assert len(records) == 4
    assert records[0]["patient_id"] == "001"


def test_load_clinical_records_rejects_non_list_payload(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"patient_id": "001"}), encoding="utf-8")

    with pytest.raises(ClinicalDataError, match="expected a JSON list"):
        load_clinical_records(path)
