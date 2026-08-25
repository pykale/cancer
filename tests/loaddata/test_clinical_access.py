"""Tests for clinical survival labels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalecancer.loaddata import ClinicalDataError, load_clinical_records, survival_table
from kalecancer.loaddata.clinical_access import EndpointSpec
from tests.conftest import OS_ENDPOINT

DSS_ENDPOINT = EndpointSpec(
    name="DSS",
    time_field="days_to_last_information",
    status_field="survival_status_with_cause",
    event_values=frozenset({"deceased tumor specific"}),
    unknown_values=frozenset({"deceased"}),
)


def labels_by_patient(records: list[dict], **kwargs):
    table, excluded = survival_table(records, endpoint=kwargs.pop("endpoint", OS_ENDPOINT), **kwargs)
    return table.set_index("patient_id"), excluded


def test_overall_survival_normalises_event_indicator() -> None:
    labels, excluded = labels_by_patient(
        [
            {"patient_id": "001", "days_to_last_information": 500, "survival_status": "deceased"},
            {"patient_id": "002", "days_to_last_information": 800, "survival_status": "living"},
        ]
    )

    assert labels.loc["001", "event"] == 1
    assert labels.loc["002", "event"] == 0
    assert labels.loc["001", "duration"] == 500.0
    assert not any(excluded.values())


def test_table_exposes_the_label_contract() -> None:
    table, _ = survival_table(
        [{"patient_id": "001", "days_to_last_information": 10, "survival_status": "living"}], endpoint=OS_ENDPOINT
    )

    assert list(table.columns) == ["patient_id", "duration", "event"]


def test_patient_ids_keep_zero_padding() -> None:
    table, _ = survival_table(
        [{"patient_id": "007", "days_to_last_information": 100, "survival_status": "living"}], endpoint=OS_ENDPOINT
    )

    assert list(table["patient_id"]) == ["007"]


def test_numeric_patient_ids_are_read_as_strings() -> None:
    table, _ = survival_table(
        [{"patient_id": 7, "days_to_last_information": 100, "survival_status": "living"}], endpoint=OS_ENDPOINT
    )

    assert list(table["patient_id"]) == ["7"]


def test_status_matching_ignores_case_and_padding() -> None:
    labels, _ = labels_by_patient(
        [{"patient_id": "001", "days_to_last_information": 100, "survival_status": " Deceased "}]
    )

    assert labels.loc["001", "event"] == 1


def test_disease_specific_survival_excludes_unknown_cause() -> None:
    labels, excluded = labels_by_patient(
        [
            {
                "patient_id": "001",
                "days_to_last_information": 500,
                "survival_status_with_cause": "deceased tumor specific",
            },
            {
                "patient_id": "002",
                "days_to_last_information": 600,
                "survival_status_with_cause": "deceased not tumor specific",
            },
            {"patient_id": "003", "days_to_last_information": 700, "survival_status_with_cause": "deceased"},
        ],
        endpoint=DSS_ENDPOINT,
    )

    assert labels.loc["001", "event"] == 1
    # Died of another cause: censored for a disease-specific endpoint.
    assert labels.loc["002", "event"] == 0
    # Cause unknown: cannot be classified, so excluded rather than assumed censored.
    assert "003" not in labels.index
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
    table, excluded = survival_table([record], endpoint=OS_ENDPOINT)

    assert table.empty
    assert excluded[reason] == ["001"]


def test_each_excluded_patient_is_reported_once() -> None:
    table, excluded = survival_table([{"patient_id": "001"}], endpoint=OS_ENDPOINT)

    assert table.empty
    assert sum(len(ids) for ids in excluded.values()) == 1


def test_duplicate_patient_ids_are_rejected() -> None:
    records = [
        {"patient_id": "001", "days_to_last_information": 100, "survival_status": "living"},
        {"patient_id": "001", "days_to_last_information": 200, "survival_status": "deceased"},
    ]

    with pytest.raises(ClinicalDataError, match="duplicate patient id"):
        survival_table(records, endpoint=OS_ENDPOINT)


def test_record_without_an_identifier_is_rejected() -> None:
    with pytest.raises(ClinicalDataError, match="needs a 'patient_id' field"):
        survival_table([{"days_to_last_information": 100, "survival_status": "living"}], endpoint=OS_ENDPOINT)


def test_an_endpoint_is_built_from_configuration() -> None:
    """Column names identify a dataset, so they arrive as configuration."""
    from kalecancer.config import get_cfg_defaults
    from kalecancer.loaddata.clinical_access import endpoint_from_config

    cfg = get_cfg_defaults()
    cfg.SURVIVAL.ENDPOINT = "DSS"
    cfg.SURVIVAL.STATUS_FIELD = "survival_status_with_cause"
    cfg.SURVIVAL.EVENT_VALUES = ["Deceased Tumor Specific"]
    cfg.SURVIVAL.UNKNOWN_VALUES = ["deceased"]

    spec = endpoint_from_config(cfg)

    assert spec.name == "DSS"
    assert spec.status_field == "survival_status_with_cause"
    # Comparison is case-insensitive, so configured values are lowered once here.
    assert spec.event_values == frozenset({"deceased tumor specific"})
    assert spec.unknown_values == frozenset({"deceased"})


def test_load_clinical_records_reads_a_list_of_patients(clinical_path: Path) -> None:
    records = load_clinical_records(clinical_path)

    assert len(records) == 4
    assert records[0]["patient_id"] == "001"


def test_load_clinical_records_rejects_non_list_payload(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"patient_id": "001"}), encoding="utf-8")

    with pytest.raises(ClinicalDataError, match="expected a JSON list"):
        load_clinical_records(path)


def test_endpoint_values_are_matched_case_insensitively() -> None:
    spec = EndpointSpec(
        name="OS",
        time_field="days_to_last_information",
        status_field="survival_status",
        event_values=frozenset({"Deceased"}),
    )
    records = [{"patient_id": "001", "days_to_last_information": 10, "survival_status": "deceased"}]

    table, _ = survival_table(records, endpoint=spec)

    assert table["event"].tolist() == [1]
