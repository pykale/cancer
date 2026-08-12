"""Clinical survival labels for time-to-event modelling.

Records follow the label contract used across ``kalecancer``:

``duration``
    Time from baseline to event or censoring, in the unit of the source field (days
    for HANCOCK).
``event``
    ``1`` if the event was observed, ``0`` if the patient was censored.

Source files store outcomes as free text (e.g. ``"deceased"``), so each endpoint
declares which field to read and which values count as an event.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class ClinicalDataError(ValueError):
    """Raised when clinical survival records violate the label contract."""


@dataclass(frozen=True)
class EndpointSpec:
    """How to derive ``(duration, event)`` for one survival endpoint."""

    time_field: str
    status_field: str
    event_values: frozenset[str]
    #: Status values that cannot be classified for this endpoint. Matching records
    #: are excluded rather than silently treated as censored.
    unknown_values: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class SurvivalRecord:
    """One patient's time-to-event label."""

    patient_id: str
    duration: float
    event: int


ENDPOINTS: dict[str, EndpointSpec] = {
    # Overall survival: death from any cause. Primary target - death is unambiguous
    # and more consistently recorded than recurrence.
    "OS": EndpointSpec(
        time_field="days_to_last_information",
        status_field="survival_status",
        event_values=frozenset({"deceased"}),
    ),
    # Disease-specific survival: death attributed to the tumour. Records stating only
    # "deceased" carry no cause and cannot be classified.
    "DSS": EndpointSpec(
        time_field="days_to_last_information",
        status_field="survival_status_with_cause",
        event_values=frozenset({"deceased tumor specific"}),
        unknown_values=frozenset({"deceased"}),
    ),
}

EXCLUSION_REASONS = ("missing_time", "invalid_time", "missing_status", "unknown_status")


def load_clinical_records(path: str | Path) -> list[dict]:
    """Load clinical records from a JSON file holding a list of patient objects.

    Raises:
        ClinicalDataError: If the file does not contain a list of objects.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list) or not all(isinstance(record, dict) for record in payload):
        raise ClinicalDataError(f"{path}: expected a JSON list of patient objects")
    return payload


def build_survival_records(
    records: list[dict],
    endpoint: str = "OS",
    patient_id_field: str = "patient_id",
) -> tuple[dict[str, SurvivalRecord], dict[str, list[str]]]:
    """Convert raw clinical records into validated survival labels.

    Args:
        records: Raw patient objects, e.g. from :func:`load_clinical_records`.
        endpoint: Key into :data:`ENDPOINTS`.
        patient_id_field: Field holding the patient identifier. Values are read as
            strings so zero padding matches the slide filenames.

    Returns:
        Survival records keyed by patient id, and the patient ids excluded per reason
        in :data:`EXCLUSION_REASONS`.

    Raises:
        ClinicalDataError: If the endpoint is unknown, an identifier is missing, or a
            patient id appears more than once.
    """
    if endpoint not in ENDPOINTS:
        raise ClinicalDataError(f"unknown endpoint {endpoint!r}; available: {sorted(ENDPOINTS)}")
    spec = ENDPOINTS[endpoint]

    survival: dict[str, SurvivalRecord] = {}
    excluded: dict[str, list[str]] = {reason: [] for reason in EXCLUSION_REASONS}
    seen: set[str] = set()

    for record in records:
        raw_id = record.get(patient_id_field)
        if raw_id is None:
            raise ClinicalDataError(f"clinical record without {patient_id_field!r}: {record}")

        patient_id = str(raw_id)
        if patient_id in seen:
            raise ClinicalDataError(f"duplicate patient id {patient_id!r} in clinical records")
        seen.add(patient_id)

        duration = record.get(spec.time_field)
        status = record.get(spec.status_field)

        if duration is None:
            excluded["missing_time"].append(patient_id)
            continue
        if float(duration) <= 0:
            excluded["invalid_time"].append(patient_id)
            continue
        if status is None:
            excluded["missing_status"].append(patient_id)
            continue

        status = str(status).strip().lower()
        if status in spec.unknown_values:
            excluded["unknown_status"].append(patient_id)
            continue

        survival[patient_id] = SurvivalRecord(
            patient_id=patient_id,
            duration=float(duration),
            event=int(status in spec.event_values),
        )

    return survival, excluded
