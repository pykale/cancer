"""Clinical survival labels for time-to-event modelling.

Labels follow one contract throughout ``kalecancer``: ``duration`` is the time to
event or censoring, and ``event`` is ``1`` when the event was observed and ``0`` when
the patient was censored. Source files record outcomes as free text, so an endpoint
declares which field to read and which values count as an event.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


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


ENDPOINTS: dict[str, EndpointSpec] = {
    # Overall survival: death from any cause.
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


SURVIVAL_COLUMNS = ["patient_id", "duration", "event"]
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


def survival_table(
    records: list[dict],
    endpoint: str = "OS",
    patient_id_field: str = "patient_id",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Convert clinical records into a table of validated survival labels.

    Args:
        records: Raw patient objects, e.g. from :func:`load_clinical_records`.
        endpoint: Key into :data:`ENDPOINTS`.
        patient_id_field: Field holding the patient identifier. Read as a string so
            zero padding matches the identifiers used elsewhere.

    Returns:
        A table indexed by position with ``patient_id``, ``duration`` and ``event``
        columns, and the identifiers excluded per reason.

    Raises:
        ClinicalDataError: If the endpoint is unknown, an identifier is missing, or a
            patient appears more than once.
    """
    if endpoint not in ENDPOINTS:
        raise ClinicalDataError(f"unknown endpoint {endpoint!r}; available: {sorted(ENDPOINTS)}")
    spec = ENDPOINTS[endpoint]

    if any(patient_id_field not in record for record in records):
        raise ClinicalDataError(f"every clinical record needs a {patient_id_field!r} field")
    if not records:
        return pd.DataFrame(columns=SURVIVAL_COLUMNS), {reason: [] for reason in EXCLUSION_REASONS}

    table = pd.DataFrame(records)
    table["patient_id"] = table[patient_id_field].astype(str)
    duplicates = table["patient_id"][table["patient_id"].duplicated()].unique()
    if len(duplicates):
        raise ClinicalDataError(f"duplicate patient id(s) in clinical records: {sorted(duplicates)}")

    def column(name: str) -> pd.Series:
        return table[name] if name in table.columns else pd.Series(index=table.index, dtype=object)

    duration = pd.to_numeric(column(spec.time_field), errors="coerce")
    status = column(spec.status_field).astype("string").str.strip().str.lower()

    reasons = {
        "missing_time": duration.isna(),
        "invalid_time": duration.notna() & (duration <= 0),
        "missing_status": status.isna(),
        "unknown_status": status.isin(spec.unknown_values),
    }
    excluded, dropped = {}, pd.Series(False, index=table.index)
    for reason in EXCLUSION_REASONS:
        mask = reasons[reason]
        mask = mask & ~dropped
        excluded[reason] = table.loc[mask, "patient_id"].tolist()
        dropped |= mask

    survival = pd.DataFrame(
        {
            "patient_id": table.loc[~dropped, "patient_id"],
            "duration": duration[~dropped].astype(float),
            "event": status[~dropped].isin(spec.event_values).astype(int),
        }
    ).reset_index(drop=True)
    return survival, excluded
