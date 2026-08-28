"""Clinical survival labels for time-to-event modelling.

Labels follow one contract throughout ``kalecancer``: ``duration`` is the time to
event or censoring, and ``event`` is ``1`` when the event was observed and ``0`` when
the patient was censored. Source files record outcomes as free text, so the caller supplies an
:class:`EndpointSpec` naming the columns and the values that count as an event.
Those definitions belong to a dataset, so they live with the experiment rather
than here.
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
    """How to derive ``(duration, event)`` for one survival endpoint.

    Attributes:
        name: Identifier used when reporting the cohort, e.g. ``"OS"``.
    """

    name: str
    time_field: str
    status_field: str
    event_values: frozenset[str]
    #: Status values that cannot be classified for this endpoint. Matching records
    #: are excluded rather than silently treated as censored.
    unknown_values: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Status text is compared case-insensitively, so the values are folded once
        # here: a spec written with "Deceased" would otherwise match nothing and
        # silently mark every patient censored.
        object.__setattr__(self, "event_values", frozenset(value.strip().lower() for value in self.event_values))
        object.__setattr__(self, "unknown_values", frozenset(value.strip().lower() for value in self.unknown_values))


def endpoint_from_config(cfg) -> EndpointSpec:
    """Build an endpoint from a ``SURVIVAL`` configuration section.

    Keeps the column names that identify a dataset out of the library: they arrive
    as configuration, alongside the paths.

    Raises:
        ClinicalDataError: If the configuration names no columns. The library carries
            no default for them, and an endpoint built from empty names matches
            nothing and would mark every patient censored.
    """
    missing = [
        name
        for name, value in (
            ("SURVIVAL.TIME_FIELD", cfg.SURVIVAL.TIME_FIELD),
            ("SURVIVAL.STATUS_FIELD", cfg.SURVIVAL.STATUS_FIELD),
            ("SURVIVAL.EVENT_VALUES", cfg.SURVIVAL.EVENT_VALUES),
        )
        if not value
    ]
    if missing:
        raise ClinicalDataError(
            f"the survival endpoint is unset: {missing}. These name the dataset's own columns, "
            "so the experiment configuration must set them."
        )
    return EndpointSpec(
        name=cfg.SURVIVAL.ENDPOINT,
        time_field=cfg.SURVIVAL.TIME_FIELD,
        status_field=cfg.SURVIVAL.STATUS_FIELD,
        event_values=frozenset(value.lower() for value in cfg.SURVIVAL.EVENT_VALUES),
        unknown_values=frozenset(value.lower() for value in cfg.SURVIVAL.UNKNOWN_VALUES),
    )


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
    endpoint: EndpointSpec,
    patient_id_field: str = "patient_id",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Convert clinical records into a table of validated survival labels.

    Args:
        records: Raw patient objects, e.g. from :func:`load_clinical_records`.
        endpoint: How this dataset's columns define the endpoint.
        patient_id_field: Field holding the patient identifier. Read as a string so
            zero padding matches the identifiers used elsewhere.

    Returns:
        A table indexed by position with ``patient_id``, ``duration`` and ``event``
        columns, and the identifiers excluded per reason.

    Raises:
        ClinicalDataError: If an identifier is missing, or a patient appears more
            than once.
    """
    spec = endpoint

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
