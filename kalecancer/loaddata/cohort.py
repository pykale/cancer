"""Patient-level matching of slide features to clinical survival labels.

Matching is deterministic and happens entirely at patient level: a patient's slides
are always kept together, which makes slide-level leakage between splits impossible
downstream. Every excluded patient or file is recorded in :class:`CohortSummary`
rather than dropped silently.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kalecancer.loaddata.clinical_access import (
    SurvivalRecord,
    build_survival_records,
    load_clinical_records,
)
from kalecancer.loaddata.wsi_feature_access import (
    DEFAULT_SLIDE_PATTERN,
    InvalidFeatureFileError,
    SlideRecord,
    discover_slides,
    inspect_feature_bag,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatientBag:
    """All primary-tumour slides for one patient, with their survival label."""

    patient_id: str
    slides: tuple[SlideRecord, ...]
    duration: float
    event: int


@dataclass
class CohortSummary:
    """Provenance of the matched cohort, suitable for JSON export."""

    endpoint: str
    num_clinical_patients: int = 0
    num_wsi_patients: int = 0
    num_wsi_slides: int = 0
    num_matched_patients: int = 0
    num_matched_slides: int = 0
    num_events: int = 0
    num_censored: int = 0
    #: Clinical patients with no usable slide features.
    patients_without_wsi: list[str] = field(default_factory=list)
    #: Slide patients with no usable survival label, either absent from the clinical
    #: file or excluded for this endpoint (see ``clinical_exclusions``).
    unmatched_wsi_patients: list[str] = field(default_factory=list)
    clinical_exclusions: dict[str, list[str]] = field(default_factory=dict)
    invalid_feature_files: list[dict[str, str]] = field(default_factory=list)
    unparsed_feature_files: list[dict[str, str]] = field(default_factory=list)
    patients_with_multiple_slides: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def log(self) -> None:
        logger.info(
            "cohort (%s): %d matched patients / %d slides from %d clinical and %d WSI patients "
            "(%d events, %d censored)",
            self.endpoint,
            self.num_matched_patients,
            self.num_matched_slides,
            self.num_clinical_patients,
            self.num_wsi_patients,
            self.num_events,
            self.num_censored,
        )
        for label, values in (
            ("clinical patients without WSI features", self.patients_without_wsi),
            ("WSI patients without a usable survival label", self.unmatched_wsi_patients),
            ("invalid feature files", self.invalid_feature_files),
            ("unparsed feature files", self.unparsed_feature_files),
        ):
            if values:
                logger.warning("excluded %d %s", len(values), label)
        if self.clinical_exclusions:
            logger.warning(
                "clinical records excluded for the %s endpoint: %s",
                self.endpoint,
                {reason: len(ids) for reason, ids in self.clinical_exclusions.items()},
            )


def build_cohort(
    feature_root: str | Path,
    clinical_path: str | Path,
    endpoint: str = "OS",
    expected_dim: int | None = None,
    slide_pattern: re.Pattern[str] = DEFAULT_SLIDE_PATTERN,
    validate_features: bool = True,
) -> tuple[list[PatientBag], CohortSummary]:
    """Match slide feature files to clinical survival labels at patient level.

    Args:
        feature_root: Directory of HDF5 slide features, searched recursively.
        clinical_path: JSON file of clinical records.
        endpoint: Survival endpoint, see :data:`~kalecancer.loaddata.clinical_access.ENDPOINTS`.
        expected_dim: Feature dimension every slide must provide, if it should be checked.
        slide_pattern: Regular expression exposing a ``patient_id`` named group.
        validate_features: Read each file's header to reject invalid bags up front.

    Returns:
        Patient bags sorted by patient id, and the matching summary.
    """
    slides, unparsed = discover_slides(feature_root, pattern=slide_pattern)
    summary = CohortSummary(endpoint=endpoint)
    summary.unparsed_feature_files = [{"path": str(path), "reason": reason} for path, reason in unparsed]

    usable: list[SlideRecord] = []
    for slide in slides:
        if validate_features:
            try:
                inspect_feature_bag(slide.path, expected_dim=expected_dim)
            except InvalidFeatureFileError as error:
                summary.invalid_feature_files.append({"path": str(slide.path), "reason": str(error)})
                continue
        usable.append(slide)

    slides_by_patient: dict[str, list[SlideRecord]] = {}
    for slide in usable:
        slides_by_patient.setdefault(slide.patient_id, []).append(slide)

    clinical = load_clinical_records(clinical_path)
    survival, clinical_exclusions = build_survival_records(clinical, endpoint=endpoint)

    summary.num_clinical_patients = len(clinical)
    summary.num_wsi_patients = len(slides_by_patient)
    summary.num_wsi_slides = len(usable)
    summary.clinical_exclusions = {reason: ids for reason, ids in clinical_exclusions.items() if ids}

    bags = _match(slides_by_patient, survival, summary)
    summary.num_matched_patients = len(bags)
    summary.num_matched_slides = sum(len(bag.slides) for bag in bags)
    summary.num_events = sum(bag.event for bag in bags)
    summary.num_censored = summary.num_matched_patients - summary.num_events
    summary.patients_with_multiple_slides = {bag.patient_id: len(bag.slides) for bag in bags if len(bag.slides) > 1}
    return bags, summary


def _match(
    slides_by_patient: dict[str, list[SlideRecord]],
    survival: dict[str, SurvivalRecord],
    summary: CohortSummary,
) -> list[PatientBag]:
    """Intersect slide and survival patients, recording both sides of the mismatch."""
    summary.unmatched_wsi_patients = sorted(set(slides_by_patient) - set(survival))
    summary.patients_without_wsi = sorted(set(survival) - set(slides_by_patient))

    bags = []
    for patient_id in sorted(set(slides_by_patient) & set(survival)):
        label = survival[patient_id]
        bags.append(
            PatientBag(
                patient_id=patient_id,
                slides=tuple(sorted(slides_by_patient[patient_id], key=lambda slide: slide.slide_id)),
                duration=label.duration,
                event=label.event,
            )
        )
    return bags
