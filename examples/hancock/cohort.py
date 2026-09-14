"""Joining HANCOCK slide feature files to its clinical survival labels.

Dataset-specific by nature: it knows how a HANCOCK slide filename encodes its
patient, and which clinical columns define an endpoint. The library supplies the
pieces -- :func:`~kalecancer.loaddata.slide_table` to discover files,
:func:`~kalecancer.loaddata.inspect_feature_bag` to validate them -- and this
composes them for one dataset.

Produces one row per slide, which is the representation the generic splitters and
datasets consume:

======================  ==========================================================
``patient_id``          Identifier shared by a patient's slides; the grouping key
``slide_id``            Identifier of the individual slide
``path``                Feature file for the slide
``duration``, ``event`` Survival label, repeated for each of a patient's slides
======================  ==========================================================

Slides without a survival label, and labelled patients without usable features, are
dropped from the table; :func:`~examples.hancock.cohort.cohort_summary`
reports what was excluded.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from examples.hancock.clinical import EndpointSpec, load_clinical_records, survival_table
from examples.hancock.dataset import SLIDE_PATTERN
from kalecancer.loaddata.wsi_access import InvalidFeatureFileError, inspect_feature_bag, slide_table

logger = logging.getLogger(__name__)

COHORT_COLUMNS = ["patient_id", "slide_id", "path", "duration", "event"]


def build_cohort(
    feature_root: str | Path,
    clinical_path: str | Path,
    endpoint: EndpointSpec,
    expected_dim: int | None = None,
    slide_pattern: re.Pattern[str] = SLIDE_PATTERN,
    validate_features: bool = True,
) -> pd.DataFrame:
    """Build the cohort table for a set of slides and clinical records.

    Args:
        feature_root: Directory of slide feature files, searched recursively.
        clinical_path: JSON file of clinical records.
        endpoint: How this dataset's columns define the survival endpoint.
        expected_dim: Feature dimension every slide must provide, if it should be checked.
        slide_pattern: Regular expression exposing a ``patient_id`` named group.
        validate_features: Read each file's header and drop unreadable slides.

    Returns:
        One row per usable slide, sorted by ``(patient_id, slide_id)``. Excluded
        slides and patients are recorded in the ``attrs`` mapping under
        ``"invalid_features"``, ``"unparsed_files"`` and ``"clinical_exclusions"``.
    """
    slides = slide_table(feature_root, pattern=slide_pattern)

    invalid: list[dict[str, str]] = []
    if validate_features and not slides.empty:
        keep = []
        for path in slides["path"]:
            try:
                inspect_feature_bag(path, expected_dim=expected_dim)
            except InvalidFeatureFileError as error:
                invalid.append({"path": str(path), "reason": str(error)})
                keep.append(False)
            else:
                keep.append(True)
        slides = slides[keep]

    records = load_clinical_records(clinical_path)
    survival, exclusions = survival_table(records, endpoint=endpoint)

    cohort = slides.merge(survival, on="patient_id", how="inner")
    cohort = cohort.sort_values(["patient_id", "slide_id"], ignore_index=True)[COHORT_COLUMNS]
    cohort.attrs.update(
        endpoint=endpoint.name,
        num_clinical_patients=len(records),
        slides=slides,
        survival=survival,
        invalid_features=invalid,
        unparsed_files=slides.attrs.get("unparsed", []),
        clinical_exclusions={reason: ids for reason, ids in exclusions.items() if ids},
    )
    return cohort


# --------------------------------------------------------------------------- #
# Describing what the join produced
# --------------------------------------------------------------------------- #
#
# These read the ``attrs`` that build_cohort above records, so they only make sense
# for a table it produced -- which is why they live here rather than in
# ``kalecancer.evaluate``.


def cohort_summary(cohort: pd.DataFrame, group_key: str = "patient_id") -> dict:
    """Summarise a cohort table and the exclusions recorded while building it.

    Args:
        cohort: Table from :func:`build_cohort`.
        group_key: Column identifying the unit the labels belong to.

    Returns:
        Counts and excluded identifiers, ready for JSON export.
    """
    attrs = cohort.attrs
    slides = attrs.get("slides", pd.DataFrame(columns=[group_key]))
    survival = attrs.get("survival", pd.DataFrame(columns=[group_key]))

    labelled = set(survival[group_key]) if len(survival) else set()
    discovered = set(slides[group_key]) if len(slides) else set()
    matched = set(cohort[group_key]) if len(cohort) else set()
    slides_per_group = cohort.groupby(group_key).size() if len(cohort) else pd.Series(dtype=int)
    labels = cohort.drop_duplicates(group_key) if len(cohort) else cohort

    return {
        "endpoint": attrs.get("endpoint"),
        "num_clinical_patients": attrs.get("num_clinical_patients", len(labelled)),
        "num_discovered_groups": len(discovered),
        "num_discovered_slides": len(slides),
        "num_matched_groups": len(matched),
        "num_matched_slides": len(cohort),
        "num_events": int(labels["event"].sum()) if len(labels) else 0,
        "num_censored": int((1 - labels["event"]).sum()) if len(labels) else 0,
        "labelled_without_features": sorted(labelled - discovered),
        "features_without_label": sorted(discovered - matched),
        "groups_with_multiple_slides": slides_per_group[slides_per_group > 1].to_dict(),
        "clinical_exclusions": attrs.get("clinical_exclusions", {}),
        "invalid_features": attrs.get("invalid_features", []),
        "unparsed_files": attrs.get("unparsed_files", []),
    }


def log_cohort_summary(summary: dict) -> None:
    """Log the headline counts and every exclusion category that is non-empty."""
    logger.info(
        "cohort (%s): %d matched groups / %d slides from %d clinical and %d discovered groups (%d events, %d censored)",
        summary["endpoint"],
        summary["num_matched_groups"],
        summary["num_matched_slides"],
        summary["num_clinical_patients"],
        summary["num_discovered_groups"],
        summary["num_events"],
        summary["num_censored"],
    )
    for key, label in (
        ("labelled_without_features", "labelled groups without usable features"),
        ("features_without_label", "discovered groups without a usable label"),
        ("invalid_features", "invalid feature files"),
        ("unparsed_files", "unparsed feature files"),
    ):
        if summary[key]:
            logger.warning("excluded %d %s", len(summary[key]), label)
    if summary["clinical_exclusions"]:
        logger.warning(
            "clinical records excluded for the %s endpoint: %s",
            summary["endpoint"],
            {reason: len(ids) for reason, ids in summary["clinical_exclusions"].items()},
        )


def split_summary(cohort: pd.DataFrame, splits: dict, group_key: str = "patient_id") -> dict:
    """Describe the size and label balance of each split.

    Args:
        cohort: Table the indices refer to.
        splits: Positional indices keyed by split name.
        group_key: Column identifying the unit the labels belong to.
    """
    summary = {}
    for name, indices in splits.items():
        rows = cohort.iloc[indices]
        labels = rows.drop_duplicates(group_key)
        summary[name] = {
            "num_groups": int(rows[group_key].nunique()),
            "num_slides": len(rows),
            "num_events": int(labels["event"].sum()),
            "event_rate": round(float(labels["event"].mean()), 4) if len(labels) else 0.0,
        }
    return summary
