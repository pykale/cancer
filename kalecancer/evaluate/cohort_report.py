"""Reporting on a matched cohort.

Kept apart from loading: the loaders build the cohort table, and these functions
describe what it contains and what was excluded.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def cohort_summary(cohort: pd.DataFrame, group_key: str = "patient_id") -> dict:
    """Summarise a cohort table and the exclusions recorded while building it.

    Args:
        cohort: Table from :func:`~kalecancer.loaddata.cohort.build_cohort`.
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
