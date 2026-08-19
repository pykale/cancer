"""Joining slide feature files to clinical survival labels.

Produces one row per slide, which is the representation the generic splitting and
dataset utilities consume:

======================  ==========================================================
``patient_id``          Identifier shared by a patient's slides; the grouping key
``slide_id``            Identifier of the individual slide
``path``                Feature file for the slide
``duration``, ``event`` Survival label, repeated for each of a patient's slides
======================  ==========================================================

Slides without a survival label, and labelled patients without usable features, are
dropped from the table; :func:`~kalecancer.evaluate.cohort_report.cohort_summary`
reports what was excluded.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from kalecancer.loaddata.clinical_access import load_clinical_records, survival_table
from kalecancer.loaddata.wsi_feature_access import (
    DEFAULT_SLIDE_PATTERN,
    InvalidFeatureFileError,
    inspect_feature_bag,
    slide_table,
)

COHORT_COLUMNS = ["patient_id", "slide_id", "path", "duration", "event"]


def build_cohort(
    feature_root: str | Path,
    clinical_path: str | Path,
    endpoint: str = "OS",
    expected_dim: int | None = None,
    slide_pattern: re.Pattern[str] = DEFAULT_SLIDE_PATTERN,
    validate_features: bool = True,
) -> pd.DataFrame:
    """Build the cohort table for a set of slides and clinical records.

    Args:
        feature_root: Directory of slide feature files, searched recursively.
        clinical_path: JSON file of clinical records.
        endpoint: Survival endpoint, see :data:`~kalecancer.loaddata.clinical_access.ENDPOINTS`.
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
        endpoint=endpoint,
        num_clinical_patients=len(records),
        slides=slides,
        survival=survival,
        invalid_features=invalid,
        unparsed_files=slides.attrs.get("unparsed", []),
        clinical_exclusions={reason: ids for reason, ids in exclusions.items() if ids},
    )
    return cohort
