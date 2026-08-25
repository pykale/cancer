"""The two endpoints Dörrich et al. predict, and the published train/test splits.

Both endpoints are binary and both are defined by *filtering* rows before labelling:
a patient whose outcome cannot be determined is dropped rather than called negative.
That is why these are frame operations here and not a target object.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

LABEL = "label"


def survival_status_cohort(frame: pd.DataFrame, cfg) -> pd.DataFrame:
    """Vital status, restricted to deaths that can be attributed to the tumour.

    Deaths recorded as not tumour specific are excluded rather than counted as
    negative: the patient did die, so calling them a survivor would be wrong, and the
    tumour did not kill them, so calling them a positive would be wrong too. Nine
    patients recorded simply as "deceased" carry no cause and are kept as positives,
    which is what the published code does.

    Returns:
        The frame with a 0/1 ``label`` column, restricted to usable rows.
    """
    excluded = frame["survival_status_with_cause"].isin(list(cfg.CLASSIFY.EXCLUDE_STATUS))
    kept = frame[~excluded].copy()
    kept[LABEL] = (kept["survival_status"] == "deceased").astype(int)
    logger.info("survival_status: %d patients, %d excluded by cause", len(kept), int(excluded.sum()))
    return kept


def recurrence_cohort(frame: pd.DataFrame, cfg) -> pd.DataFrame:
    """Recurrence within three years.

    Follows the *published code* rather than the paper's prose, which differ. The code
    keeps::

        (recurrence == "yes" and days_to_recurrence <= horizon)
        or (recurrence == "no" and (days_to_last_information > horizon
                                    or survival_status == "living"))

    The trailing ``or survival_status == "living"`` admits patients who are alive and
    recurrence-free with less than three years of follow-up, whom the prose would
    exclude for insufficient follow-up. It roughly doubles the negative class. We
    follow the code because that is what produced the published numbers.

    Patients whose recurrence came after the horizon are dropped: a late recurrence is
    not a three-year recurrence, and calling it negative would contradict the label.

    Returns:
        The frame with a 0/1 ``label`` column, restricted to usable rows.
    """
    horizon = cfg.CLASSIFY.RECURRENCE_HORIZON_DAYS
    recurred = (frame["recurrence"] == "yes") & (frame["days_to_recurrence"] <= horizon)
    disease_free = (frame["recurrence"] == "no") & (
        (frame["days_to_last_information"] > horizon) | (frame["survival_status"] == "living")
    )

    kept = frame[recurred | disease_free].copy()
    kept[LABEL] = (kept["recurrence"] == "yes").astype(int)
    logger.info(
        "recurrence: %d patients (%d recurred, %d disease-free), %d dropped as undeterminable",
        len(kept),
        int(recurred.sum()),
        int(disease_free.sum()),
        len(frame) - len(kept),
    )
    return kept


#: Endpoint name to the function that builds it.
COHORTS = {"survival_status": survival_status_cohort, "recurrence": recurrence_cohort}


def labelled_cohort(frame: pd.DataFrame, cfg, target: str | None = None) -> pd.DataFrame:
    """Apply an endpoint's filter and labelling.

    Args:
        frame: The merged structured table.
        cfg: Experiment configuration, for the horizon and exclusion rules.
        target: Endpoint name; defaults to the configured one.

    Raises:
        KeyError: If the target is not one of the two endpoints.
    """
    target = target or cfg.CLASSIFY.TARGET
    if target not in COHORTS:
        raise KeyError(f"unknown target {target!r}; available: {sorted(COHORTS)}")
    return COHORTS[target](frame, cfg)


def official_split(path) -> dict[str, list[str]]:
    """Read a published train/test assignment.

    Args:
        path: One of the ``dataset_split_*.json`` files.

    Returns:
        Patient identifiers keyed ``"training"`` and ``"test"``.
    """
    assignment = pd.read_json(path, dtype={"patient_id": str})
    return {name: rows["patient_id"].tolist() for name, rows in assignment.groupby("dataset")}
