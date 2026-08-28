"""HANCOCK's structured data as one leakage-safe feature matrix.

The published structured archive is three tables: clinical (demographics, treatment
and event history), pathological (staging, grading, invasion) and blood (one row per
analyte per patient). Dörrich et al. build their patient vector from all three, so
"tabular" here means all three, not demographics alone.

Encoding follows their Methods, mapped onto the two column roles
:class:`~kalecancer.loaddata.tabular.TabularCohort` offers:

=================  ====================================  ==================
Their role         Their transform                       Role used here
=================  ====================================  ==================
nominal            most-frequent impute, one-hot         ``categorical``
binary flags       most-frequent impute, no scaling      ``categorical``
blood, discrete    mean impute, standardise              ``continuous``
ordinal stage      integer rank, then as above           ``continuous``
=================  ====================================  ==================

The binary flags become two-column one-hots rather than staying single columns, which
no linear or MLP model can distinguish. Imputation is fitted inside the fold, so a
test patient never contributes to the statistic that fills its own gaps.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kalecancer.loaddata.tabular_access import TabularCohort

logger = logging.getLogger(__name__)

IDENTIFIER = "patient_id"
MODALITY = "tabular"

#: Stage rank, lowest burden first. ``TX``/``NX`` mean the stage was not assessed, so
#: they map to missing rather than to a rank: an unassessed stage is not a mild one.
T_STAGE_RANK = {"pTis": 0, "pT1": 1, "pT1a": 2, "pT1b": 3, "pT2": 4, "pT3": 5, "pT4a": 6, "pT4b": 7}
N_STAGE_RANK = {"pN0": 0, "pN1": 1, "pN1a": 2, "pN2": 3, "pN2a": 4, "pN2b": 5, "pN2c": 6, "pN3": 7, "pN3b": 8}
STAGE_RANKS = {"pT_stage": T_STAGE_RANK, "pN_stage": N_STAGE_RANK}


def read_table(path) -> pd.DataFrame:
    """Read a published JSON table, keeping the identifier's zero padding."""
    return pd.read_json(path, dtype={IDENTIFIER: str})


def hematology_parameters(reference_ranges: pd.DataFrame) -> list[str]:
    """The hematology analytes the paper keeps from the blood panel.

    Their rule: rows in the Hematology group that have a recorded reference range,
    excluding percentage differentials. The percentages are determined by the absolute
    counts already present, so they add no independent signal.

    Args:
        reference_ranges: The published ``blood_data_reference_ranges`` table.

    Returns:
        LOINC names, sorted so the feature order is stable across runs.
    """
    keep = reference_ranges[
        (reference_ranges["group"] == "Hematology") & reference_ranges["normal_female_max"].notnull()
    ]
    return sorted(name for name in keep["LOINC_name"] if not name.endswith("%"))


def blood_matrix(blood: pd.DataFrame, parameters: list[str]) -> pd.DataFrame:
    """Pivot the long blood table to one column per analyte.

    Args:
        blood: The published ``blood_data`` table, one row per measurement.
        parameters: LOINC names to keep.

    Returns:
        A frame carrying ``patient_id`` and one column per analyte, in the order given.
    """
    selected = blood[blood["LOINC_name"].isin(parameters)]
    wide = selected.pivot_table(index=IDENTIFIER, columns="LOINC_name", values="value", aggfunc="first")
    return wide.reindex(columns=parameters).reset_index()


def structured_frame(paths: dict, cfg) -> tuple[pd.DataFrame, list[str]]:
    """Merge the structured tables into one row per patient.

    Clinical and pathological are joined inner because the paper's vector needs both.
    Blood is joined left because not every patient has a panel; the gaps are filled by
    the fold-local imputer rather than by dropping the patient.

    Args:
        paths: Local paths keyed ``"clinical"``, ``"pathological"``, ``"blood"`` and
            ``"ranges"``.
        cfg: Experiment configuration.

    Returns:
        The merged frame with stage columns replaced by their integer rank, and the
        blood column names that were added.

    Raises:
        ValueError: If the identifier lost its zero padding, which would silently
            empty every join against the published splits.
    """
    frame = read_table(paths["clinical"]).merge(read_table(paths["pathological"]), on=IDENTIFIER, how="inner")

    blood_columns: list[str] = []
    if cfg.TABULAR.USE_BLOOD:
        blood_columns = hematology_parameters(read_table(paths["ranges"]))
        logger.info("blood: %d hematology analytes", len(blood_columns))
        frame = frame.merge(blood_matrix(read_table(paths["blood"]), blood_columns), on=IDENTIFIER, how="left")

    # Identifiers are zero-padded strings in every published file. Read as integers
    # anywhere and the padding is gone for good, leaving joins against the split files
    # that quietly match nothing rather than failing.
    if not frame[IDENTIFIER].str.fullmatch(r"\d{3}").all():
        raise ValueError(f"patient ids must stay zero-padded strings, got e.g. {frame[IDENTIFIER].head(3).tolist()}")

    for column, ranks in STAGE_RANKS.items():
        frame[column] = frame[column].map(ranks)

    # An unrecorded category becomes a level of its own rather than being filled with
    # the most common value. Whether a field was assessed is itself informative here
    # -- perinodal invasion is unrecorded for the 365 patients who had no neck
    # dissection -- and the reference implementation treats it the same way. Making it
    # explicit also stops the result depending on whether scikit-learn's imputer
    # recognises ``None`` in an object column, which it does not.
    categorical = list(cfg.TABULAR.BINARY) + list(cfg.TABULAR.NOMINAL)
    frame[categorical] = frame[categorical].fillna("unrecorded")
    return frame, blood_columns


def build_cohort(frame: pd.DataFrame, blood_columns: list[str], cfg) -> TabularCohort:
    """Wrap the merged frame with the paper's encoding.

    No target is attached: the label travels with the features in this example's own
    dataset, and the cohort is used for its leakage-safe preprocessor alone.
    """
    return TabularCohort(
        frame,
        identifier=IDENTIFIER,
        continuous=list(cfg.TABULAR.DISCRETE) + list(cfg.TABULAR.ORDINAL) + list(blood_columns),
        continuous_transform=make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()),
        categorical=list(cfg.TABULAR.BINARY) + list(cfg.TABULAR.NOMINAL),
        categorical_transform=make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
        name=MODALITY,
    )


def encode(cohort: TabularCohort, fit_ids: list[str], all_ids: list[str]) -> dict:
    """Fit the encoding on ``fit_ids`` and apply it to ``all_ids``.

    Args:
        cohort: The structured cohort.
        fit_ids: Training identifiers; the only rows any statistic is fitted on.
        all_ids: Identifiers to encode.

    Returns:
        One feature vector per identifier.

    Raises:
        AssertionError: If the preprocessor saw anything outside ``fit_ids``.
    """
    preprocessor = cohort.fit_preprocessor(fit_ids)
    assert set(preprocessor.fitted_on) == set(fit_ids), "preprocessor was fitted outside the training split"

    features = cohort.view(all_ids, preprocessor).batch().modalities[MODALITY]
    return dict(zip(all_ids, features, strict=True))
