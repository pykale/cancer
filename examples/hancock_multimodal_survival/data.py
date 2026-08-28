"""Joining HANCOCK's clinical table and slide features into one cohort.

Reuses the pieces each single-modality example already established:

* clinical -- :class:`~kalecancer.loaddata.tabular.TabularCohort` with the same column
  roles and transforms as ``examples/hancock_tabular_survival``, embedded by
  :class:`~kalecancer.model.embed.TabICLEmbedder`;
* imaging -- :func:`~kalecancer.loaddata.wsi_access.slide_table` and
  :func:`~kalecancer.loaddata.wsi_access.read_feature_bag`, as in
  ``examples/hancock_wsi_survival``.

TabICL is frozen and re-embeds its whole context on every call, so clinical rows are
embedded once here rather than inside the training loop. What the model then learns
on that modality is the projection, which is why the trainer's clinical embedder is
an :class:`~kalecancer.model.embed.MLPEmbedder`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from examples.hancock.dataset import SLIDE_PATTERN
from kalecancer.loaddata import SurvivalTarget
from kalecancer.loaddata.tabular_access import TabularCohort
from kalecancer.loaddata.wsi_access import slide_table
from kalecancer.model.embed import TabICLEmbedder

logger = logging.getLogger(__name__)

CLINICAL = "clinical"
IMAGING = "imaging"


def build_tabular_cohort(clinical_path: str | Path, cfg) -> TabularCohort:
    """The clinical cohort, encoded exactly as the tabular example encodes it."""
    return TabularCohort(
        clinical_path,
        identifier="patient_id",
        target=SurvivalTarget(
            time=cfg.SURVIVAL.TIME_FIELD,
            event=cfg.SURVIVAL.STATUS_FIELD,
            event_value=cfg.SURVIVAL.EVENT_VALUES[0],
        ),
        continuous=list(cfg.TABULAR.CONTINUOUS),
        continuous_transform=make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
        categorical=list(cfg.TABULAR.CATEGORICAL),
        categorical_transform=make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
    )


def slides_by_patient(feature_root: str | Path) -> dict[str, list[Path]]:
    """Every patient's slide feature files, keyed by patient id."""
    table = slide_table(feature_root, SLIDE_PATTERN)
    return {patient: list(rows["path"]) for patient, rows in table.groupby("patient_id")}


def embed_clinical(cohort: TabularCohort, train_ids: list[str], all_ids: list[str], cfg) -> dict[str, torch.Tensor]:
    """Embed every patient's clinical row, with TabICL's context taken from training.

    The context is fold state, like a scaler's mean: built from the training rows so
    the test rows never inform the representation they are scored with.

    Raises:
        ValueError: If ``cohort`` carries no target to draw context labels from.
    """
    if cohort.target is None:
        raise ValueError("embedding needs a labelled cohort: TabICL conditions on labelled context rows")

    preprocessor = cohort.fit_preprocessor(train_ids)
    if set(all_ids) & set(preprocessor.fitted_on) != set(train_ids) & set(preprocessor.fitted_on):
        raise AssertionError("preprocessor was fitted on rows outside the training split")

    context = cohort.view(train_ids, preprocessor)
    embedder = TabICLEmbedder(
        context_x=context.batch().modalities[CLINICAL],
        context_y=cohort.target.values_for(context.identifiers)["event"],
        trainable=cfg.TABULAR.TRAINABLE,
        random_state=cfg.SOLVER.SEED,
        checkpoint=cfg.TABULAR.CHECKPOINT or None,
        n_estimators=cfg.TABULAR.N_ESTIMATORS,
    )

    view = cohort.view(all_ids, preprocessor)
    with torch.no_grad():
        embeddings = embedder(view.batch().modalities[CLINICAL]).cpu()
    logger.info("clinical embeddings %s from %d context rows", tuple(embeddings.shape), len(train_ids))
    return dict(zip(view.identifiers, embeddings, strict=True))
