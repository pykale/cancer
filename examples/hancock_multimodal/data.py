"""Joining HANCOCK's clinical table and slide features into one cohort.

Reuses the pieces each single-modality example already established:

* clinical -- :class:`~kalecancer.loaddata.tabular.TabularCohort` with the same column
  roles and transforms as ``examples/HANCOCK_tabular``, embedded by
  :class:`~kalecancer.model.embed.TabICLEmbedder`;
* imaging -- :func:`~kalecancer.loaddata.wsi_feature_access.slide_table` and
  :func:`~kalecancer.loaddata.wsi_feature_access.read_feature_bag`, as in
  ``examples/wsi_survival``.

TabICL is frozen and re-embeds its whole context on every call, so clinical rows are
embedded once here rather than inside the training loop. What the model then learns
on that modality is the projection, which is why the trainer's clinical embedder is
an :class:`~kalecancer.model.embed.MLPEmbedder`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import Dataset

from kalecancer.loaddata.sample import PatientBatch
from kalecancer.loaddata.tabular import TabularCohort
from kalecancer.loaddata.wsi_feature_access import read_feature_bag, slide_table
from kalecancer.model.embed import TabICLEmbedder
from kalecancer.survival.survival_target import SurvivalTarget

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


def official_split(path: str | Path) -> dict[str, list[str]]:
    """Read the published train/test assignment, keyed by split name."""
    assignments = json.loads(Path(path).read_text(encoding="utf-8"))
    grouped: dict[str, list[str]] = {}
    for row in assignments:
        grouped.setdefault(str(row["dataset"]), []).append(str(row["patient_id"]))
    return grouped


def slides_by_patient(feature_root: str | Path) -> dict[str, list[Path]]:
    """Every patient's slide feature files, keyed by patient id."""
    table = slide_table(feature_root)
    return {patient: list(rows["path"]) for patient, rows in table.groupby("patient_id")}


def embed_clinical(cohort: TabularCohort, train_ids: list[str], all_ids: list[str], cfg) -> dict[str, torch.Tensor]:
    """Embed every patient's clinical row, with TabICL's context taken from training.

    The context is fold state, like a scaler's mean: built from the training rows so
    the test rows never inform the representation they are scored with.
    """
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


class MultimodalCohort(Dataset):
    """One patient per item, carrying whichever modalities the run selected.

    Args:
        patient_ids: Patients in this split.
        modalities: Modality names to emit.
        clinical: Precomputed clinical embeddings, by patient.
        slides: Slide feature files, by patient.
        targets: ``time`` and ``event`` per patient.
        feature_dim: Patch feature width, used to shape a placeholder bag.
        max_patches: Cap on patches per bag; ``None`` uses the whole bag.
        seed: Base seed for patch subsampling.
    """

    def __init__(
        self,
        patient_ids: list[str],
        modalities: list[str],
        clinical: dict[str, torch.Tensor],
        slides: dict[str, list[Path]],
        targets: pd.DataFrame,
        feature_dim: int = 1024,
        max_patches: int | None = None,
        seed: int = 0,
    ) -> None:
        self.patient_ids = list(patient_ids)
        self.modalities = list(modalities)
        self.clinical = clinical
        self.slides = slides
        self.targets = targets.set_index("patient_id")
        self.feature_dim = feature_dim
        self.max_patches = max_patches
        self.seed = seed

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict:
        patient = self.patient_ids[index]
        row = self.targets.loc[patient]

        modalities: dict[str, torch.Tensor] = {}
        present: dict[str, torch.Tensor] = {}
        if CLINICAL in self.modalities:
            vector = self.clinical.get(patient)
            present[CLINICAL] = torch.tensor(vector is not None)
            modalities[CLINICAL] = vector if vector is not None else torch.zeros(self._clinical_dim())
        if IMAGING in self.modalities:
            bag = self._bag(patient, index)
            present[IMAGING] = torch.tensor(bag is not None)
            modalities[IMAGING] = bag if bag is not None else torch.zeros(1, self.feature_dim)

        return {
            "patient_id": patient,
            "modalities": modalities,
            "present": present,
            "target": {
                "time": torch.tensor(float(row["time"]), dtype=torch.float32),
                "event": torch.tensor(float(row["event"]), dtype=torch.float32),
            },
        }

    def _clinical_dim(self) -> int:
        return next(iter(self.clinical.values())).shape[0]

    def _bag(self, patient: str, index: int) -> torch.Tensor | None:
        paths = self.slides.get(patient)
        if not paths:
            return None

        features = torch.cat([torch.from_numpy(read_feature_bag(path)[0]) for path in paths])
        if self.max_patches and len(features) > self.max_patches:
            generator = torch.Generator().manual_seed(self.seed + index)
            keep = torch.randperm(len(features), generator=generator)[: self.max_patches]
            features = features[keep.sort().values]
        return features


def collate(samples: list[dict]) -> PatientBatch:
    """Collate into a :class:`PatientBatch`, keeping ragged slide bags in a list.

    Padding a batch to the largest bag would waste most of the tensor when patch
    counts vary by orders of magnitude, and attention pools each bag independently.
    """
    names = list(samples[0]["modalities"])
    modalities: dict = {}
    for name in names:
        values = [sample["modalities"][name] for sample in samples]
        modalities[name] = values if values[0].dim() > 1 else torch.stack(values)

    return PatientBatch(
        patient_id=[sample["patient_id"] for sample in samples],
        modalities=modalities,
        present={name: torch.stack([s["present"][name] for s in samples]) for name in names},
        target={key: torch.stack([s["target"][key] for s in samples]) for key in samples[0]["target"]},
    )
