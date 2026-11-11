from dataclasses import dataclass
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from kalecancer.loaddata import BaseTarget, Classification, MultimodalDataset, PatchFeatures, Tabular, TimeToEvent
from kalecancer.pipeline import Pipeline
from kalecancer.prepdata import ColumnGroup, TableTransform


@dataclass
class Cohort:
    root: Path
    ids: list[str]
    clinical: pd.DataFrame
    wsi_pattern: str
    id_pattern: str
    wsi_files: pd.Series
    bags: dict[str, np.ndarray]
    coords: dict[str, np.ndarray]
    time: pd.Series
    event: pd.Series
    labels: pd.Series


def _write_bag(path: Path, features: np.ndarray, coords: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f["features"] = features
        f["coords"] = coords


def make_cohort(root: Path, n_patients: int = 40, n_with_wsi: int = 30, dim: int = 16, seed: int = 0) -> Cohort:
    """Synthetic cohort with signal. Patient "005" has two slide files; the last patients have no slides."""
    rng = np.random.default_rng(seed)
    ids = [f"{k:03d}" for k in range(1, n_patients + 1)]
    age = rng.normal(60, 10, n_patients)
    sex = rng.choice(["female", "male"], n_patients).astype(object)
    sex[3] = None
    smoking = rng.choice(["never", "former", "current"], n_patients).astype(object)
    clinical = pd.DataFrame({"age": age, "sex": sex, "smoking": smoking}, index=pd.Index(ids, name="patient_id"))

    bags, coords = {}, {}
    paths: dict[str, list] = {}
    signal = np.zeros(n_patients)
    for k, pid in enumerate(ids[:n_with_wsi]):
        n_files = 2 if pid == "005" else 1
        parts, xy_parts = [], []
        for j in range(n_files):
            n = int(rng.integers(5, 60))
            features = rng.standard_normal((n, dim)).astype(np.float32)
            xy = np.stack([np.arange(n) * 256 + j * 100_000, np.full(n, k * 256)], axis=1).astype(np.int32)
            suffix = "" if j == 0 else "_a"
            site = "site_a" if k % 2 == 0 else "site_b"
            path = root / site / "h5_files" / f"PT_{pid}{suffix}.h5"
            _write_bag(path, features, xy)
            paths.setdefault(pid, []).append(str(path))
            parts.append(features)
            xy_parts.append(xy)
        bags[pid] = np.concatenate(parts)
        coords[pid] = np.concatenate(xy_parts)
        signal[k] = bags[pid][:, 0].mean()

    risk = 0.05 * (age - 60) + signal
    time = rng.exponential(1000 * np.exp(-risk)) + 1.0
    event = rng.random(n_patients) < 0.5
    labels = np.where(risk > np.median(risk), "high", "low")

    index = pd.Index(ids, name="patient_id")
    wsi_files = pd.Series(
        [p for pid in paths for p in paths[pid]], index=pd.Index([pid for pid in paths for _ in paths[pid]], name="id")
    )
    return Cohort(
        root=root,
        ids=ids,
        clinical=clinical,
        wsi_pattern=str(root / "*" / "h5_files" / "*.h5"),
        id_pattern=r"PT_(\d{3})(?:_[a-z])?\.h5$",
        wsi_files=wsi_files,
        bags=bags,
        coords=coords,
        time=pd.Series(time, index=index),
        event=pd.Series(event, index=index),
        labels=pd.Series(labels, index=index),
    )


@pytest.fixture
def cohort(tmp_path: Path) -> Cohort:
    return make_cohort(tmp_path)


@pytest.fixture
def cohort_factory(tmp_path: Path):
    """A factory for a ``Cohort`` of a custom size/seed, for tests that need one other than ``cohort``'s default."""

    def factory(**kwargs) -> Cohort:
        return make_cohort(tmp_path, **kwargs)

    return factory


def _clinical_transform() -> TableTransform:
    return TableTransform(
        groups={
            "numeric": ColumnGroup(["age"], [StandardScaler()]),
            "categorical": ColumnGroup(
                ["sex", "smoking"],
                [
                    SimpleImputer(strategy="constant", fill_value="missing"),
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ],
            ),
        },
        drop=[],
    )


@pytest.fixture
def dataset_factory():
    """A factory for a ``MultimodalDataset`` of a ``Cohort``'s clinical and wsi modalities."""

    def factory(cohort: Cohort, required=("clinical", "wsi"), task="survival", event=None) -> MultimodalDataset:
        target: BaseTarget | None
        if task == "survival":
            target = TimeToEvent(cohort.time, cohort.event if event is None else event)
        elif task == "classification":
            target = Classification(cohort.labels, classes=["low", "high"])
        else:
            target = None
        modalities = {
            "clinical": Tabular(cohort.clinical),
            "wsi": PatchFeatures(cohort.wsi_files, multiple_files="concatenate"),
        }
        return MultimodalDataset(modalities, target=target, required_modalities=list(required))

    return factory


@pytest.fixture
def split_factory():
    """A factory splitting a dataset's ids into a leading training set and a trailing test set."""

    def factory(data: MultimodalDataset, n_train: int = 22) -> tuple[MultimodalDataset, MultimodalDataset]:
        return data.subset(data.ids[:n_train]), data.subset(data.ids[n_train:])

    return factory


@pytest.fixture
def pipeline_factory():
    """A factory for a ``Pipeline`` with this test suite's standard training settings, built around a model."""

    def factory(model, **overrides) -> Pipeline:
        arguments = {
            "model": model,
            "transforms": {"clinical": _clinical_transform()},
            "optimizer": partial(torch.optim.AdamW, lr=1e-2),
            "batch_size": 8,
            "drop_last": True,
            "max_epochs": 2,
            "validation": None,
            "early_stopping": None,
            "accelerator": "cpu",
            "precision": "32-true",
            "random_state": 0,
        }
        arguments.update(overrides)
        return Pipeline(**arguments)

    return factory
