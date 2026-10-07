"""Survival prediction on synthetic data.

The simplest way to use the kalecancer Pipeline.

Each patient has a small clinical table row and a bag of patch features (what a
whole-slide image encoder would produce). They are fused and trained end to end
with a Cox head.

Runs on a CPU in under a minute and downloads nothing.

Run with: uv run python examples/synthetic/synthetic_survival.py
"""

import tempfile
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from kalecancer.evaluate import HarrellC
from kalecancer.loaddata import MultimodalDataset, PatchFeatures, TimeToEvent, train_test_split, Tabular
from kalecancer.model import ABMIL, Concat, CoxHead, IntermediateFusion
from kalecancer.pipeline import Pipeline

# Synthetic cohort: a clinical table, one h5 file of patch features per patient in a temporary folder, and censored
# survival times.

N_PATIENTS = 300
PATCH_DIM = 32


def make_cohort(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write one h5 file of patch features per patient into ``folder``; return the clinical table and outcomes."""
    rng = np.random.default_rng(0)
    ids = pd.Index([f"patient_{k:03d}" for k in range(N_PATIENTS)], name="patient_id")
    clinical = pd.DataFrame({"age": rng.normal(60, 10, N_PATIENTS), "stage": rng.integers(1, 5, N_PATIENTS)}, index=ids)

    aggressive_fraction = rng.uniform(0, 0.5, N_PATIENTS)
    for patient, fraction in zip(ids, aggressive_fraction, strict=True):
        n_patches = int(rng.integers(20, 200))
        features = rng.normal(size=(n_patches, PATCH_DIM)).astype(np.float32)
        features[: int(fraction * n_patches), 0] += 3.0  # aggressive patches stand out in one feature
        with h5py.File(folder / f"{patient}.h5", "w") as f:
            f["features"] = features
            f["coords"] = rng.integers(0, 50_000, size=(n_patches, 2))

    risk = 0.09 * (clinical["age"] - 60) + 1.2 * (clinical["stage"] - 2.5) + 12.0 * aggressive_fraction
    event_time = rng.exponential(2000 * np.exp(-risk))
    censoring_time = rng.uniform(200, 3000, N_PATIENTS)
    outcome = pd.DataFrame(
        {"time": np.minimum(event_time, censoring_time), "event": event_time <= censoring_time}, index=ids
    )
    return clinical, outcome


workdir = tempfile.TemporaryDirectory()
folder = Path(workdir.name)
clinical, outcome = make_cohort(folder)

# 1. Data: modalities and a target, all indexed by patient id, and which modalities every patient must have.

data = MultimodalDataset(
    modalities={
        "clinical": Tabular(clinical),
        "slides": PatchFeatures.from_glob(f"{folder}/*.h5", id_pattern=r"(patient_\d{3})\.h5$"),
    },
    target=TimeToEvent(time=outcome["time"], event=outcome["event"]),
    required_modalities=["clinical", "slides"],
)
print(data.summary())
train, test = train_test_split(data, test_size=0.25, stratify=True, random_state=0)

# 2. Model: stages for each modality, a fusion method, and a prediction head.

model = IntermediateFusion(
    encoding={
        "clinical": [torch.nn.Linear(2, 8)],
        "slides": [
            ABMIL(in_dim=PATCH_DIM, hidden_dim=16, attention_dim=8, dropout=0.0),
            torch.nn.Linear(16, 8),
        ],
    },
    fusion=Concat(),
    head=CoxHead(in_dim=16, ties="efron"),
)

# 3. Pipeline: preprocessing and training settings. Transforms are 'trained' on the training patient only, and then
# used to transform the test data. For a validation split with early stopping, see
# `examples/hancock/survival_intermediate.py`.

pipeline = Pipeline(
    model=model,
    transforms={"clinical": StandardScaler()},
    optimizer=partial(torch.optim.AdamW, lr=1e-3),
    batch_size=32,
    drop_last=True,
    max_epochs=40,
    validation=None,
    early_stopping=None,
    accelerator="cpu",
    precision="32-true",
    random_state=0,
)

# 4. Train and evaluate: fit on training data, and predict on test data.

pipeline.fit(train)
print(pipeline.evaluate(test, metrics={"harrell_c": HarrellC()}))
print(pipeline.predict(test).head())

workdir.cleanup()
