"""Survival prediction on synthetic data.

The simplest way to use the kalecancer Pipeline.

Each patient has a small clinical table row and a bag of patch features (what a
whole-slide image encoder would produce). They are fused and trained end to end
with a Cox head.

Runs on a CPU in under a minute and downloads nothing.

Run with: uv run python examples/synthetic/survival.py
"""

import pandas as pd
import tempfile
import torch
from functools import partial
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from kalecancer.evaluate import EvalContext, HarrellC, bootstrap_ci
from kalecancer.loaddata import MultimodalDataset, PatchFeatures, TimeToEvent, train_test_split, Tabular
from kalecancer.model import ABMIL, Concat, CoxHead, IntermediateFusion
from kalecancer.pipeline import Pipeline
from synthetic_data_generation import PATCH_DIM, make_cohort

# Synthetic cohort: a clinical table with some missing values, one h5 file of patch features per patient in a
# temporary folder, and censored survival times. See `synthetic_data_generation.py`.

workdir = tempfile.TemporaryDirectory()
folder = Path(workdir.name)
make_cohort(folder, n_patients=300, random_state=0, task="survival")
table = pd.read_csv(folder / "clinical.csv", index_col="patient_id")
clinical, outcome = table.drop(columns=["time", "event"]), table[["time", "event"]]

# 1. Data: modalities and a target, all indexed by patient id, and which modalities every patient must have.

data = MultimodalDataset(
    modalities={
        "clinical": Tabular(clinical),
        "slides": PatchFeatures.from_glob(f"{folder}/patches/*.h5", id_pattern=r"(patient_\d{3})\.h5$"),
    },
    target=TimeToEvent(time=outcome["time"], event=outcome["event"]),
    required_modalities=["clinical", "slides"],
)
print(data.summary())
train, test = train_test_split(data, test_size=0.25, stratify=True, random_state=0)

# 2. Model: stages for each modality, a fusion method, and a prediction head. Pipeline(random_state=...) seeds
# training, but the initial weights are drawn here, so seed them too.

torch.manual_seed(0)
model = IntermediateFusion(
    encoders={
        "clinical": [torch.nn.Linear(3, 8)],
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

clinical_transform = ColumnTransformer(
    [
        ("numeric", make_pipeline(SimpleImputer(), StandardScaler()), ["age", "stage"]),
        ("categorical", OrdinalEncoder(encoded_missing_value=-1), ["gender"]),
    ]
)
pipeline = Pipeline(
    model=model,
    transforms={"clinical": clinical_transform},
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
prediction = pipeline.predict(test)
print(prediction.head())

# 5. Uncertainty: a 95% bootstrap interval of Harrell's C over the test patients.

assert test.target is not None
target = test.target[prediction.index]
context = EvalContext(train_target=pipeline.train_target_, classes=None)
print("Harrell's C, 95% CI:", bootstrap_ci(HarrellC(), prediction, target, context))

workdir.cleanup()
