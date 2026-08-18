"""Quickstart for the cohort/preprocessor/view API, on the HANCOCK clinical table.

Three objects with three lifetimes: a **cohort** indexes patients and is never
mutated; a **preprocessor** is fitted on the rows you name and belongs to that fold;
a **view** pairs a row subset with one preprocessor and is what a ``DataLoader``
iterates. You cannot build a view without naming a preprocessor, which is what keeps
a fold's statistics visible in the script.

See ``pipeline.py`` for the same thing driven from a YAML config.

Run with::

    uv run python examples/HANCOCK_tabular/quickstart.py
"""

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kalecancer.loaddata import CohortDataModule, TabularCohort
from kalecancer.model.embed import TabICLEncoder
from kalecancer.survival import SurvivalTarget

SEED = 0
DATA_LOC = "data/HANCOCK/raw/StructuredData/clinical_data.json"


# --------------------------------------------------------------------------- #
# 1. Declare the cohort
# --------------------------------------------------------------------------- #
# Nothing is fitted here, and nothing is imputed unless you asked for it -- so this
# block is the whole of the preprocessing.

cohort = TabularCohort(
    source=DATA_LOC,
    identifier="patient_id",
    target=SurvivalTarget(
        time="days_to_last_information",
        event="survival_status",
        event_value="deceased",
        unit="days",  # a display label only; nothing is converted
    ),
    continuous=["age_at_initial_diagnosis", "year_of_initial_diagnosis"],
    continuous_transform=[SimpleImputer(strategy="median"), StandardScaler()],
    categorical=["sex", "smoking_status", "primarily_metastasis"],
    categorical_transform=[
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    ],
)

print(cohort)
print(cohort.describe_transforms())


# --------------------------------------------------------------------------- #
# 2. Hold out a test set -- once
# --------------------------------------------------------------------------- #
# Indices, not cohorts, so this composes with KFold and friends later. `stratify` is
# required: unstratified, a 20% split of a few hundred patients can land several
# points off on the event rate.

train_idx, test_idx = cohort.split(test_size=0.2, random_state=SEED, stratify=True)
print(f"\ntrain {len(train_idx)} patients | test {len(test_idx)} patients")


# --------------------------------------------------------------------------- #
# 3. Prepare the fold
# --------------------------------------------------------------------------- #
# Fit on the training rows, then build both views from that same preprocessor.
# Fitting a second one on the test rows is the leak this API makes visible.

prep = cohort.fit_preprocessor(train_idx)
train = cohort.view(train_idx, prep)
test = cohort.view(test_idx, prep)

print(f"\n{prep.describe()}")
# Keyed by modality: one key here, several once a slide cohort joins it.
print(f"features ({prep.width}): {prep.feature_names['clinical']}")

# The preprocessor records whose rows are in it, so the fold boundary is assertable.
assert set(test.identifiers).isdisjoint(prep.fitted_on)
print(f"\nno test patient contributed to the fitted statistics ({len(prep.fitted_on)} train rows)")


# --------------------------------------------------------------------------- #
# 4. Look at one patient
# --------------------------------------------------------------------------- #
# A view yields PatientSample: features by modality, an availability flag per
# modality, and the target under named keys -- never a positional [time, event] pair,
# which runs perfectly when built backwards and predicts survival inverted.

sample = train[0]
print(f"\npatient {sample.patient_id}")
print(f"  clinical : {tuple(sample.modalities['clinical'].shape)} {sample.modalities['clinical'].dtype}")
print(f"  present  : {sample.present}")
print(f"  target   : {sample.target}")
print(f"  sample type : {type(sample)}")


# --------------------------------------------------------------------------- #
# 5. Hand it to Lightning
# --------------------------------------------------------------------------- #
# Fold-local: it fits the preprocessor in setup(), builds the views, and hands a
# Trainer its loaders. One per fold.
#
# batch_size has no default. With a Cox head the partial likelihood is averaged
# within a batch, so it selects the risk-set approximation -- a modelling decision,
# not a loader detail. "full" is exact and affordable at this size.

dm = CohortDataModule(cohort, train_idx, test_idx=test_idx, batch_size="full", shuffle=True)
dm.setup()

batch = next(iter(dm.train_dataloader()))
print(f"\n{dm!r}")
print(
    f"  batch of {len(batch)}: clinical {tuple(batch.modalities['clinical'].shape)}, "
    f"time {tuple(batch.target['time'].shape)}, event {tuple(batch.target['event'].shape)}"
)
print(f"  events in batch: {int(batch.target['event'].sum())}")

# From here a LightningModule would go straight into:
#     L.Trainer(max_epochs=50).fit(model, datamodule=dm)


# --------------------------------------------------------------------------- #
# 6. Encode with a frozen foundation model
# --------------------------------------------------------------------------- #
# TabICL conditions on context rows, so "fitting" is storing them -- making the
# context fold state exactly as a scaler mean is. Fit on the training view only.
# Downloads a ~100MB checkpoint from Hugging Face on first use.

encoder = TabICLEncoder(random_state=SEED).fit(train)

encoded_train = encoder.encode(train)
encoded_test = encoder.encode(test)

print(f"\n{encoder}")
print(f"  encoded_train: {tuple(encoded_train.shape)}")
print(f"  encoded_test : {tuple(encoded_test.shape)}")
