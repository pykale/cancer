"""Quickstart for the cohort/preprocessor/view API, on the HANCOCK clinical table.

The same job as ``main.py``, written against the current API. ``main.py`` and the
scripts in ``other/`` still use the removed ``TabularDataset`` API and no longer
run; they are kept for reference until they are ported.

The shape to take away is three objects with three different lifetimes:

* a **cohort** is an index over patients. Built once, read by every fold, never
  mutated. It holds no fitted state at all.
* a **preprocessor** is fitted on the rows you name and belongs to that fold. It
  records which patients it was fitted on, so the claim "this fold never saw the
  test set" is checkable rather than merely intended.
* a **view** pairs a row subset with one preprocessor. It is the only
  ``torch.utils.data.Dataset`` here, and the thing a ``DataLoader`` iterates.

You cannot build a view without naming a preprocessor, which is what keeps "whose
statistics is this fold using" visible in the script instead of hidden in object
state.

Run with::

    uv run python examples/HANCOCK_tabular/quickstart.py
"""

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kalecancer.loaddata import CohortDataModule, TabularCohort
from kalecancer.model.embed import TabICLEncoder
from kalecancer.survival import SurvivalTarget

SEED = 0
DATA_LOC = "data/StructuredData/clinical_data.json"


# --------------------------------------------------------------------------- #
# 1. Declare the cohort
# --------------------------------------------------------------------------- #
# Nothing is fitted here and nothing is imputed by default. Every transform you
# want is one you wrote down, so the methods section of your paper can be read
# off this block.

cohort = TabularCohort(
    source=DATA_LOC,
    identifier="patient_id",
    target=SurvivalTarget(
        time="days_to_last_information",
        event="survival_status",
        event_value="deceased",
        unit="days",
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
# split() returns *indices*, not cohorts. That is what lets it compose with
# scikit-learn's splitters (KFold, StratifiedKFold, ...) when you move from a
# single split to cross-validation. It stratifies on event status by default,
# because an unstratified 20% split of a few hundred patients can easily land
# with a badly skewed event rate.

train_idx, test_idx = cohort.split(test_size=0.2, random_state=SEED)
print(f"\ntrain {len(train_idx)} patients | test {len(test_idx)} patients")


# --------------------------------------------------------------------------- #
# 3. Prepare the fold
# --------------------------------------------------------------------------- #
# Fit on the training rows only, then apply those same statistics to both halves.
# Note that `test` is built with `prep`, not with a preprocessor of its own --
# fitting a second one on the test rows is the leak this API is shaped to make
# visible.

prep = cohort.fit_preprocessor(train_idx)
train = cohort.view(train_idx, prep)
test = cohort.view(test_idx, prep)

print(f"\n{prep.describe()}")
# Feature names are keyed by modality -- one key here, several once a slide
# cohort is composed alongside this one.
print(f"features ({prep.width}): {prep.feature_names['clinical']}")

# The preprocessor knows whose rows are inside it, so the fold boundary is a fact
# you can assert rather than a convention you hope was followed.
assert set(test.identifiers).isdisjoint(prep.fitted_on)
print(f"\nno test patient contributed to the fitted statistics ({len(prep.fitted_on)} train rows)")


# --------------------------------------------------------------------------- #
# 4. Look at one patient
# --------------------------------------------------------------------------- #
# A view yields PatientSample objects: features by modality, a per-modality
# availability flag, and the target's values under named keys. Named, never a
# positional [time, event] pair -- that is the pair everyone eventually builds
# backwards, and it runs perfectly while predicting survival inverted.

sample = train[0]
print(f"\npatient {sample.patient_id}")
print(f"  clinical : {tuple(sample.modalities['clinical'].shape)} {sample.modalities['clinical'].dtype}")
print(f"  present  : {sample.present}")
print(f"  target   : {sample.target}")
print(f"  sample type : {type(sample)}")



# --------------------------------------------------------------------------- #
# 5. Hand it to Lightning
# --------------------------------------------------------------------------- #
# CohortDataModule is a fold-local object: it fits this fold's preprocessor in
# setup(), builds the views, and hands a Trainer its loaders. One per fold.
#
# batch_size is required and has no default. With a Cox head the partial
# likelihood is averaged within a batch, so batch size selects the risk-set
# approximation -- a modelling decision, not a loader detail. "full" is the exact
# partial likelihood and is affordable at this cohort size.

dm = CohortDataModule(cohort, train_idx, test_idx=test_idx, batch_size="full", shuffle=True)
dm.setup()

batch = next(iter(dm.train_dataloader()))
print(f"\n{dm!r}")
print(f"  batch of {len(batch)}: clinical {tuple(batch.modalities['clinical'].shape)}, "
      f"time {tuple(batch.target['time'].shape)}, event {tuple(batch.target['event'].shape)}")
print(f"  events in batch: {int(batch.target['event'].sum())}")

# From here a LightningModule would go straight into:
#     L.Trainer(max_epochs=50).fit(model, datamodule=dm)


# --------------------------------------------------------------------------- #
# 6. Encode with a frozen foundation model
# --------------------------------------------------------------------------- #
# TabICL predicts by conditioning on context rows, so "fitting" it means storing
# them -- which makes the context fold state in exactly the way a scaler mean is.
# Fit it on the training view only.
#
# Downloads a ~100MB checkpoint from Hugging Face on first use.

encoder = TabICLEncoder(random_state=SEED).fit(train)

encoded_train = encoder.encode(train)
encoded_test = encoder.encode(test)

print(f"\n{encoder}")
print(f"  encoded_train: {tuple(encoded_train.shape)}")
print(f"  encoded_test : {tuple(encoded_test.shape)}")
