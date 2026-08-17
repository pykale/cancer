# # `TabICLEncoder` — row embeddings for a clinical table
#
# TabICL is a pretrained tabular foundation model. This wrapper uses it **frozen**: no
# gradients, no fine-tuning, one 512-d vector per patient, ready to feed to a head.
#
# Needs the optional extra — `pip install 'kalecancer[tabular]'`. The checkpoint is
# downloaded from Hugging Face on first use.
#
# ```python
# encoder = TabICLEncoder().fit(fitted_train)   # absorb the training fold as context
# Z_test = encoder.encode(fitted_test)          # (n_test, 512)
# ```
#
# `fit` and `encode` are separate because **the context is fitted state** — in exactly
# the sense a `StandardScaler`'s mean is. Sections 3 and 4 measure that, and measure the
# property that makes encoding safe to do in batches.


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kalecancer.loaddata.tabular import TabularDataset
from kalecancer.model.embed import TabICLEncoder
from kalecancer.survival.survival_target import SurvivalTarget

DATA = "data/StructuredData/clinical_data.json"
SEED = 0

# ## 1. The usual data preparation
#
# Nothing here is specific to the encoder — see `detailed_loading_demo.py` for what each
# argument does. Split, fit the training fold, transform the held-out fold.

cohort = TabularDataset(
    DATA,
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

train, test = cohort.split(test_size=0.2, random_state=SEED)
fitted_train = train.fit_transform()
fitted_test = fitted_train.transform(test)

print("train:", fitted_train)
print("test: ", fitted_test)

# ## 2. Fit the context, then encode
#
# `fit` takes no gradient step — TabICL predicts by conditioning on labelled context
# rows, so "fitting" means storing them. It needs a dataset that carries a target,
# because TabICL's column embedder is label-aware.
#
# Like `TabularDataset.fit_transform`, it returns a **new** encoder and leaves the
# original alone. That is what keeps cross-validation folds from sharing a context — and
# it is the one thing to watch, because the sklearn reflex of writing
# `encoder.fit(train)` on its own line silently does nothing.

base = TabICLEncoder(random_state=SEED)
encoder = base.fit(fitted_train)  # a NEW encoder; `base` is untouched

print("base:   ", base)
print("encoder:", encoder)
print("out_dim:", encoder.out_dim)

Z_train = encoder.encode(fitted_train)
Z_test = encoder.encode(fitted_test)

print(f"Z_train {tuple(Z_train.shape)}")
print(f"Z_test  {tuple(Z_test.shape)}")

# ## 3. The context is fold state
#
# Encode the *same* test rows against two different contexts and the embeddings move.
# That is the whole reason `fit` exists as a separate step: fit the context on the full
# cohort and every embedding — including the held-out ones — carries information from
# rows the model should never have seen. No error, no warning, and correct
# `ColumnTransformer` handling does not save you.
#
# If the number below ever reads `0.0000`, the encoder has stopped conditioning on its
# context at all.


half = fitted_train.subset(range(len(fitted_train) // 2))
Z_test_half = TabICLEncoder(random_state=SEED).fit(half).encode(fitted_test)

drift = (Z_test - Z_test_half).abs().mean().item()
print(f"{len(fitted_test)} test rows, encoded against two different contexts:")
print(f"  context = {len(fitted_train)} rows  vs  context = {len(half)} rows")
print(f"  mean |difference| per dimension: {drift:.4f}")

# ## 4. ...but encoding is batch-invariant
#
# The converse property: context rows do not attend to the rows being encoded, so a
# row's embedding does not depend on what else was in the call. You can chunk `encode`
# however you like and get identical answers.
#
# Anything but `0.00000000` below would mean held-out rows are reaching each other's
# representations.


Z_alone = encoder.encode(fitted_test.subset(range(10)))

leak = (Z_test[:10] - Z_alone).abs().max().item()
print(f"same encoder, same 10 rows, alongside {len(fitted_test) - 10} others vs on their own:")
print(f"  max |difference| over all dimensions: {leak:.8f}")

# ## Next
#
# `Z_train` and `Z_test` are ordinary tensors, aligned with `fitted_train.identifiers`
# and `fitted_test.identifiers`. `../main.py` puts a Cox head on top of them and trains
# it, driven by a YAML config.
#
# One caveat worth knowing: rows that are *in* the context see their own label when
# encoded, so `encode(fitted_train)` is optimistic relative to `encode(fitted_test)`.
# That is inherent to in-context learning rather than a defect here, but a head fitted
# on training representations and applied to held-out ones is crossing a distribution
# boundary. Removing it properly needs cross-fitted contexts, which this class does not
# do.
