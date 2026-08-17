# # `TabularDataset` — a clinical table, end to end
#
# The HANCOCK head-and-neck cohort, from a JSON file to batches a model can train on.
#
# The whole API, in four lines:
#
# ```python
# cohort = TabularDataset(...)                 # declare roles + preprocessing
# train, test = cohort.split(test_size=0.2)
# fitted_train = train.fit_transform()         # statistics come from these rows only
# fitted_test = fitted_train.transform(test)   # ...and are reused for held-out rows
# ```
#
# Everything below is that, plus the reasons it is shaped that way.



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import stack
from torch.utils.data import DataLoader

from kalecancer.loaddata.tabular import TabularDataset
from kalecancer.survival.survival_target import SurvivalTarget

DATA = "data/StructuredData/clinical_data.json"
SEED = 0

CONTINUOUS = ["age_at_initial_diagnosis", "year_of_initial_diagnosis"]
CATEGORICAL = ["sex", "smoking_status", "primarily_metastasis"]


# ## 1. Declare the cohort
#
# Each feature column is given a *role* (continuous or categorical), and each role is
# given its preprocessing as plain scikit-learn transformers. There is deliberately no
# default pipeline: what you impute and scale with is a modelling decision, so it lives
# here in the script where it can be read and reported.
#
# The constructor reads the table, checks the columns exist and binds the target.
# Nothing is fitted yet.


cohort = TabularDataset(
    DATA,
    identifier="patient_id",
    target=SurvivalTarget(
        time="days_to_last_information",
        event="survival_status",
        event_value="deceased",  # every other recorded value is treated as censored
        unit="days",
    ),
    continuous=CONTINUOUS,
    continuous_transform=[SimpleImputer(strategy="median"), StandardScaler()],
    categorical=CATEGORICAL,
    categorical_transform=[
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    ],
)

print(cohort)
print()
print(cohort.describe_transforms())


# ## 2. Explore first — `.frame`
#
# The untransformed table is always available, fitted or not. This is the hatch for
# everything the dataset does not do for you: looking at distributions, checking
# missingness, deciding what the transforms above should actually be.


print("first few rows, as read:")
print(cohort.frame[["patient_id", *CONTINUOUS, *CATEGORICAL]].head().to_string(index=False))

print("\nmissing values the imputers will have to handle:")
print(cohort.frame[cohort.feature_columns].isna().sum().to_string())


# ## 3. Split
#
# Stratified on event status whenever a target is present. At a few hundred patients an
# unstratified split can easily land with a badly skewed event rate.


train, test = cohort.split(test_size=0.2, random_state=SEED)

for label, part in (("train", train), ("test", test)):
    print(f"{label:<6}{len(part):>5} patients | {part.target.summarise(part.identifiers)}")


# ## 4. Fit the training fold
#
# `fit_transform()` fits the declared transforms on the rows *this dataset holds* and
# returns a **new** dataset. The original is untouched, so folds are independent and can
# be prepared in parallel. A cross-validation fold is just
# `cohort.subset(train_idx).fit_transform()`.


fitted_train = train.fit_transform()
print(fitted_train)

print(f"\n{len(fitted_train.feature_columns)} declared columns -> {fitted_train.n_features} features:")
print("  " + ", ".join(fitted_train.feature_names))

print(f"\ntrain.is_fitted={train.is_fitted}   fitted_train.is_fitted={fitted_train.is_fitted}")


# ## 5. Held-out rows reuse the training statistics
#
# `fitted_train.transform(test)` is the only correct way to prepare held-out data. The
# proof is below: standardised age has mean exactly 0 on train by construction, and
# something *other* than 0 on test — because the test rows were centred on the training
# mean, not their own.


fitted_test = fitted_train.transform(test)
print(fitted_test)


def column(dataset, name):
    """Pull one named feature column out of a fitted dataset."""
    j = dataset.feature_names.index(name)
    return stack([dataset.get_by_id(i) for i in dataset.identifiers])[:, j]


train_mean = column(fitted_train, "age_at_initial_diagnosis").mean()
test_mean = column(fitted_test, "age_at_initial_diagnosis").mean()
print(f"\nstandardised age, mean over train: {train_mean:.4f}  (0 by construction)")
print(f"standardised age, mean over test:  {test_mean:.4f}  (offset -- test statistics were never used)")


# ## 6. Items and batches
#
# A fitted dataset is a `torch.utils.data.Dataset`. Items are dicts — features under the
# modality name, the identifier, and whatever the target contributes — which PyTorch's
# `default_collate` handles without a custom `collate_fn`.



def describe(value):
    """One-line description of an item or batch entry."""
    if isinstance(value, str):
        return f"str {value!r}"
    if isinstance(value, list):
        return f"list of {len(value)} str"
    return f"tensor {tuple(value.shape)}" + (f" = {value.item():g}" if value.ndim == 0 else "")


print("fitted_train[0]:")
for key, value in fitted_train[0].items():
    print(f"  {key:<12} {describe(value)}")

print("\none batch of 8:")
batch = next(iter(DataLoader(fitted_train, batch_size=8, shuffle=True)))
for key, value in batch.items():
    print(f"  {key:<12} {describe(value)}")


# ## 7. What it refuses to do
#
# Nothing is preprocessed unless you asked for it, and nothing is guessed. Each of these
# raises at *construction*, not halfway through training. Only the first sentence of each
# message is shown here — the full text names the fix.



def refuses(what, call):
    """Run `call`, expecting it to complain, and print the headline of the complaint."""
    try:
        call()
    except (ValueError, RuntimeError) as exc:  # NotFittedError subclasses RuntimeError
        print(f"{what}\n  {type(exc).__name__}: {str(exc).split('. ')[0]}.\n")
    else:
        print(f"{what}\n  no error raised -- unexpected\n")


target = SurvivalTarget(time="days_to_last_information", event="survival_status", event_value="deceased")

refuses("read features before fit_transform()", lambda: cohort[0])
refuses(
    "categorical columns, no categorical_transform",
    lambda: TabularDataset(DATA, identifier="patient_id", target=target, categorical=CATEGORICAL),
)
refuses(
    "a continuous column with missing values, no continuous_transform",
    lambda: TabularDataset(
        DATA, identifier="patient_id", target=target, continuous=["age_at_initial_diagnosis", "days_to_recurrence"]
    ),
)
refuses(
    "a transform shorthand -- there isn't one",
    lambda: TabularDataset(
        DATA, identifier="patient_id", target=target, continuous=CONTINUOUS, continuous_transform="auto"
    ),
)
refuses(
    "an event_value that matches nothing (the column says 'deceased')",
    lambda: TabularDataset(
        DATA,
        identifier="patient_id",
        target=SurvivalTarget(time="days_to_last_information", event="survival_status", event_value="dead"),
        continuous=["age_at_initial_diagnosis"],
    ),
)
refuses(
    "split() after fitting -- both halves would share the statistics",
    lambda: fitted_train.split(test_size=0.2),
)


# The flip side: clean numeric columns with no transform declared are simply passed
# through. There is nothing to fit, so the dataset serves values immediately.


plain = TabularDataset(DATA, identifier="patient_id", target=target, continuous=CONTINUOUS)
print(plain)
print(f"raw age of patient 001: {plain.get_by_id('001')[0]:.0f} (untouched -- nothing was asked for)")


# ## Next
#
# * `detailed_encoder_demo.py` — turning these rows into TabICL embeddings.
# * `../main.py` — the same pipeline with a survival head on top, driven by a YAML config.
