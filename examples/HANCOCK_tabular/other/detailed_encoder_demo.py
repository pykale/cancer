"""TabICL row embeddings for the HANCOCK clinical table.

Run from the repo root: python examples/HANCOCK_tabular/encoder_demo.py
Needs the optional extra: pip install 'kalecancer[tabular]'

Covers the ordinary path -- split, fit the fold, encode -- and then measures the
two properties the unit tests cannot, because they are properties of the
pretrained weights rather than of the wrapper:

* the context is fitted state (a different context moves the embeddings);
* the context does not see what is encoded against it (no leak).

Both print a number. If the first ever reads 0.0000, the encoder has stopped
conditioning on its context; if the second ever reads anything but 0.00000000,
held-out rows are reaching the representations of the rows used to train on.
"""

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kalecancer.loaddata.base import NotFittedError
from kalecancer.loaddata.tabular import TabularDataset
from kalecancer.model.embed import TabICLEncoder
from kalecancer.survival.survival_target import SurvivalTarget

SEED = 0


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------- 1. load
rule("1. load and split -- nothing here is new")

cohort = TabularDataset(
    "data/clinical_data.json",
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

train, test = cohort.split(test_size=0.2, random_state=SEED)
fitted_train = train.fit_transform()
fitted_test = fitted_train.transform(test)
print("train:", fitted_train)
print("test: ", fitted_test)


# ------------------------------------------------------------- 2. guards
rule("2. the encoder refuses to guess")

unfitted = TabICLEncoder()
print("repr:", unfitted)

try:
    unfitted.encode(fitted_test)
except NotFittedError as exc:
    print(f"\nencode() before fit() -> NotFittedError:\n  {exc}")

try:
    TabICLEncoder().fit(train)  # dataset transforms not fitted
except NotFittedError as exc:
    print(f"\nfit() on an unfitted dataset -> NotFittedError:\n  {str(exc)[:110]}...")


# ------------------------------------------------------- 3. fit and encode
rule("3. fit the context on the training fold, then encode")

encoder = TabICLEncoder(random_state=SEED).fit(fitted_train)
print("repr:", encoder)
print("out_dim:", encoder.out_dim)

Z_train = encoder.encode(fitted_train)
Z_test = encoder.encode(fitted_test)
print(f"\nZ_train {tuple(Z_train.shape)}")
print(f"Z_test  {tuple(Z_test.shape)}")
print(f"\nfit() returned a new object, the original is untouched: {not unfitted.is_fitted}")


rule("4. the context is fold state, exactly like a scaler's mean")

half = fitted_train.subset(range(len(fitted_train) // 2))
encoder_half = TabICLEncoder(random_state=SEED).fit(half)
Z_test_half = encoder_half.encode(fitted_test)

drift = (Z_test - Z_test_half).abs().mean().item()
print(f"same {len(fitted_test)} test rows, encoded against two different contexts")
print(f"  context = {len(fitted_train)} rows  vs  context = {len(half)} rows")
print(f"  mean |difference| per dimension: {drift:.4f}")
print(
    "\nThis is why the context must come from the training fold. Fit it on the whole\n"
    "cohort and every embedding carries information from the held-out rows -- with no\n"
    "error, and no help from correct ColumnTransformer handling."
)


# --------------------------------------------- 5. the context does not leak
rule("5. ...but the context itself is not disturbed by what is encoded")

Z_ctx_all = encoder.encode(fitted_test)
Z_ctx_few = encoder.encode(fitted_test.subset(range(10)))

leak = (Z_ctx_all[:10] - Z_ctx_few).abs().mean().item()
print("same encoder, same 10 rows, encoded alongside 153 others vs alone")
print(f"  mean |difference| per dimension: {leak:.8f}")
print(
    "\nExactly zero, because TabICL's inducing points attend only to context rows\n"
    "(embed_with_test=False, hard-coded). Encoding is therefore batch-invariant:\n"
    "you get the same answer however you chunk the calls."
)
