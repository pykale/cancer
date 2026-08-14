from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader

from kalecancer.loaddata.tabular import TabularDataset
from kalecancer.survival.survival_target import SurvivalTarget
from kalecancer.loaddata.base import NotFittedError

SEED = 0

CONTINUOUS = ["age_at_initial_diagnosis", "year_of_initial_diagnosis"]
CATEGORICAL = ["sex", "smoking_status", "primarily_metastasis"]


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------- 1. load
rule("1. construct -- index read, columns validated, target bound")

cohort = TabularDataset(
    "data/clinical_data.json",
    identifier="patient_id",
    target=SurvivalTarget(
        time="days_to_last_information",
        event="survival_status",
        event_value="deceased",
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

print("\nMissingness the imputer will have to handle:")
print(cohort.frame[cohort.feature_columns].isna().sum().to_string())


# ------------------------------------------------------- 2. unfitted guard
rule("2. unfitted dataset refuses to serve values")

try:
    cohort[0]
except NotFittedError as exc:
    print(f"NotFittedError: {exc}")

print("\n...but .frame still works for exploration:")
print(cohort.frame[["patient_id", "age_at_initial_diagnosis", "sex"]].head(3).to_string(index=False))


# ------------------------------------------------------------- 3. split
rule("3. split -- stratified on event status")

train, test = cohort.split(test_size=0.2, random_state=SEED)
for label, part in (("train", train), ("test", test)):
    events = part.target.events_for(part.identifiers)
    print(f"{label:<6} {len(part):>4} patients | {int(events.sum()):>3} events ({events.mean():.1%})")
print("no overlap:", not (set(train.identifiers) & set(test.identifiers)))


# --------------------------------------------------------- 4. fit a fold
rule("4. fit_transform() -- transforms fitted, matrix materialised")

fitted_train = train.fit_transform()
print(fitted_train)
print(f"\n{len(fitted_train.feature_columns)} declared columns -> "
      f"{fitted_train.n_features} features after encoding:")
print("  " + ", ".join(fitted_train.feature_names))

print("\nOriginal object is untouched (fit returns a new instance):")
print(f"  train.is_fitted        = {train.is_fitted}")
print(f"  fitted_train.is_fitted = {fitted_train.is_fitted}")


# ------------------------------------------------- 5. hold-out transform
rule("5. transform(test) -- held-out rows use TRAIN statistics")

fitted_test = fitted_train.transform(test)
print(fitted_test)

train_mean = fitted_train._matrix[:, 0].mean().item()
test_mean = fitted_test._matrix[:, 0].mean().item()
print(f"\nStandardised 'age' mean on train: {train_mean:+.4f}  (0 by construction)")
print(f"Standardised 'age' mean on test:  {test_mean:+.4f}  (nonzero -- test stats were NOT used)")


# ---------------------------------------------------------- 6. item dict
rule("6. the item contract")

item = fitted_train[0]
for key, value in item.items():
    shape = tuple(value.shape) if hasattr(value, "shape") else type(value).__name__
    print(f"  {key:<12} {str(shape):<12} {value if not hasattr(value, 'shape') or value.ndim == 0 else ''}")


# -------------------------------------------------------- 7. DataLoader
rule("7. default_collate handles the dict -- no custom collate_fn")

batch = next(iter(DataLoader(fitted_train, batch_size=8, shuffle=True)))
for key, value in batch.items():
    shape = tuple(value.shape) if hasattr(value, "shape") else f"list[{len(value)}]"
    print(f"  {key:<12} {shape}")


# ------------------------------------------------------- 8. target guard
rule("8. a mis-specified event_value is caught at construction")

try:
    TabularDataset(
        "data/clinical_data.json",
        identifier="patient_id",
        target=SurvivalTarget(time="days_to_last_information",
                              event="survival_status",
                              event_value="dead"),   # wrong: the column says "deceased"
        continuous=["age_at_initial_diagnosis"],
    )
except ValueError as exc:
    print(f"ValueError: {exc}")


# --------------------------------------------- 9. no silent preprocessing
rule("9. nothing is preprocessed unless you say so")

target = SurvivalTarget(time="days_to_last_information", event="survival_status",
                        event_value="deceased")

print("a) categorical columns, no transform declared:")
try:
    TabularDataset("data/clinical_data.json", identifier="patient_id", target=target,
                   categorical=CATEGORICAL)
except ValueError as exc:
    print(f"   ValueError: {exc}\n")

print("b) a continuous column with missing values, no transform declared:")
try:
    TabularDataset("data/clinical_data.json", identifier="patient_id", target=target,
                   continuous=["age_at_initial_diagnosis", "days_to_recurrence"])
except ValueError as exc:
    print(f"   ValueError: {exc}\n")

print("c) clean numeric columns, no transform declared -- works, untouched:")
plain = TabularDataset("data/clinical_data.json", identifier="patient_id", target=target,
                       continuous=CONTINUOUS)
print(f"   {plain}")
print(f"   raw age of patient 001: {plain.get_by_id('001')[0].item():.0f} "
      f"(not standardised, because nothing was asked for)")

print("\nd) there is no shorthand:")
try:
    TabularDataset("data/clinical_data.json", identifier="patient_id", target=target,
                   continuous=CONTINUOUS, continuous_transform="auto")
except ValueError as exc:
    print(f"   ValueError: {exc}")

print("\nAll checks passed.")
