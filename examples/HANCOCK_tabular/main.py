from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kalecancer.loaddata.tabular import TabularDataset
from kalecancer.model.embed import TabICLEncoder
from kalecancer.survival.survival_target import SurvivalTarget

SEED = 0

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

train, test = cohort.split(test_size=0.2, random_state=SEED)

train_transformed = train.fit_transform()
test_transformed = train_transformed.transform(test)

encoder = TabICLEncoder(random_state=SEED)

encoder = encoder.fit(train_transformed)

encoded_train = encoder.encode(train_transformed)
encoded_test = encoder.encode(test_transformed)

print(f"encoded_train.shape: {encoded_train.shape}")
print(f"encoded_test.shape: {encoded_test.shape}")
