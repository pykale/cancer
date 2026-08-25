"""Quickstart for the cohort/preprocessor/view API, on the HANCOCK clinical table.

See ``pipeline.py`` for the same thing driven from a YAML config.

Run with::

    uv run python examples/HANCOCK_tabular/quickstart.py
"""
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kalecancer.loaddata import TabularCohort
from kalecancer.model.embed import TabICLEmbedder
from kalecancer.survival import SurvivalTarget

SEED = 0
N_FOLDS = 5
DATA_LOC = "data/HANCOCK/raw/StructuredData/clinical_data.json"

target = SurvivalTarget(
    time="days_to_last_information",
    event="survival_status",
    event_value="deceased",
    unit="days",
)

cohort = TabularCohort(
    source=DATA_LOC,
    identifier="patient_id",
    target=target,
    continuous=["age_at_initial_diagnosis", "year_of_initial_diagnosis"],
    continuous_transform=[SimpleImputer(strategy="median"), StandardScaler()],
    categorical=["sex", "smoking_status", "primarily_metastasis"],
    categorical_transform=[
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    ],
)
print(cohort)

dev_ids, test_ids = cohort.split(test_size=0.2, random_state=SEED, stratify=True)
print(f"dev {len(dev_ids)} patients | test {len(test_ids)} patients")

# Each patient is embedded against a context that excludes them. Not merely fold
# hygiene: TabICL's context rows see their own label, so a patient embedded against a
# context containing them is optimistic in a way a held-out patient never is. Embedding
# every fold's validation rows is what makes these 610 vectors comparable to each other.
#
# The ids are carried alongside because torch.cat below returns them in *fold* order,
# not dev_ids order -- same patients, different sequence.
embeddings, embedded_ids = [], []
labels = target.values_for(dev_ids)["event"]
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for fold, (train_pos, val_pos) in enumerate(folds.split(dev_ids, labels), start=1):
    train_ids = [dev_ids[i] for i in train_pos]
    val_ids = [dev_ids[i] for i in val_pos]

    transform = cohort.fit_preprocessor(train_ids)
    train = cohort.view(train_ids, transform)
    val = cohort.view(val_ids, transform)
    assert set(val.identifiers).isdisjoint(transform.fitted_on)

    context = train.batch()
    embedder = TabICLEmbedder(
        context_x=context.modalities["clinical"],
        context_y=context.target["event"],
        trainable=False,
        random_state=SEED,
    )

    with torch.no_grad():
        encoded = embedder(val.batch().modalities["clinical"])
    embeddings.append(encoded)
    embedded_ids.extend(val.identifiers)
    print(f"fold {fold}: {embedder} -> encoded {tuple(encoded.shape)}")

all_embeddings = torch.cat(embeddings)
print(f"\nembeddings {tuple(all_embeddings.shape)} | mean {all_embeddings.mean():.4f}")

# What the ids are for. Row i of all_embeddings is patient embedded_ids[i], so
# supervision is fetched by name; pairing the block with dev_ids instead would put
# every patient against someone else's outcome, and would run without complaint.
targets = target.values_for(embedded_ids)
assert embedded_ids != dev_ids and sorted(embedded_ids) == sorted(dev_ids)
print(f"aligned to {len(embedded_ids)} patients | {int(targets['event'].sum())} events")
