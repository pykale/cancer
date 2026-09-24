"""HANCOCK: clinical table (TabICL) + primary-tumour UNI patch features (attention MIL), intermediate fusion, Cox.

Trains on the chosen split and reports test concordance. With about 40 test events on the "in" split, expect
Harrell's C of roughly 0.60-0.80.

For a walkthrough with output already captured, see survival_intermediate.ipynb.

    uv run python examples/hancock/survival_intermediate.py
"""

from functools import partial
from pathlib import Path

import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from kalecancer.evaluate import HarrellC, UnoC
from kalecancer.loaddata import MultimodalDataset, PatchFeatures, TimeToEvent
from kalecancer.model import ABMIL, Concat, CoxHead, IntermediateFusion, TabICLEncoder
from kalecancer.pipeline import EarlyStopping, Pipeline

DATA = Path(__file__).resolve().parents[2] / "data" / "hancock"

BASELINE_COLUMNS = ["age_at_initial_diagnosis", "sex", "smoking_status", "primarily_metastasis"]

SPLIT_NAME = "in"  # "in", "out", "Oropharynx".


def clinical() -> pd.DataFrame:
    # patient ids are zero-padded strings; without dtype pandas reads "001" as the integer 1
    return pd.read_json(DATA / "StructuredData" / "clinical_data.json", dtype={"patient_id": str}).set_index(
        "patient_id"
    )


def primary_tumour_features() -> PatchFeatures:
    """UNI patch features of the primary-tumour slides. Eight patients have a second slide (suffix _a)."""
    return PatchFeatures.from_glob(
        str(DATA / "WSI_PrimaryTumor_UNI_Encodings" / "*" / "h5_files" / "*.h5"),
        id_pattern=r"PrimaryTumor_HE_(\d{3})(?:_[a-z])?\.h5$",
        multiple_files="concatenate",
    )


def official_split(data: MultimodalDataset, name: str) -> tuple[MultimodalDataset, MultimodalDataset]:
    """The published training/test assignment, restricted to the patients the dataset includes.

    Names: "in", "out", "Oropharynx". The "treatment_outcome" test set is selected on outcome (76% events), so it
    is not a fair test set for overall survival.
    """
    path = DATA / "DataSplits_DataDictionaries" / f"dataset_split_{name}.json"
    split = pd.read_json(path, dtype={"patient_id": str}).set_index("patient_id")["dataset"]
    split = split[split.index.isin(data.ids)]
    return data.subset(split.index[split == "training"]), data.subset(split.index[split == "test"])


def build_dataset() -> MultimodalDataset:
    table = clinical()
    return MultimodalDataset(
        modalities={"clinical": table[BASELINE_COLUMNS], "wsi": primary_tumour_features()},
        target=TimeToEvent(
            time=table["days_to_last_information"],
            event=table["survival_status"].map({"deceased": True, "living": False}),
        ),
        required_modalities=["clinical", "wsi"],
    )


def clinical_transform() -> ColumnTransformer:
    categorical = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    )
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), ["age_at_initial_diagnosis"]),
            ("categorical", categorical, ["sex", "smoking_status", "primarily_metastasis"]),
        ]
    )


def build_pipeline() -> Pipeline:
    model = IntermediateFusion(
        encoding={
            "clinical": [
                TabICLEncoder(
                    checkpoint="tabicl-classifier-v2-20260212.ckpt",
                    output="row",
                    trainable=[],
                    context_label="event",
                    context_folds=5,
                    random_state=0,
                ),
                torch.nn.Linear(512, 64),
            ],
            "wsi": [
                ABMIL(in_dim=1024, hidden_dim=256, attention_dim=128, dropout=0.25),
                torch.nn.Linear(256, 64),
            ],
        },
        fusion=Concat(),
        head=CoxHead(in_dim=128, ties="efron"),
    )
    return Pipeline(
        model=model,
        transforms={"clinical": clinical_transform()},
        optimizer=partial(torch.optim.AdamW, lr=3e-4, weight_decay=1e-2),
        batch_size=32,
        drop_last=True,
        max_epochs=30,
        validation=StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0),
        early_stopping=EarlyStopping(metric=HarrellC(), patience=5, restore_best=True),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="32-true",
        random_state=0,
    )


def main() -> None:
    data = build_dataset()
    print(data.summary())
    train, test = official_split(data, SPLIT_NAME)
    pipe = build_pipeline().fit(train)
    print(pipe.history_)
    print(pipe.fit_report_)
    print(pipe.evaluate(test, metrics={"harrell_c": HarrellC(), "uno_c_5y": UnoC(tau=1825.0)}))


if __name__ == "__main__":
    main()
