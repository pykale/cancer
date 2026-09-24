"""HANCOCK: late fusion of a clinical (TabICL) branch and a WSI (attention MIL) branch, classifying survival status.

The target follows the HANCOCK paper: deceased vs living, excluding deaths not related to the tumour. Both branches
train jointly on the sum of their losses; predictions are combined by averaging logits.

    uv run python examples/hancock/classification_late.py
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

from kalecancer.evaluate import AUROC, BalancedAccuracy
from kalecancer.loaddata import Classification, MultimodalDataset, PatchFeatures
from kalecancer.model import ABMIL, ClassificationHead, LateFusion, MeanLogits, TabICLEncoder, Unimodal
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
    """The published training/test assignment, restricted to the patients the dataset includes."""
    path = DATA / "DataSplits_DataDictionaries" / f"dataset_split_{name}.json"
    split = pd.read_json(path, dtype={"patient_id": str}).set_index("patient_id")["dataset"]
    split = split[split.index.isin(data.ids)]
    return data.subset(split.index[split == "training"]), data.subset(split.index[split == "test"])


def build_dataset() -> MultimodalDataset:
    table = clinical()
    keep = table["survival_status_with_cause"] != "deceased not tumor specific"
    return MultimodalDataset(
        modalities={"clinical": table[BASELINE_COLUMNS], "wsi": primary_tumour_features()},
        target=Classification(labels=table.loc[keep, "survival_status"], classes=["living", "deceased"]),
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


def build_model() -> LateFusion:
    return LateFusion(
        branches={
            "clinical": Unimodal(
                modality="clinical",
                encoding=[
                    TabICLEncoder(
                        checkpoint="tabicl-classifier-v2-20260212.ckpt",
                        output="row",
                        trainable=[],
                        context_label="label",
                        context_folds=5,
                        random_state=0,
                    ),
                    torch.nn.Linear(512, 64),
                ],
                head=ClassificationHead(in_dim=64, n_classes=2),
            ),
            "wsi": Unimodal(
                modality="wsi",
                encoding=[
                    ABMIL(in_dim=1024, hidden_dim=256, attention_dim=128, dropout=0.25),
                    torch.nn.Linear(256, 64),
                ],
                head=ClassificationHead(in_dim=64, n_classes=2),
            ),
        },
        combine=MeanLogits(),
    )


def build_pipeline() -> Pipeline:
    return Pipeline(
        model=build_model(),
        transforms={"clinical": clinical_transform()},
        optimizer=partial(torch.optim.AdamW, lr=3e-4, weight_decay=1e-2),
        batch_size=32,
        drop_last=True,
        max_epochs=30,
        validation=StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0),
        early_stopping=EarlyStopping(metric=AUROC(positive_class="deceased"), patience=5, restore_best=True),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="32-true",
        random_state=0,
    )


def main() -> None:
    data = build_dataset()
    print(data.summary())
    train, test = official_split(data, SPLIT_NAME)
    pipe = build_pipeline().fit(train)
    metrics = {"auroc": AUROC(positive_class="deceased"), "balanced_accuracy": BalancedAccuracy()}
    print("fused:", pipe.evaluate(test, metrics=metrics).to_dict())
    for branch in ("clinical", "wsi"):
        print(f"{branch} branch:", pipe.evaluate(test, metrics=metrics, branch=branch).to_dict())


if __name__ == "__main__":
    main()
