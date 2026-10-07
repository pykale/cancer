"""HANCOCK: the survival pipeline of survival_intermediate.py, loaded from configs/intermediate_cox.yaml instead of
built in Python.

    uv run python examples/hancock/from_config.py
"""

from pathlib import Path

import pandas as pd

from kalecancer.evaluate import HarrellC
from kalecancer.loaddata import MultimodalDataset, PatchFeatures, TimeToEvent, Tabular
from kalecancer.pipeline import load_pipeline

DATA = Path(__file__).resolve().parents[2] / "data" / "hancock"
CONFIG = Path(__file__).resolve().parent / "configs" / "intermediate_cox.yaml"

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
    return MultimodalDataset(
        modalities={"clinical": Tabular(table[BASELINE_COLUMNS]), "wsi": primary_tumour_features()},
        target=TimeToEvent(
            time=table["days_to_last_information"],
            event=table["survival_status"].map({"deceased": True, "living": False}),
        ),
        required_modalities=["clinical", "wsi"],
    )


def main() -> None:
    train, test = official_split(build_dataset(), SPLIT_NAME)
    pipe = load_pipeline(CONFIG).fit(train)
    print(pipe.evaluate(test, metrics={"harrell_c": HarrellC()}))


if __name__ == "__main__":
    main()
