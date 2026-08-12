"""Synthetic end-to-end smoke test of the WSI survival pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from kalecancer.evaluate import evaluate_predictions, predict_split, save_survival_report
from kalecancer.interpret import export_attention
from kalecancer.loaddata import WSIFeatureBagDataset, build_cohort, collate_bags, split_patients
from kalecancer.pipeline import WSISurvivalTrainer
from kalecancer.utils import set_seed
from tests.conftest import FEATURE_DIM, write_bag

NUM_PATIENTS = 24


@pytest.fixture
def synthetic_cohort(tmp_path: Path) -> tuple[Path, Path]:
    """A cohort large enough to split three ways and to contain events per batch."""
    feature_root = tmp_path / "features"
    records = []
    for index in range(NUM_PATIENTS):
        patient_id = f"{index:03d}"
        write_bag(
            feature_root / "site" / "h5_files" / f"PrimaryTumor_HE_{patient_id}.h5",
            num_patches=6 + index % 5,
            seed=index,
        )
        records.append(
            {
                "patient_id": patient_id,
                "days_to_last_information": 100 + index * 40,
                "survival_status": "deceased" if index % 2 == 0 else "living",
            }
        )

    clinical_path = tmp_path / "clinical.json"
    clinical_path.write_text(json.dumps(records), encoding="utf-8")
    return feature_root, clinical_path


def test_pipeline_trains_evaluates_and_exports_attention(synthetic_cohort, tmp_path: Path) -> None:
    feature_root, clinical_path = synthetic_cohort
    set_seed(0)
    out_dir = tmp_path / "outputs"

    bags, summary = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)
    assert summary.num_matched_patients == NUM_PATIENTS

    split = split_patients(bags, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=0)

    def loader(patient_bags, shuffle=False, max_patches=None):
        dataset = WSIFeatureBagDataset(patient_bags, expected_dim=FEATURE_DIM, max_patches=max_patches)
        return DataLoader(dataset, batch_size=8, shuffle=shuffle, collate_fn=collate_bags, num_workers=0)

    model = WSISurvivalTrainer(input_dim=FEATURE_DIM, hidden_dim=8, attention_dim=4, max_epochs=2)
    trainer = pl.Trainer(
        max_epochs=2, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False
    )
    trainer.fit(model, loader(split.train, shuffle=True, max_patches=4), loader(split.val))

    predictions = [
        predict_split(model, loader(split.train), "train"),
        predict_split(model, loader(split.test), "test"),
    ]
    metrics = {
        prediction.split: evaluate_predictions(prediction, predictions[0], [365.0, 1095.0])
        for prediction in predictions
    }

    predictions_path, metrics_path = save_survival_report(out_dir, predictions, metrics)
    attention_dir = export_attention(model, loader(split.test), out_dir / "attention", top_k=3)

    assert predictions_path.exists() and metrics_path.exists()
    assert 0.0 <= metrics["test"]["c_index"] <= 1.0

    # Every patient is traceable in the predictions file.
    prediction_ids = {row["patient_id"] for split_prediction in predictions for row in split_prediction.rows()}
    assert prediction_ids == {bag.patient_id for bag in split.train + split.test}

    # Attention is exported per test patient, one row per patch of the full bag.
    dataset = WSIFeatureBagDataset(split.test, expected_dim=FEATURE_DIM)
    for index, bag in enumerate(split.test):
        exported = (attention_dir / f"{bag.patient_id}.csv").read_text(encoding="utf-8").strip().splitlines()
        assert exported[0] == "patient_id,slide_id,x,y,attention"
        assert len(exported) - 1 == dataset[index].features.shape[0]
    assert (attention_dir / "top_patches.csv").exists()


def test_risk_scores_are_one_per_patient(synthetic_cohort) -> None:
    feature_root, clinical_path = synthetic_cohort
    bags, _ = build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)
    dataset = WSIFeatureBagDataset(bags, expected_dim=FEATURE_DIM)
    loader = DataLoader(dataset, batch_size=5, collate_fn=collate_bags, num_workers=0)

    model = WSISurvivalTrainer(input_dim=FEATURE_DIM, hidden_dim=8, attention_dim=4)
    batch = next(iter(loader))
    risk, attentions = model.predict_risk(batch)

    assert risk.shape == (len(batch),)
    assert [len(attention) for attention in attentions] == [sample.features.shape[0] for sample in batch.samples]
