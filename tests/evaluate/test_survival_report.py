"""Tests for patient-level prediction reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from kalecancer.evaluate import (
    SplitPredictions,
    evaluate_predictions,
    predict_split,
    save_survival_report,
    summarise_folds,
)
from kalecancer.loaddata import WSIFeatureDataset, collate_bags
from kalecancer.pipeline import WSISurvivalTrainer
from tests.conftest import FEATURE_DIM


def make_predictions(split: str = "test", n: int = 12, seed: int = 0) -> SplitPredictions:
    """Predictions whose follow-up comfortably spans the horizons used below."""
    generator = torch.Generator().manual_seed(seed)
    return SplitPredictions(
        split=split,
        patient_ids=[f"{i:03d}" for i in range(n)],
        risk=torch.randn(n, generator=generator),
        duration=torch.rand(n, generator=generator) * 1800 + 100,
        # A censored majority keeps the censoring distribution defined at every horizon.
        event=(torch.rand(n, generator=generator) < 0.3).int(),
    )


def test_rows_are_traceable_to_their_patient() -> None:
    predictions = make_predictions()

    rows = predictions.rows()

    assert len(rows) == 12
    assert [row["patient_id"] for row in rows] == predictions.patient_ids
    assert set(rows[0]) == {"patient_id", "split", "risk_score", "duration", "event"}


def test_predict_split_returns_one_risk_per_patient(cohort) -> None:
    dataset = WSIFeatureDataset(cohort, expected_dim=FEATURE_DIM)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_bags, num_workers=0)
    model = WSISurvivalTrainer(input_dim=FEATURE_DIM, hidden_dim=8, attention_dim=4)

    predictions = predict_split(model, loader, "test")

    assert predictions.patient_ids == ["001", "002", "003"]
    assert predictions.risk.shape == (3,)
    assert predictions.duration.tolist() == [500.0, 1200.0, 900.0]


def test_evaluate_reports_harrell_without_training_predictions() -> None:
    metrics = evaluate_predictions(make_predictions())

    assert "c_index" in metrics
    assert "c_index_ipcw" not in metrics


def test_evaluate_adds_censoring_aware_metrics_with_training_predictions() -> None:
    metrics = evaluate_predictions(
        make_predictions(n=120, seed=1), make_predictions("train", n=200, seed=2), [365.0, 1095.0]
    )

    assert {"c_index", "auc", "mean_auc", "integrated_brier_score"} <= set(metrics)
    assert set(metrics["auc"]) == {"365", "1095"}


def test_evaluate_records_why_censoring_weighted_metrics_were_unavailable() -> None:
    """A cohort too small to support IPCW must not abort the run."""
    metrics = evaluate_predictions(make_predictions(n=6), make_predictions("train", n=6), [365.0])

    assert "c_index" in metrics
    assert "censoring_weighted_metrics_error" in metrics


def test_summarise_folds_reports_mean_and_spread() -> None:
    folds = [
        {"test": {"c_index": 0.6, "num_patients": 100.0}},
        {"test": {"c_index": 0.8, "num_patients": 100.0}},
    ]

    summary = summarise_folds(folds)

    assert summary["c_index"]["mean"] == pytest.approx(0.7)
    assert summary["c_index"]["std"] == pytest.approx(0.1414, abs=1e-3)
    assert summary["c_index"]["folds"] == [0.6, 0.8]


def test_summarise_folds_skips_nested_time_dependent_metrics() -> None:
    folds = [{"test": {"c_index": 0.6, "auc": {"365": 0.7}}}]

    summary = summarise_folds(folds)

    assert set(summary) == {"c_index"}


def test_report_writes_predictions_and_metrics(tmp_path: Path) -> None:
    predictions = [make_predictions("train", 10), make_predictions("test", 6)]
    metrics = {"test": {"c_index": 0.6}}

    predictions_path, metrics_path = save_survival_report(tmp_path, predictions, metrics)

    rows = list(csv.DictReader(predictions_path.open(encoding="utf-8")))
    assert len(rows) == 16
    assert {row["split"] for row in rows} == {"train", "test"}
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["test"]["c_index"] == 0.6
