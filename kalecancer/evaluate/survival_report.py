"""Patient-level survival predictions and metric reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kalecancer.survival.metrics import survival_metrics
from kalecancer.utils.io import write_csv, write_json


@dataclass
class SplitPredictions:
    """Risk predictions for one split, in loader order."""

    split: str
    patient_ids: list[str]
    risk: torch.Tensor
    duration: torch.Tensor
    event: torch.Tensor

    def rows(self) -> list[dict]:
        return [
            {
                "patient_id": patient_id,
                "split": self.split,
                "risk_score": float(risk),
                "duration": float(duration),
                "event": int(event),
            }
            for patient_id, risk, duration, event in zip(
                self.patient_ids, self.risk, self.duration, self.event, strict=True
            )
        ]


def predict_split(model, loader: DataLoader, split: str) -> SplitPredictions:
    """Run a trained model over a loader and collect patient-level risk scores.

    Args:
        model: A trained :class:`~kalecancer.pipeline.WSISurvivalTrainer`.
        loader: Loader yielding collated bags.
        split: Name recorded against each prediction.

    Returns:
        Predictions traceable to their patient identifiers.
    """
    patient_ids: list[str] = []
    risks, durations, events = [], [], []

    for batch in loader:
        risk, _ = model.predict_risk(batch)
        patient_ids.extend(sample["group_id"] for sample in batch["samples"])
        risks.append(risk.detach().cpu())
        durations.append(batch["duration"])
        events.append(batch["event"])

    return SplitPredictions(
        split=split,
        patient_ids=patient_ids,
        risk=torch.cat(risks),
        duration=torch.cat(durations),
        event=torch.cat(events),
    )


def evaluate_predictions(
    predictions: SplitPredictions,
    train_predictions: SplitPredictions | None = None,
    eval_times: list[float] | None = None,
) -> dict:
    """Compute survival metrics for a split.

    Censoring-weighted metrics and the Brier score need the training outcomes, which
    supply a leakage-free censoring distribution and the baseline hazard.

    Args:
        predictions: Predictions for the split being scored.
        train_predictions: Predictions on the training split.
        eval_times: Horizons for the time-dependent metrics, in the unit of ``duration``.
    """
    horizons = torch.tensor(eval_times, dtype=torch.float32) if eval_times else None
    return survival_metrics(
        predictions.risk,
        predictions.event,
        predictions.duration,
        train_risk=None if train_predictions is None else train_predictions.risk,
        train_event=None if train_predictions is None else train_predictions.event,
        train_time=None if train_predictions is None else train_predictions.duration,
        eval_times=horizons,
    )


def summarise_folds(fold_metrics: list[dict], split: str = "test") -> dict:
    """Aggregate cross-validation folds into mean and standard deviation.

    Only scalar metrics are aggregated; nested time-dependent entries are left to the
    per-fold records.

    Args:
        fold_metrics: One ``{split: metrics}`` mapping per fold.
        split: Which split to summarise.

    Returns:
        ``{metric: {"mean": ..., "std": ..., "folds": [...]}}``.
    """
    values: dict[str, list[float]] = {}
    for fold in fold_metrics:
        for name, value in fold.get(split, {}).items():
            if isinstance(value, (int, float)):
                values.setdefault(name, []).append(float(value))

    summary = {}
    for name, scores in values.items():
        scores_tensor = torch.tensor(scores)
        summary[name] = {
            "mean": float(scores_tensor.mean()),
            "std": float(scores_tensor.std(unbiased=len(scores) > 1)),
            "folds": scores,
        }
    return summary


def save_survival_report(
    out_dir: str | Path,
    predictions: list[SplitPredictions],
    metrics: dict[str, dict],
) -> tuple[Path, Path]:
    """Write ``predictions.csv`` and ``metrics.json``.

    Returns:
        Paths to the written files.
    """
    out_dir = Path(out_dir)
    rows = [row for split in predictions for row in split.rows()]
    return (
        write_csv(out_dir / "predictions.csv", rows),
        write_json(out_dir / "metrics.json", metrics),
    )
