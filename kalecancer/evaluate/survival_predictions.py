"""Collecting survival predictions from a model, and scoring them.

The counterpart of :mod:`~kalecancer.evaluate.survival_metrics`, which scores arrays:
this runs a trained model over a loader first, keeping each risk score attached to
the patient it belongs to, so a prediction stays traceable from the model to the
written file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kalecancer.evaluate.survival_metrics import (
    breslow_baseline_hazard,
    concordance_index,
    integrated_brier,
    predict_survival_function,
    time_dependent_auc,
    usable_eval_times,
)
from kalecancer.utils.io import write_csv, write_json

logger = logging.getLogger(__name__)


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
        model: A trained :class:`~kalecancer.pipeline.CohortTrainer`.
        loader: Loader yielding :class:`~kalecancer.loaddata.multimodal_access.PatientBatch`.
        split: Name recorded against each prediction.

    Returns:
        Predictions traceable to their patient identifiers.
    """
    patient_ids: list[str] = []
    risks, durations, events = [], [], []

    for batch in loader:
        risk = model.predict(batch).prediction.reshape(-1)
        patient_ids.extend(batch.patient_id)
        risks.append(risk.detach().cpu())
        durations.append(batch.target["time"])
        events.append(batch.target["event"])

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

    Harrell's C-index is always reported. The censoring-weighted metrics additionally
    need the training outcomes, which supply a leakage-free estimate of the censoring
    distribution.

    Args:
        predictions: Predictions for the split being scored.
        train_predictions: Predictions on the training split.
        eval_times: Horizons for the time-dependent metrics, in the unit of ``duration``.

    Returns:
        Metric names mapped to values; time-dependent entries are keyed by horizon.
    """
    risk = predictions.risk.numpy().astype(float)
    times = predictions.duration.numpy().astype(float)
    events = predictions.event.numpy().astype(bool)

    metrics: dict = {"num_patients": float(len(risk)), "num_events": float(events.sum())}
    try:
        metrics["c_index"] = concordance_index(risk, times, events)
    except ValueError as error:
        # A split with no observed event, or none preceding another patient, has no
        # orderable pair. Reporting why beats aborting a finished training run.
        logger.warning("concordance index unavailable: %s", error)
        metrics["c_index_error"] = str(error)

    if train_predictions is None or not eval_times:
        return metrics

    train_times = train_predictions.duration.numpy().astype(float)
    train_events = train_predictions.event.numpy().astype(bool)

    # Horizons must sit inside both follow-ups: the training one supplies the IPCW
    # weights, the test one is what the metrics are computed over.
    horizons = usable_eval_times(np.asarray(eval_times, dtype=float), times)
    horizons = usable_eval_times(horizons, train_times)
    if horizons.size < 2:
        metrics["censoring_weighted_metrics_error"] = (
            f"fewer than two of {list(eval_times)} fall inside the observed follow-up"
        )
        return metrics

    # IPCW weighting needs a censoring distribution that supports every horizon;
    # too few censored subjects leaves it undefined, which must not abort a run.
    try:
        auc, mean_auc = time_dependent_auc(train_times, train_events, times, events, risk, horizons)
        metrics["auc"] = {f"{time:.0f}": float(value) for time, value in zip(horizons, auc, strict=True)}
        metrics["mean_auc"] = float(mean_auc)

        # The baseline hazard is fitted on the training split only; fitting it on the
        # split being scored would leak its event times into its own evaluation.
        event_times, cumulative_hazard = breslow_baseline_hazard(
            train_predictions.risk.numpy().astype(float), train_times, train_events
        )
        survival_probs = predict_survival_function(risk, event_times, cumulative_hazard, horizons)
        metrics["integrated_brier_score"] = float(
            integrated_brier(train_times, train_events, times, events, survival_probs, horizons)
        )
    except ValueError as error:
        logger.warning("censoring-weighted metrics unavailable: %s", error)
        metrics["censoring_weighted_metrics_error"] = str(error)
    return metrics


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
            if isinstance(value, int | float):
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


def save_predictions(
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
