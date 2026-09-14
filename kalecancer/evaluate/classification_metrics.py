"""Metrics for binary endpoints such as recurrence or vital status.

These live here rather than in ``kalecancer/survival/`` for the same reason the
time-dependent metrics do: they lean on scikit-learn rather than reimplementing
established machinery, and that package is quarantined from such dependencies (see
its docstring).

Scores may be logits or probabilities throughout. Ranking metrics are invariant to
the sigmoid, so only the threshold metrics apply it.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


class MetricError(ValueError):
    """Raised when a metric cannot be computed from the values given."""


def _as_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    present = np.unique(labels)
    if present.size < 2:
        raise MetricError(
            f"a binary metric needs both classes, but every subject has label {present.tolist()}; "
            "this usually means the split is too small or the endpoint filter removed one class"
        )
    return labels.astype(int)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve.

    Args:
        labels: ``(N,)`` targets, 1 positive and 0 negative.
        scores: ``(N,)`` predicted scores, higher meaning more likely positive.

    Returns:
        The area, 0.5 for a random ranking and 1.0 for a perfect one.

    Raises:
        MetricError: If only one class is present.
    """
    return float(roc_auc_score(_as_labels(labels), np.asarray(scores).reshape(-1)))


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    """Summarise a binary prediction.

    ``roc_auc`` and ``average_precision`` rank the scores and ignore ``threshold``;
    the remaining metrics apply a sigmoid and cut at it.

    Args:
        labels: ``(N,)`` targets, 1 positive and 0 negative.
        scores: ``(N,)`` logits or probabilities.
        threshold: Probability above which a subject is called positive.

    Returns:
        Counts and metrics, ready for JSON export.

    Raises:
        MetricError: If only one class is present.
    """
    labels = _as_labels(labels)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    probabilities = scores if scores.min() >= 0.0 and scores.max() <= 1.0 else 1.0 / (1.0 + np.exp(-scores))
    predicted = (probabilities >= threshold).astype(int)

    return {
        "n": int(labels.size),
        "n_positive": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
    }


def mean_roc_curve(labels_per_run: list, scores_per_run: list, grid_size: int = 100) -> dict:
    """Average several runs' ROC curves onto one false-positive-rate grid.

    Runs differ in how many distinct thresholds they produce, so their curves cannot
    be averaged point by point. Each is interpolated onto a shared grid first, which
    is what makes a mean curve and a variability band well defined.

    Note this is not the same as pooling every run's predictions and computing one
    curve: averaging interpolated curves weights each run equally regardless of its
    size, and is the convention used when repeats of the same experiment are
    summarised together.

    Args:
        labels_per_run: One ``(N,)`` label array per run.
        scores_per_run: One ``(N,)`` score array per run, aligned with the labels.
        grid_size: Number of false-positive-rate points to interpolate onto.

    Returns:
        A dict with ``"fpr"``, ``"tpr_mean"``, ``"tpr_std"``, ``"tpr_runs"``,
        ``"auc_mean"``, ``"auc_std"`` and ``"auc_runs"``.

    Raises:
        MetricError: If no run is given, the two lists disagree in length, or a run
            has only one class.
    """
    if not labels_per_run:
        raise MetricError("mean_roc_curve needs at least one run")
    if len(labels_per_run) != len(scores_per_run):
        raise MetricError(
            f"got {len(labels_per_run)} label arrays but {len(scores_per_run)} score arrays; "
            "each run contributes exactly one of each"
        )

    grid = np.linspace(0.0, 1.0, grid_size)
    curves, areas = [], []
    for labels, scores in zip(labels_per_run, scores_per_run, strict=True):
        labels = _as_labels(labels)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        false_positive, true_positive, _ = roc_curve(labels, scores)
        interpolated = np.interp(grid, false_positive, true_positive)
        # Interpolating onto a fixed grid does not guarantee the endpoints, which are
        # (0, 0) and (1, 1) for every ROC curve by construction.
        interpolated[0] = 0.0
        curves.append(interpolated)
        areas.append(float(roc_auc_score(labels, scores)))

    stacked = np.vstack(curves)
    mean = stacked.mean(axis=0)
    mean[-1] = 1.0

    return {
        "fpr": grid,
        "tpr_mean": mean,
        "tpr_std": stacked.std(axis=0),
        "tpr_runs": stacked,
        "auc_mean": float(np.mean(areas)),
        "auc_std": float(np.std(areas)),
        "auc_runs": areas,
    }
