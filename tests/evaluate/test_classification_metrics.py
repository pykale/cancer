"""Tests for ``kalecancer.evaluate.classification_metrics``."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from kalecancer.evaluate.classification_metrics import MetricError, binary_metrics, mean_roc_curve, roc_auc

LABELS = np.array([0, 0, 0, 1, 1, 1])
SEPARABLE = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])


def test_a_perfect_ranking_scores_one() -> None:
    assert roc_auc(LABELS, SEPARABLE) == 1.0


def test_an_inverted_ranking_scores_zero() -> None:
    assert roc_auc(LABELS, -SEPARABLE) == 0.0


def test_the_sigmoid_does_not_change_the_ranking() -> None:
    logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

    assert roc_auc(LABELS, logits) == roc_auc(LABELS, 1 / (1 + np.exp(-logits)))


def test_a_single_class_is_refused_by_name() -> None:
    with pytest.raises(MetricError, match="needs both classes"):
        roc_auc(np.zeros(6), SEPARABLE)


def test_binary_metrics_reports_counts_alongside_scores() -> None:
    metrics = binary_metrics(LABELS, SEPARABLE)

    assert metrics["n"] == 6
    assert metrics["n_positive"] == 3
    assert metrics["positive_rate"] == 0.5
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0


def test_binary_metrics_accepts_logits() -> None:
    """Logits fall outside [0, 1], so the threshold metrics have to squash them first."""
    metrics = binary_metrics(LABELS, np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]))

    assert metrics["accuracy"] == 1.0


def test_mean_roc_curve_lands_on_the_requested_grid() -> None:
    curve = mean_roc_curve([LABELS, LABELS], [SEPARABLE, SEPARABLE], grid_size=50)

    assert curve["fpr"].shape == (50,)
    assert curve["tpr_mean"].shape == (50,)
    assert curve["tpr_mean"][0] == 0.0
    assert curve["tpr_mean"][-1] == 1.0


def test_identical_runs_have_no_spread() -> None:
    curve = mean_roc_curve([LABELS] * 3, [SEPARABLE] * 3)

    assert np.allclose(curve["tpr_std"], 0.0)
    assert curve["auc_std"] == 0.0
    assert curve["auc_mean"] == 1.0


def test_each_run_contributes_one_area() -> None:
    curve = mean_roc_curve([LABELS, LABELS], [SEPARABLE, -SEPARABLE])

    assert curve["auc_runs"] == [1.0, 0.0]
    assert curve["auc_mean"] == 0.5


def test_averaging_curves_is_not_pooling_predictions() -> None:
    """Pinned because the two are easy to confuse and give different answers.

    Averaging interpolated curves weights each run equally; pooling every run's
    predictions into one curve weights each *subject* equally. The paper's figure
    averages curves, so that is what this reproduces.
    """
    labels = [np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])]
    scores = [np.array([0.1, 0.4, 0.35, 0.8]), np.array([0.2, 0.3, 0.9, 0.4])]

    averaged = mean_roc_curve(labels, scores)["auc_mean"]
    pooled = roc_auc_score(np.concatenate(labels), np.concatenate(scores))

    assert averaged != pytest.approx(pooled)


def test_mismatched_run_counts_are_refused() -> None:
    with pytest.raises(MetricError, match="exactly one of each"):
        mean_roc_curve([LABELS, LABELS], [SEPARABLE])


def test_no_runs_are_refused() -> None:
    with pytest.raises(MetricError, match="at least one run"):
        mean_roc_curve([], [])
