"""Tests for the task abstraction the trainers are parametrised by.

These pin the four things a task decides -- head, loss, whether a batch carries a
gradient, and the epoch metric -- independently of any trainer, so a new task can be
checked against the same contract.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from kalecancer.evaluate.classification_metrics import MetricError
from kalecancer.model.embed.multimodal_fusion import MultimodalOutput
from kalecancer.model.predict import CoxHead, LinearHead
from kalecancer.pipeline.task import ClassificationTask, PredictionTask, SurvivalTask

WIDTH = 6
SIZE = 8


def output(scores: torch.Tensor) -> MultimodalOutput:
    return MultimodalOutput(prediction=scores.reshape(-1, 1))


@pytest.mark.parametrize("task", [SurvivalTask(), ClassificationTask()], ids=lambda t: type(t).__name__)
def test_a_task_declares_the_contract_a_trainer_reads(task: PredictionTask) -> None:
    assert task.target_keys
    assert task.metric_name
    assert isinstance(task.build_head(WIDTH), nn.Module)


def test_the_cox_head_is_bias_free() -> None:
    """The partial likelihood is shift-invariant, so an intercept would never train."""
    head = SurvivalTask().build_head(WIDTH)

    assert isinstance(head, CoxHead)
    assert head.linear.bias is None


def test_the_classification_head_keeps_its_bias() -> None:
    """Cross-entropy is not shift-invariant: the intercept sets the base rate."""
    head = ClassificationTask().build_head(WIDTH)

    assert isinstance(head, LinearHead)
    assert head.linear.bias is not None


def test_survival_reports_no_signal_without_an_observed_event() -> None:
    task = SurvivalTask()
    targets = {"time": torch.arange(1.0, SIZE + 1), "event": torch.zeros(SIZE)}

    assert not task.has_signal(targets)


def test_survival_reports_signal_when_an_event_is_observed() -> None:
    task = SurvivalTask()
    targets = {"time": torch.arange(1.0, SIZE + 1), "event": torch.tensor([1.0] + [0.0] * (SIZE - 1))}

    assert task.has_signal(targets)


def test_classification_always_reports_signal() -> None:
    """Per-sample loss: an all-negative batch is still informative."""
    assert ClassificationTask().has_signal({"label": torch.zeros(SIZE)})


def test_the_c_index_needs_a_comparable_pair() -> None:
    task = SurvivalTask()
    targets = {"time": torch.arange(1.0, SIZE + 1), "event": torch.zeros(SIZE)}

    with pytest.raises(MetricError, match="no observed events"):
        task.metric(torch.randn(SIZE), targets)


def test_the_auc_needs_both_classes() -> None:
    task = ClassificationTask()

    with pytest.raises(MetricError):
        task.metric(torch.randn(SIZE), {"label": torch.zeros(SIZE)})


def test_a_perfect_ranking_scores_one_for_either_task() -> None:
    """Both metrics are ranking metrics, so a correct ordering is the maximum."""
    times = torch.arange(1.0, SIZE + 1)
    events = torch.ones(SIZE)
    # Higher risk must mean shorter survival, so risk descends as time ascends.
    assert SurvivalTask().metric(-times, {"time": times, "event": events}) == pytest.approx(1.0)

    labels = torch.tensor([0.0] * 4 + [1.0] * 4)
    assert ClassificationTask().metric(labels, {"label": labels}) == pytest.approx(1.0)


def test_the_positive_weight_reaches_the_loss() -> None:
    scores = torch.randn(SIZE)
    targets = {"label": torch.tensor([0.0] * 6 + [1.0] * 2)}

    plain = ClassificationTask(pos_weight=0.0).loss(output(scores), targets)
    weighted = ClassificationTask(pos_weight=5.0).loss(output(scores), targets)

    assert not torch.isclose(plain, weighted)


def test_a_disabled_positive_weight_is_none() -> None:
    """``None`` is what the loss expects for "unweighted", not a tensor of one."""
    assert ClassificationTask(pos_weight=0.0).pos_weight is None
