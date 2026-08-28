"""Tests for the trainer, across both tasks and both shapes of modality.

The trainer names no modality, no dataset and no endpoint, so these use arbitrary
modality names and synthetic tensors. Anything that holds for every task is
parametrised over both; anything that follows from one objective in particular is
tested against that task alone and says which property of the objective it comes from.

The ragged case matters as much as the fixed-width one: a whole-slide model is this
same trainer with a single bag modality, so there is no separate trainer to test.
"""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from kalecancer.loaddata.multimodal_access import PatientBatch
from kalecancer.model.embed import BagEncoder
from kalecancer.model.layers import AttentionMIL
from kalecancer.pipeline import ClassificationTask, CohortTrainer, PredictionTask, SurvivalTask

BATCH = 8
RAW_DIMS = {"alpha": 12, "beta": 7}
STAGES = ["intermediate", "late", "hybrid"]


class ToyEmbedder(nn.Module):
    needs_full_batch = False

    def __init__(self, in_dim: int, out_dim: int = 16) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def survival_targets(size: int, events: list[int] | None = None) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    events = events if events is not None else [index % 2 for index in range(size)]
    return {
        "time": torch.rand(size, generator=generator) * 900 + 50,
        "event": torch.tensor(events, dtype=torch.float32),
    }


def classification_targets(size: int, labels: list[int] | None = None) -> dict[str, torch.Tensor]:
    labels = labels if labels is not None else [index % 2 for index in range(size)]
    return {"label": torch.tensor(labels, dtype=torch.float32)}


#: Each task with a matching target builder, so one test body serves both.
TASKS = {
    "survival": (SurvivalTask, survival_targets, "c_index"),
    "classification": (ClassificationTask, classification_targets, "auc"),
}


def make_batch(targets: dict[str, torch.Tensor], size: int = BATCH, absent: str | None = None) -> PatientBatch:
    generator = torch.Generator().manual_seed(0)
    present = {name: torch.ones(size, dtype=torch.bool) for name in RAW_DIMS}
    if absent:
        present[absent] = torch.zeros(size, dtype=torch.bool)

    return PatientBatch(
        patient_id=[f"{index:03d}" for index in range(size)],
        modalities={name: torch.randn(size, dim, generator=generator) for name, dim in RAW_DIMS.items()},
        present=present,
        target=targets,
    )


def make_trainer(task: PredictionTask, stage: str = "intermediate", **kwargs) -> CohortTrainer:
    torch.manual_seed(0)
    embedders = {name: ToyEmbedder(dim) for name, dim in RAW_DIMS.items()}
    return CohortTrainer(embedders, task=task, stage=stage, fusion_dim=16, **kwargs)


@pytest.mark.parametrize("task_name", list(TASKS))
@pytest.mark.parametrize("stage", STAGES)
def test_loss_is_finite_and_differentiable(task_name: str, stage: str) -> None:
    task_cls, targets, _ = TASKS[task_name]
    model = make_trainer(task_cls(), stage)

    loss, metrics = model.compute_loss(make_batch(targets(BATCH)), split_name="train")
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["train_loss"] is loss


@pytest.mark.parametrize("task_name", list(TASKS))
@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_predicts_one_score_per_patient(task_name: str, stage: str) -> None:
    task_cls, targets, _ = TASKS[task_name]

    output = make_trainer(task_cls(), stage).predict(make_batch(targets(BATCH)))

    assert output.prediction.reshape(-1).shape == (BATCH,)


@pytest.mark.parametrize("task_name", list(TASKS))
def test_an_absent_modality_is_tolerated(task_name: str) -> None:
    task_cls, targets, _ = TASKS[task_name]

    output = make_trainer(task_cls(), "late").predict(make_batch(targets(BATCH), absent="beta"))

    assert bool(torch.isfinite(output.prediction).all())


@pytest.mark.parametrize("task_name", list(TASKS))
def test_auxiliary_weight_changes_the_loss(task_name: str) -> None:
    task_cls, targets, _ = TASKS[task_name]
    batch = make_batch(targets(BATCH))

    plain, _ = make_trainer(task_cls(), "hybrid", auxiliary_weight=0.0).compute_loss(batch)
    weighted, _ = make_trainer(task_cls(), "hybrid", auxiliary_weight=0.5).compute_loss(batch)

    assert not torch.isclose(plain, weighted)


@pytest.mark.parametrize("task_name", list(TASKS))
def test_predicting_restores_the_training_mode(task_name: str) -> None:
    task_cls, targets, _ = TASKS[task_name]
    model = make_trainer(task_cls())
    model.train()

    model.predict(make_batch(targets(BATCH)))

    assert model.training


@pytest.mark.parametrize("task_name", list(TASKS))
def test_training_batches_are_not_accumulated_for_metrics(task_name: str) -> None:
    """Only the evaluation splits report an epoch-level metric."""
    task_cls, targets, _ = TASKS[task_name]
    model = make_trainer(task_cls())

    model.compute_loss(make_batch(targets(BATCH)), split_name="train")

    assert "train" not in model._epoch_outputs


def test_the_batch_decides_which_modalities_are_used() -> None:
    """No modality is named in the trainer, so the batch drives everything."""
    model = make_trainer(SurvivalTask())

    assert set(model.model.modalities) == set(RAW_DIMS)


def test_a_batch_without_events_is_skipped() -> None:
    """A purely censored batch carries no Cox gradient, so no optimiser step happens."""
    model = make_trainer(SurvivalTask())

    assert model.training_step(make_batch(survival_targets(BATCH, events=[0] * BATCH)), 0) is None


def test_a_batch_with_events_produces_a_loss() -> None:
    model = make_trainer(SurvivalTask())

    assert model.training_step(make_batch(survival_targets(BATCH, events=[1, 0] * (BATCH // 2))), 0) is not None


def test_a_batch_without_positives_still_trains() -> None:
    """The contrast with the Cox task, which skips a batch with no events.

    Cross-entropy is a per-sample loss, so an all-negative batch is perfectly
    informative: it says every one of these patients is negative.
    """
    model = make_trainer(ClassificationTask())

    loss = model.training_step(make_batch(classification_targets(BATCH, labels=[0] * BATCH)), batch_idx=0)

    assert loss is not None
    assert torch.isfinite(loss)
    assert loss.grad_fn is not None


def test_positive_weighting_changes_the_loss() -> None:
    batch = make_batch(classification_targets(BATCH, labels=[0, 0, 0, 0, 0, 0, 1, 1]))

    plain, _ = make_trainer(ClassificationTask(pos_weight=0.0)).compute_loss(batch)
    weighted, _ = make_trainer(ClassificationTask(pos_weight=5.0)).compute_loss(batch)

    assert not torch.isclose(plain, weighted)


def test_the_class_weight_follows_the_trainer_to_its_device() -> None:
    """A task is a module so its tensors move with the trainer, not left on the CPU."""
    model = make_trainer(ClassificationTask(pos_weight=3.0))

    assert model.task.pos_weight in set(model.buffers())


@pytest.mark.parametrize(
    ("task", "targets"),
    [
        (SurvivalTask(), survival_targets(BATCH, events=[0] * BATCH)),
        (ClassificationTask(), classification_targets(BATCH, labels=[0] * BATCH)),
    ],
    ids=["no-events", "one-class"],
)
def test_a_split_the_metric_cannot_score_is_skipped(task: PredictionTask, targets: dict) -> None:
    """Neither the C-index nor the AUC exists here, and neither should fail the run."""
    model = make_trainer(task)
    model._evaluation_step(make_batch(targets), "valid")

    model._log_epoch_metric("valid")

    assert model._epoch_outputs.get("valid") is None


def test_an_all_censored_validation_batch_does_not_raise() -> None:
    """With heavy censoring and small batches this is ordinary, not exceptional.

    The Cox loss needs an observed event, so the guard has to sit in front of the
    loss rather than in front of the logging.
    """
    model = make_trainer(SurvivalTask())

    model._evaluation_step(make_batch(survival_targets(BATCH, events=[0] * BATCH)), "valid")

    # Skipping the loss must not skip the predictions: censored patients are still
    # comparable to anyone who failed before them.
    assert len(model._epoch_outputs["valid"]) == 1


class _Batches(Dataset):
    def __init__(self, targets, count: int = 3) -> None:
        self.batches = [make_batch(targets(BATCH)) for _ in range(count)]

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> PatientBatch:
        return self.batches[index]


@pytest.mark.parametrize("task_name", list(TASKS))
@pytest.mark.parametrize("stage", STAGES)
def test_lightning_can_fit_the_trainer(task_name: str, stage: str) -> None:
    """The end this exists for: Lightning drives it with no adapters."""
    task_cls, targets, metric_name = TASKS[task_name]
    loader = DataLoader(_Batches(targets), batch_size=None, collate_fn=None)
    model = make_trainer(task_cls(), stage, max_epochs=1)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model, loader, loader)

    assert f"valid_{metric_name}" in trainer.callback_metrics


# --------------------------------------------------------------------------- #
# A bag modality is an ordinary modality
# --------------------------------------------------------------------------- #
#
# This is what removed the need for a whole-slide trainer of its own: a bag of
# patches is a modality whose value is a list rather than a stacked tensor, and
# BagEncoder pools it to one vector like any other embedder.

BAG_DIM = 8


def bag_batch(size: int = BATCH, events: list[int] | None = None) -> PatientBatch:
    """A single ragged bag modality, as a whole-slide cohort produces."""
    generator = torch.Generator().manual_seed(3)
    return PatientBatch(
        patient_id=[f"{index:03d}" for index in range(size)],
        modalities={"slides": [torch.randn(4 + index, BAG_DIM, generator=generator) for index in range(size)]},
        present={"slides": torch.ones(size, dtype=torch.bool)},
        target=survival_targets(size, events),
    )


def bag_trainer(task: PredictionTask) -> CohortTrainer:
    torch.manual_seed(0)
    encoder = BagEncoder(AttentionMIL(input_dim=BAG_DIM, hidden_dim=8, attention_dim=4))
    return CohortTrainer({"slides": encoder}, task=task, fusion_dim=8)


def test_a_ragged_bag_modality_trains_like_any_other() -> None:
    model = bag_trainer(SurvivalTask())

    loss = model.training_step(bag_batch(events=[1, 0] * (BATCH // 2)), 0)

    assert loss is not None and torch.isfinite(loss)


def test_a_bag_modality_predicts_one_score_per_patient() -> None:
    batch = bag_batch()

    output = bag_trainer(SurvivalTask()).predict(batch)

    assert output.prediction.reshape(-1).shape == (BATCH,)


def test_bag_attention_stays_aligned_with_each_patients_patches() -> None:
    """What interpretation depends on: one weight per patch, per patient."""
    batch = bag_batch()
    model = bag_trainer(SurvivalTask())

    model.predict(batch)

    weights = model.embedders["slides"].last_attention
    assert [len(w) for w in weights] == [len(bag) for bag in batch.modalities["slides"]]


def test_the_same_bag_pipeline_serves_a_binary_endpoint() -> None:
    """The whole point: swapping the task is the only change a new endpoint needs."""
    batch = bag_batch()
    batch.target = classification_targets(BATCH)
    model = bag_trainer(ClassificationTask())

    loss = model.training_step(batch, 0)

    assert loss is not None and torch.isfinite(loss)
    assert model.task.metric_name == "auc"
