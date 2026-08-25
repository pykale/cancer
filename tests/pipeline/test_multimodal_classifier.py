"""Tests for the generic multimodal classification trainer.

The trainer names no modality and no dataset, so these use arbitrary modality names
and synthetic tensors, mirroring ``test_multimodal_trainer.py``.
"""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from kalecancer.loaddata.sample import PatientBatch
from kalecancer.pipeline.multimodal_classifier import MultimodalClassificationTrainer

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


def make_batch(size: int = BATCH, labels: list[int] | None = None, absent: str | None = None) -> PatientBatch:
    generator = torch.Generator().manual_seed(0)
    labels = labels if labels is not None else [index % 2 for index in range(size)]
    present = {name: torch.ones(size, dtype=torch.bool) for name in RAW_DIMS}
    if absent:
        present[absent] = torch.zeros(size, dtype=torch.bool)

    return PatientBatch(
        patient_id=[f"{index:03d}" for index in range(size)],
        modalities={name: torch.randn(size, dim, generator=generator) for name, dim in RAW_DIMS.items()},
        present=present,
        target={"label": torch.tensor(labels, dtype=torch.float32)},
    )


def make_trainer(stage: str = "intermediate", **kwargs) -> MultimodalClassificationTrainer:
    torch.manual_seed(0)
    embedders = {name: ToyEmbedder(dim) for name, dim in RAW_DIMS.items()}
    return MultimodalClassificationTrainer(embedders, stage=stage, fusion_dim=16, **kwargs)


@pytest.mark.parametrize("stage", STAGES)
def test_loss_is_finite_and_differentiable(stage: str) -> None:
    model = make_trainer(stage)

    loss, metrics = model.compute_loss(make_batch(), split_name="train")
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["train_loss"] is loss


@pytest.mark.parametrize("stage", STAGES)
def test_one_logit_per_patient(stage: str) -> None:
    model = make_trainer(stage)

    output = model.predict_logits(make_batch())

    assert output.prediction.reshape(-1).shape == (BATCH,)


def test_a_batch_without_positives_still_trains() -> None:
    """The contrast with the Cox trainer, which skips a batch with no events.

    Cross-entropy is a per-sample loss, so an all-negative batch is perfectly
    informative: it says every one of these patients is negative.
    """
    model = make_trainer()

    loss = model.training_step(make_batch(labels=[0] * BATCH), batch_idx=0)

    assert loss is not None
    assert torch.isfinite(loss)
    assert loss.grad_fn is not None


def test_an_absent_modality_is_tolerated() -> None:
    model = make_trainer()

    output = model.predict_logits(make_batch(absent="beta"))

    assert torch.isfinite(output.prediction).all()


def test_positive_weighting_changes_the_loss() -> None:
    batch = make_batch(labels=[0, 0, 0, 0, 0, 0, 1, 1])
    plain, _ = make_trainer(pos_weight=0.0).compute_loss(batch)
    weighted, _ = make_trainer(pos_weight=5.0).compute_loss(batch)

    assert not torch.isclose(plain, weighted)


def test_auxiliary_supervision_changes_the_loss() -> None:
    batch = make_batch()
    plain, _ = make_trainer("hybrid", auxiliary_weight=0.0).compute_loss(batch)
    auxiliary, _ = make_trainer("hybrid", auxiliary_weight=0.5).compute_loss(batch)

    assert not torch.isclose(plain, auxiliary)


def test_predicting_restores_the_training_mode() -> None:
    model = make_trainer()
    model.train()

    model.predict_logits(make_batch())

    assert model.training


class _Records(Dataset):
    def __init__(self, batches: int = 4) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return self.batches

    def __getitem__(self, index: int) -> PatientBatch:
        return make_batch()


@pytest.mark.parametrize("stage", STAGES)
def test_lightning_can_fit_the_trainer(stage: str, tmp_path) -> None:
    model = make_trainer(stage, max_epochs=1)
    loader = DataLoader(_Records(), batch_size=None)

    trainer = pl.Trainer(
        max_epochs=1, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False
    )
    trainer.fit(model, loader, loader)

    assert "valid_auc" in trainer.callback_metrics
    assert 0.0 <= float(trainer.callback_metrics["valid_auc"]) <= 1.0


def test_a_single_class_split_logs_no_auc() -> None:
    """A split with one class has no ranking to score, so the epoch end skips it."""
    model = make_trainer()
    model.compute_loss(make_batch(labels=[0] * BATCH), split_name="valid")

    model._log_auc("valid")

    assert model._epoch_outputs.get("valid") is None
