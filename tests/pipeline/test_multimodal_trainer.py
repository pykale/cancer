"""Tests for the generic multimodal survival trainer.

The trainer names no modality and no dataset, so these use arbitrary modality names
and synthetic tensors.
"""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from kalecancer.loaddata.sample import PatientBatch
from kalecancer.pipeline.multimodal_trainer import MultimodalSurvivalTrainer

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


def make_batch(size: int = BATCH, events: list[int] | None = None, absent: str | None = None) -> PatientBatch:
    generator = torch.Generator().manual_seed(0)
    events = events if events is not None else [index % 2 for index in range(size)]
    present = {name: torch.ones(size, dtype=torch.bool) for name in RAW_DIMS}
    if absent:
        present[absent] = torch.zeros(size, dtype=torch.bool)

    return PatientBatch(
        patient_id=[f"{index:03d}" for index in range(size)],
        modalities={name: torch.randn(size, dim, generator=generator) for name, dim in RAW_DIMS.items()},
        present=present,
        target={
            "time": torch.rand(size, generator=generator) * 900 + 50,
            "event": torch.tensor(events, dtype=torch.float32),
        },
    )


def make_trainer(stage: str = "intermediate", **kwargs) -> MultimodalSurvivalTrainer:
    torch.manual_seed(0)
    embedders = {name: ToyEmbedder(dim) for name, dim in RAW_DIMS.items()}
    return MultimodalSurvivalTrainer(embedders, stage=stage, fusion_dim=16, **kwargs)


@pytest.mark.parametrize("stage", STAGES)
def test_loss_is_finite_and_differentiable(stage: str) -> None:
    loss, _ = make_trainer(stage).compute_loss(make_batch(), "train")

    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_predicts_one_risk_per_patient(stage: str) -> None:
    output = make_trainer(stage).predict_risk(make_batch())

    assert output.prediction.reshape(-1).shape == (BATCH,)


def test_the_batch_decides_which_modalities_are_used() -> None:
    """No modality is named in the trainer, so the batch drives everything."""
    model = make_trainer()

    assert set(model.model.modalities) == set(RAW_DIMS)


def test_an_absent_modality_is_tolerated() -> None:
    output = make_trainer("late").predict_risk(make_batch(absent="beta"))

    assert bool(torch.isfinite(output.prediction).all())


def test_a_batch_without_events_is_skipped() -> None:
    """A purely censored batch carries no Cox gradient, so no optimiser step happens."""
    model = make_trainer()

    assert model.training_step(make_batch(events=[0] * BATCH), 0) is None


def test_a_batch_with_events_produces_a_loss() -> None:
    model = make_trainer()

    assert model.training_step(make_batch(events=[1, 0] * (BATCH // 2)), 0) is not None


def test_auxiliary_weight_changes_the_loss() -> None:
    batch = make_batch()

    plain, _ = make_trainer("hybrid", auxiliary_weight=0.0).compute_loss(batch, "train")
    weighted, _ = make_trainer("hybrid", auxiliary_weight=0.5).compute_loss(batch, "train")

    assert not torch.isclose(plain, weighted)


def test_predict_risk_restores_training_mode() -> None:
    model = make_trainer()
    model.train()

    model.predict_risk(make_batch())

    assert model.training


class _Batches(Dataset):
    def __init__(self, count: int = 3) -> None:
        self.batches = [make_batch() for _ in range(count)]

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> PatientBatch:
        return self.batches[index]


@pytest.mark.parametrize("stage", STAGES)
def test_lightning_can_fit_the_trainer(stage: str) -> None:
    """The end this exists for: Lightning drives it with no adapters."""
    loader = DataLoader(_Batches(), batch_size=None, collate_fn=None)
    model = make_trainer(stage, max_epochs=1)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    trainer.fit(model, loader, loader)

    assert "valid_c_index" in trainer.callback_metrics
