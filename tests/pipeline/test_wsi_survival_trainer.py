"""Tests for the WSI survival trainer."""

from __future__ import annotations

import torch

from kalecancer.loaddata.wsi_dataset import collate_bags
from kalecancer.pipeline import WSISurvivalTrainer

FEATURE_DIM = 8


def make_batch(size: int = 4, events: list[int] | None = None):
    events = events if events is not None else [i % 2 for i in range(size)]
    return collate_bags(
        [
            {
                "group_id": f"{i:03d}",
                "slide_ids": ("slide",),
                "features": torch.randn(6 + i, FEATURE_DIM),
                "coords": torch.zeros(6 + i, 2, dtype=torch.long),
                "slide_index": torch.zeros(6 + i, dtype=torch.long),
                "duration": torch.tensor(100.0 + i * 50),
                "event": torch.tensor(float(events[i])),
            }
            for i in range(size)
        ]
    )


def make_model() -> WSISurvivalTrainer:
    return WSISurvivalTrainer(input_dim=FEATURE_DIM, hidden_dim=8, attention_dim=4)


def test_predict_risk_restores_training_mode() -> None:
    """Predicting mid-training must not silently disable dropout afterwards."""
    model = make_model().train()

    model.predict_risk(make_batch())

    assert model.training


def test_predict_risk_leaves_an_evaluating_model_in_eval_mode() -> None:
    model = make_model().eval()

    model.predict_risk(make_batch())

    assert not model.training


def test_predict_risk_returns_one_risk_and_aligned_attention() -> None:
    batch = make_batch()

    risk, attentions = make_model().predict_risk(batch)

    assert risk.shape == (len(batch["samples"]),)
    assert [len(a) for a in attentions] == [s["features"].shape[0] for s in batch["samples"]]


def test_predict_risk_does_not_track_gradients() -> None:
    risk, _ = make_model().predict_risk(make_batch())

    assert not risk.requires_grad


def test_training_step_skips_batches_without_events() -> None:
    """A purely censored batch has no Cox risk set, so there is nothing to step on."""
    model = make_model()

    assert model.training_step(make_batch(events=[0, 0, 0, 0]), 0) is None


def test_training_step_returns_a_loss_when_events_are_present() -> None:
    model = make_model()

    loss = model.training_step(make_batch(events=[1, 0, 1, 0]), 0)

    assert loss is not None and torch.isfinite(loss)


def test_training_batches_are_not_accumulated_for_metrics() -> None:
    """Only the evaluation splits report an epoch-level C-index."""
    model = make_model()

    model.compute_loss(make_batch(), split_name="train")

    assert "train" not in model._epoch_outputs


def test_evaluation_batches_are_accumulated_for_the_epoch_metric() -> None:
    model = make_model()

    model.compute_loss(make_batch(), split_name="valid")

    assert len(model._epoch_outputs["valid"]) == 1
