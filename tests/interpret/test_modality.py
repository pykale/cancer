"""Tests for modality contribution: leave-one-out ablation and Shapley values.

A synthetic 2-3 modality stub model stands in for a trained
:class:`~kalecancer.pipeline.CohortTrainer`: it reads ``present`` and a per-modality
linear weight directly, so a modality's true contribution to the risk score is known
exactly rather than needing a model to converge on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kalecancer.evaluate.survival_metrics import concordance_index
from kalecancer.interpret import leave_one_out_ablation, modality_shapley
from kalecancer.interpret.modality import _score
from kalecancer.loaddata.multimodal_access import PatientBatch
from kalecancer.pipeline.task import ClassificationTask, PredictionTask, SurvivalTask

FEATURE = "informative"
NOISE = "noise"


@dataclass
class _SurvivalStub:
    """The minimum surface the ablation helpers read: embedders, task and predict.

    Risk is an exact linear function of each modality's value, gated by its
    ``present`` bit -- the same mechanism :class:`~kalecancer.model.embed.MultimodalFusion`
    uses, simplified so a modality's ground-truth contribution is known exactly.
    """

    weights: dict[str, float]
    task: PredictionTask = field(default_factory=SurvivalTask)
    training: bool = True

    def __post_init__(self) -> None:
        self.embedders = dict.fromkeys(self.weights)

    def eval(self) -> None:
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def predict(self, batch: PatientBatch):
        risk = torch.zeros(len(batch.patient_id), dtype=torch.float64)
        for name, weight in self.weights.items():
            modality = batch.modalities[name]
            assert isinstance(modality, torch.Tensor)
            value = modality.squeeze(-1).double()
            present = batch.present[name].double()
            risk = risk + weight * value * present
        return SimpleNamespace(prediction=risk)


def make_survival_batch(n: int = 300, seed: int = 0, modalities: tuple[str, ...] = (FEATURE, NOISE)) -> PatientBatch:
    """A cohort whose event time is driven entirely by ``FEATURE``; every other
    modality is standard-normal noise unrelated to the outcome."""
    rng = np.random.default_rng(seed)
    values = {name: rng.standard_normal(n) for name in modalities}

    rate = np.exp(values[FEATURE]) * 1e-3
    event_time = rng.exponential(1.0 / rate)
    censor_time = rng.exponential(1.0 / 1.5e-3, size=n)
    times = np.minimum(event_time, censor_time)
    events = event_time <= censor_time

    return PatientBatch(
        patient_id=[f"{i:03d}" for i in range(n)],
        modalities={name: torch.tensor(v, dtype=torch.float32).unsqueeze(-1) for name, v in values.items()},
        present={name: torch.ones(n, dtype=torch.bool) for name in modalities},
        target={
            "time": torch.tensor(times, dtype=torch.float32),
            "event": torch.tensor(events, dtype=torch.bool),
        },
    )


class _CountingLoader:
    """Wraps a fixed list of batches, counting how many times it is iterated fully."""

    def __init__(self, batches: list[PatientBatch]) -> None:
        self.batches = batches
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(self.batches)


def test_loo_detects_the_informative_modality_only() -> None:
    model = _SurvivalStub({FEATURE: 1.0, NOISE: 0.0})
    loader = [make_survival_batch(seed=1)]

    result = leave_one_out_ablation(model, loader)

    assert result[NOISE] == pytest.approx(0.0, abs=1e-9)
    assert result[FEATURE] > 0.05
    assert "baseline" in result


def test_shapley_detects_the_informative_modality_only() -> None:
    model = _SurvivalStub({FEATURE: 1.0, NOISE: 0.0})
    loader = [make_survival_batch(seed=2)]

    result = modality_shapley(model, loader)

    assert result[NOISE] == pytest.approx(0.0, abs=1e-9)
    assert result[FEATURE] > 0.05


def test_shapley_values_sum_to_full_minus_empty_coalition() -> None:
    """Shapley efficiency: values sum to the full-coalition minus the empty-coalition score."""
    model = _SurvivalStub({FEATURE: 1.0, NOISE: 0.3})
    loader = [make_survival_batch(seed=3)]

    result = modality_shapley(model, loader)

    full_score = _score(model, loader, frozenset(model.embedders), concordance_index)
    empty_score = _score(model, loader, frozenset(), concordance_index)

    assert sum(result.values()) == pytest.approx(full_score - empty_score, abs=1e-6)


def test_three_modalities_each_coalition_scored_exactly_once() -> None:
    model = _SurvivalStub({FEATURE: 1.0, NOISE: 0.3, "clinical": -0.2})
    loader = _CountingLoader([make_survival_batch(seed=4, modalities=(FEATURE, NOISE, "clinical"))])

    modality_shapley(model, loader)

    assert loader.iterations == 2**3


def test_max_modalities_exceeded_is_refused() -> None:
    names = (FEATURE, NOISE, "m2", "m3", "m4")
    model = _SurvivalStub(dict.fromkeys(names, 1.0))
    loader = [make_survival_batch(seed=5, modalities=names)]

    with pytest.raises(ValueError, match="max_modalities"):
        modality_shapley(model, loader, max_modalities=4)


def test_training_mode_is_restored_after_ablation() -> None:
    model = _SurvivalStub({FEATURE: 1.0, NOISE: 0.0})
    model.training = True
    loader = [make_survival_batch(seed=6)]

    leave_one_out_ablation(model, loader)

    assert model.training is True


def test_training_mode_is_restored_after_shapley() -> None:
    model = _SurvivalStub({FEATURE: 1.0, NOISE: 0.0})
    model.training = True
    loader = [make_survival_batch(seed=7)]

    modality_shapley(model, loader)

    assert model.training is True


def test_an_explicit_modalities_subset_is_respected() -> None:
    model = _SurvivalStub({FEATURE: 1.0, NOISE: 0.3})
    loader = [make_survival_batch(seed=8)]

    result = leave_one_out_ablation(model, loader, modalities=[FEATURE])

    assert set(result) == {"baseline", FEATURE}


def test_a_classification_task_is_refused_explicitly() -> None:
    model = _SurvivalStub({FEATURE: 1.0, NOISE: 0.0})
    model.task = ClassificationTask()
    loader = [make_survival_batch(seed=9)]

    with pytest.raises(ValueError, match="survival"):
        leave_one_out_ablation(model, loader)

    with pytest.raises(ValueError, match="survival"):
        modality_shapley(model, loader)
