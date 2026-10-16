"""Tests for raw clinical feature attribution: permutation importance and Shapley.

A synthetic clinical modality of 3 raw features (one informative, two pure noise)
drives survival time through a *known* linear combination, read by a stand-in for a
frozen TabICL embedder (:class:`_FakeTabICLEmbedder`, a real
:class:`~kalecancer.model.embed.tabicl.TabICLEmbedder` subclass that skips the
checkpoint) wired into a real :class:`~kalecancer.pipeline.CohortTrainer`. A second,
non-tabular modality (:class:`_CountingEmbedder`) ignores its own input and counts
its calls, so the caching requirement (the non-tabular branch is embedded once, not
once per perturbation) is checked against the real pipeline rather than a mock.

Both test functions below use a custom metric, ``discrimination`` (``|concordance -
0.5|``), rather than the default ``concordance_index``: an untrained
:class:`~kalecancer.model.predict.CoxHead` on top of random fusion weights learns an
arbitrary *sign* for the informative feature's effect (high age may as well predict
low risk as high risk, with no training to fix the direction), and plain concordance
would make the test depend on which sign random initialisation happened to land on.
``discrimination`` does not care about sign, only distance from chance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from kalecancer.evaluate.survival_metrics import concordance_index
from kalecancer.interpret import feature_shapley, permutation_importance
from kalecancer.loaddata.multimodal_access import PatientBatch
from kalecancer.model.embed.tabicl import TabICLEmbedder
from kalecancer.pipeline import ClassificationTask, CohortTrainer, SurvivalTask

FEATURES = ("age", "stage", "lab")
#: Only "age" (column 0) affects the model at all; "stage" and "lab" are pure noise.
WEIGHTS_TRUE = np.array([5.0, 0.0, 0.0])


def discrimination(risk: np.ndarray, durations: np.ndarray, events: np.ndarray) -> float:
    return abs(concordance_index(risk, durations, events) - 0.5)


class _FakeTabICLEmbedder(TabICLEmbedder):
    """A frozen TabICL stand-in needing no checkpoint: a known linear map of the raw
    features, so a feature's true contribution is known exactly rather than needing a
    model to converge on it."""

    #: Declared ahead, or nn.Module's attribute lookup widens it to Tensor | Module.
    _weights: torch.Tensor

    def __init__(self, weights: np.ndarray) -> None:
        nn.Module.__init__(self)  # bypass TabICLEmbedder.__init__: no checkpoint needed
        self.n_features = len(weights)
        self.out_dim = 1
        self.register_buffer("_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        return (x.float() @ self._weights).unsqueeze(-1)

    def train(self, mode: bool = True) -> _FakeTabICLEmbedder:
        # TabICLEmbedder.train() reads self.trainable/self.backbone, neither of
        # which this no-checkpoint stand-in has; plain nn.Module behaviour suffices.
        nn.Module.train(self, mode)
        return self


class _CountingEmbedder(nn.Module):
    """A non-tabular modality's embedder: ignores its input, counts how often it runs."""

    def __init__(self, out_dim: int = 3) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.calls = 0

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        self.calls += 1
        return torch.zeros(x.shape[0], self.out_dim)


def make_tabular_batch(n: int = 60, seed: int = 0) -> PatientBatch:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(FEATURES)))
    true_signal = X @ WEIGHTS_TRUE
    rate = np.exp(true_signal) * 1e-3
    event_time = rng.exponential(1.0 / rate)
    censor_time = rng.exponential(1.0 / 1.5e-3, size=n)
    times = np.minimum(event_time, censor_time)
    events = event_time <= censor_time

    return PatientBatch(
        patient_id=[f"{i:03d}" for i in range(n)],
        modalities={
            "clinical": torch.tensor(X, dtype=torch.float32),
            "other": torch.tensor(rng.standard_normal((n, 3)), dtype=torch.float32),
        },
        present={
            "clinical": torch.ones(n, dtype=torch.bool),
            "other": torch.ones(n, dtype=torch.bool),
        },
        target={
            "time": torch.tensor(times, dtype=torch.float32),
            "event": torch.tensor(events, dtype=torch.bool),
        },
    )


def build_model(seed: int = 0) -> tuple[CohortTrainer, _CountingEmbedder]:
    torch.manual_seed(seed)
    other = _CountingEmbedder(out_dim=3)
    model = CohortTrainer(
        {"clinical": _FakeTabICLEmbedder(WEIGHTS_TRUE), "other": other}, task=SurvivalTask(), fusion_dim=4
    )
    return model, other


def test_informative_feature_dominates_by_both_methods() -> None:
    model, _ = build_model()
    loader = [make_tabular_batch(seed=1)]

    importance = permutation_importance(
        model, loader, feature_names=list(FEATURES), metric=discrimination, n_repeats=5, seed=0
    )
    shapley = feature_shapley(model, loader, feature_names=list(FEATURES), metric=discrimination, n_samples=30, seed=0)

    assert importance["age"]["mean"] > 0.05
    assert importance["stage"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert importance["lab"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert importance["stage"]["std"] == pytest.approx(0.0, abs=1e-9)

    assert shapley["age"] > 0.05
    assert shapley["stage"] == pytest.approx(0.0, abs=1e-9)
    assert shapley["lab"] == pytest.approx(0.0, abs=1e-9)
    assert max(shapley, key=lambda name: shapley[name]) == "age"


def test_feature_shapley_is_deterministic_for_a_fixed_seed() -> None:
    model, _ = build_model()
    loader = [make_tabular_batch(seed=2)]

    first = feature_shapley(model, loader, feature_names=list(FEATURES), metric=discrimination, n_samples=20, seed=7)
    second = feature_shapley(model, loader, feature_names=list(FEATURES), metric=discrimination, n_samples=20, seed=7)
    third = feature_shapley(model, loader, feature_names=list(FEATURES), metric=discrimination, n_samples=20, seed=99)

    assert first == second
    assert first != third
    ranking = lambda result: max(result, key=lambda name: result[name])  # noqa: E731
    assert ranking(first) == ranking(third) == "age"


def test_permutation_importance_is_deterministic_for_a_fixed_seed() -> None:
    model, _ = build_model()
    loader = [make_tabular_batch(seed=3)]

    first = permutation_importance(
        model, loader, feature_names=list(FEATURES), metric=discrimination, n_repeats=5, seed=11
    )
    second = permutation_importance(
        model, loader, feature_names=list(FEATURES), metric=discrimination, n_repeats=5, seed=11
    )
    third = permutation_importance(
        model, loader, feature_names=list(FEATURES), metric=discrimination, n_repeats=5, seed=12
    )

    assert first == second
    assert first != third
    ranking = lambda result: max(result, key=lambda name: result[name]["mean"])  # noqa: E731
    assert ranking(first) == ranking(third) == "age"


def test_n_repeats_changes_std_but_not_materially_the_mean() -> None:
    model, _ = build_model()
    loader = [make_tabular_batch(seed=4)]

    small = permutation_importance(
        model, loader, feature_names=list(FEATURES), metric=discrimination, n_repeats=3, seed=5
    )
    large = permutation_importance(
        model, loader, feature_names=list(FEATURES), metric=discrimination, n_repeats=50, seed=5
    )

    assert small["age"]["mean"] == pytest.approx(large["age"]["mean"], abs=0.05)
    assert small["age"]["std"] != large["age"]["std"]


def test_missing_and_undiscoverable_feature_names_is_refused() -> None:
    model, _ = build_model()
    loader = [make_tabular_batch(seed=6)]  # a plain list has no `dataset.feature_names`

    with pytest.raises(ValueError, match="feature_names"):
        permutation_importance(model, loader)

    with pytest.raises(ValueError, match="feature_names"):
        feature_shapley(model, loader)


def test_other_modality_is_embedded_once_for_permutation_importance() -> None:
    model, other = build_model()
    loader = [make_tabular_batch(seed=7, n=30)]

    permutation_importance(model, loader, feature_names=list(FEATURES), metric=discrimination, n_repeats=5, seed=0)

    assert other.calls == 1
    assert model.embedders["other"] is other


def test_other_modality_is_embedded_once_for_feature_shapley() -> None:
    model, other = build_model()
    loader = [make_tabular_batch(seed=8, n=30)]

    feature_shapley(model, loader, feature_names=list(FEATURES), metric=discrimination, n_samples=20, seed=0)

    assert other.calls == 1
    assert model.embedders["other"] is other


def test_training_mode_is_restored_after_permutation_importance() -> None:
    model, _ = build_model()
    model.train()
    loader = [make_tabular_batch(seed=9)]

    permutation_importance(model, loader, feature_names=list(FEATURES), metric=discrimination, n_repeats=2, seed=0)

    assert model.training is True


def test_training_mode_is_restored_after_feature_shapley() -> None:
    model, _ = build_model()
    model.train()
    loader = [make_tabular_batch(seed=10)]

    feature_shapley(model, loader, feature_names=list(FEATURES), metric=discrimination, n_samples=10, seed=0)

    assert model.training is True


def test_a_classification_task_is_refused_explicitly() -> None:
    model, _ = build_model()
    model.task = ClassificationTask()
    loader = [make_tabular_batch(seed=11)]

    with pytest.raises(ValueError, match="survival"):
        permutation_importance(model, loader, feature_names=list(FEATURES))

    with pytest.raises(ValueError, match="survival"):
        feature_shapley(model, loader, feature_names=list(FEATURES))


def _numel(value: torch.Tensor | list[torch.Tensor]) -> int:
    assert isinstance(value, torch.Tensor)
    return value.numel()


def test_caching_actually_releases_the_other_modalitys_raw_tensor() -> None:
    """Not just "the batch stops carrying it": a tensor can be absent from the batch
    and still be alive, retained by something else. This checks it is actually freed,
    which is what matters for peak memory -- the thing a weakref, not the batch's own
    dict, can prove."""
    import gc
    import weakref

    from kalecancer.interpret.attribution import _cache_other_modalities, _materialise, _restore_modalities

    model, _ = build_model()
    loader = [make_tabular_batch(seed=14, n=30)]

    with torch.no_grad():
        full_batch = _materialise(loader)
        original_other = full_batch.modalities["other"]
        assert _numel(original_other) > 0
        tracked = weakref.ref(original_other)

        original_embedders = _cache_other_modalities(model, full_batch, "clinical")

        # Drop every other reference before checking: the weakref itself does not
        # keep the tensor alive, but this local variable otherwise would.
        del original_other
        gc.collect()

        assert tracked() is None, "the raw tensor must actually be freed, not merely dropped from the batch"

        _restore_modalities(model, original_embedders)


def test_scoring_never_hands_predict_the_full_other_modality_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    model, _ = build_model()
    loader = [make_tabular_batch(seed=16, n=30)]

    seen_sizes: list[int] = []
    original_predict = model.predict

    def spying_predict(batch):
        seen_sizes.append(_numel(batch.modalities["other"]))
        return original_predict(batch)

    monkeypatch.setattr(model, "predict", spying_predict)

    feature_shapley(model, loader, feature_names=list(FEATURES), metric=discrimination, n_samples=5, seed=0)

    assert seen_sizes  # predict was actually called by the sampling loop
    assert all(size == 0 for size in seen_sizes)
