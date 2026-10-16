"""Tests for the per-patient interpretability bundle.

Two synthetic model setups, in the style of ``tests/interpret/test_attribution.py``:

* a 2-modality tabular model ("clinical", a known linear map of raw features read by
  a stand-in frozen TabICL embedder; "quiet", an embedder that ignores its input and
  always emits zero) -- for modality and feature contributions.
* a bag-modality model ("wsi" + "clinical") -- for the attention / top-patches path,
  where one of two patients has no bag at all.

The central thing under test, per the module's own critical distinction, is that
``patient_report``'s per-patient numbers are NOT the cohort-level
:func:`~kalecancer.interpret.modality.modality_shapley` numbers wearing a different
hat -- see ``test_per_patient_contributions_are_not_cohort_shapley_values``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from kalecancer.evaluate.survival_predictions import SplitPredictions
from kalecancer.interpret import modality_shapley, patient_report, save_reports
from kalecancer.loaddata.multimodal_access import PatientBatch
from kalecancer.model.embed import BagEncoder, MLPEmbedder
from kalecancer.model.embed.tabicl import TabICLEmbedder
from kalecancer.model.layers import AttentionMIL
from kalecancer.pipeline import CohortTrainer, SurvivalTask

FEATURES = ("age", "noise")
#: Only "age" (column 0) affects the model at all.
WEIGHTS_TRUE = np.array([5.0, 0.0])


class _FakeTabICLEmbedder(TabICLEmbedder):
    """A frozen TabICL stand-in needing no checkpoint: a known linear map of the raw
    features, so a feature's true contribution is known exactly."""

    _weights: torch.Tensor

    def __init__(self, weights: np.ndarray) -> None:
        nn.Module.__init__(self)  # bypass TabICLEmbedder.__init__: no checkpoint needed
        self.n_features = len(weights)
        self.out_dim = 1
        self.register_buffer("_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        return (x.float() @ self._weights).unsqueeze(-1)

    def train(self, mode: bool = True) -> _FakeTabICLEmbedder:
        nn.Module.train(self, mode)
        return self


class _ZeroEmbedder(nn.Module):
    """Ignores its input and always emits zero.

    Its *projected* embedding is not literally zero (the projection's bias term is
    not), so toggling it present/absent still moves the representation by some fixed
    amount -- but that amount is the same fixed amount for every patient, since
    nothing about it ever reads the patient's own data. That is the property these
    tests check: this modality's contribution carries no per-patient information,
    unlike a modality that actually reads the input."""

    def __init__(self, out_dim: int = 1) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.out_dim)


@dataclass
class _NanCoalitionStub:
    """A minimal model (not a real CohortTrainer) whose risk comes back entirely
    non-finite for exactly one modality coalition -- the same ``nan_for_coalition``
    trick as ``tests/interpret/test_modality.py``'s ``_SurvivalStub``, needed because
    reproducing the real degenerate case (e.g. a ProductOfExpertsFusion with no prior)
    numerically is not reliable to engineer in a unit test.
    """

    weights: dict[str, float]
    task: SurvivalTask = field(default_factory=SurvivalTask)
    training: bool = True
    nan_for_coalition: frozenset[str] | None = None

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

        if self.nan_for_coalition is not None:
            matches = torch.ones(len(batch.patient_id), dtype=torch.bool)
            for name in self.weights:
                is_present = batch.present[name].bool()
                should_be_present = name in self.nan_for_coalition
                matches &= is_present == should_be_present
            risk = torch.where(matches, torch.full_like(risk, float("nan")), risk)

        return SimpleNamespace(prediction=risk)


class _Loader:
    """A reusable iterable of batches exposing ``dataset.feature_names``, like a real
    ``DataLoader`` over a ``CohortView`` would."""

    class _Dataset:
        def __init__(self, feature_names: dict[str, list[str]]) -> None:
            self.feature_names = feature_names

    def __init__(self, batches: list[PatientBatch], feature_names: dict[str, list[str]] | None = None) -> None:
        self.batches = batches
        self.dataset = self._Dataset(feature_names or {})

    def __iter__(self):
        return iter(self.batches)


def make_tabular_batch(n: int = 20, seed: int = 0) -> PatientBatch:
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
            "quiet": torch.tensor(rng.standard_normal((n, 1)), dtype=torch.float32),
        },
        present={
            "clinical": torch.ones(n, dtype=torch.bool),
            "quiet": torch.ones(n, dtype=torch.bool),
        },
        target={
            "time": torch.tensor(times, dtype=torch.float32),
            "event": torch.tensor(events, dtype=torch.bool),
        },
    )


def build_tabular_model(seed: int = 0) -> CohortTrainer:
    torch.manual_seed(seed)
    return CohortTrainer(
        {"clinical": _FakeTabICLEmbedder(WEIGHTS_TRUE), "quiet": _ZeroEmbedder(out_dim=1)},
        task=SurvivalTask(),
        fusion_dim=4,
    )


def make_train_predictions(model: CohortTrainer, n: int = 100, seed: int = 50) -> SplitPredictions:
    batch = make_tabular_batch(n=n, seed=seed)
    risk = model.predict(batch).prediction.reshape(-1)
    return SplitPredictions(
        split="train",
        patient_ids=batch.patient_id,
        risk=risk,
        duration=batch.target["time"],
        event=batch.target["event"],
    )


FEATURE_DIM = 8
WSI_MODALITY = "wsi"


def make_bag_batch() -> PatientBatch:
    """Two patients: "p0" has a WSI bag, "p1" does not."""
    bags = [torch.randn(5, FEATURE_DIM), torch.randn(9, FEATURE_DIM)]
    return PatientBatch(
        patient_id=["p0", "p1"],
        modalities={WSI_MODALITY: bags, "clinical": torch.randn(2, 3)},
        present={
            WSI_MODALITY: torch.tensor([True, False]),
            "clinical": torch.ones(2, dtype=torch.bool),
        },
        metadata={
            "coords": [torch.arange(n * 2).reshape(n, 2) for n in (5, 9)],
            "slide_index": [torch.zeros(n, dtype=torch.long) for n in (5, 9)],
            "slide_ids": [("s",), ("s",)],
        },
        target={"time": torch.tensor([500.0, 900.0]), "event": torch.tensor([1, 0], dtype=torch.bool)},
    )


def build_bag_model(seed: int = 0) -> CohortTrainer:
    torch.manual_seed(seed)
    return CohortTrainer(
        {
            WSI_MODALITY: BagEncoder(AttentionMIL(input_dim=FEATURE_DIM, hidden_dim=8, attention_dim=4)),
            "clinical": MLPEmbedder(3, 8),
        },
        task=SurvivalTask(),
        fusion_dim=8,
    )


def test_driving_modality_dominates_the_quiet_one() -> None:
    model = build_tabular_model()
    loader = _Loader([make_tabular_batch(seed=1)], feature_names={"clinical": list(FEATURES)})

    reports = patient_report(model, loader, ["000", "001", "002"])

    quiet_values = [report.modality_contributions["quiet"] for report in reports]
    clinical_values = [report.modality_contributions["clinical"] for report in reports]

    # "quiet" never reads its input, so its contribution carries no per-patient
    # information -- it is the same fixed, architecture-driven constant for every
    # patient -- while "clinical" varies with each patient's own data.
    assert max(quiet_values) - min(quiet_values) == pytest.approx(0.0, abs=1e-5)
    assert max(clinical_values) - min(clinical_values) > 0.1
    for report in reports:
        assert abs(report.modality_contributions["clinical"]) > abs(report.modality_contributions["quiet"])


def test_per_patient_contributions_are_not_cohort_shapley_values() -> None:
    model = build_tabular_model()
    loader = _Loader([make_tabular_batch(seed=2)], feature_names={"clinical": list(FEATURES)})

    cohort_values = modality_shapley(model, loader)
    reports = patient_report(model, loader, ["000", "001"])

    # Different patients (different "age") must get different per-patient numbers --
    # a cohort-level value is one fixed number, shared by construction.
    assert reports[0].modality_contributions["clinical"] != reports[1].modality_contributions["clinical"]
    # Neither patient's bundle is simply the cohort dict copied in.
    assert reports[0].modality_contributions != cohort_values
    assert reports[1].modality_contributions != cohort_values


def test_unknown_patient_id_is_refused_by_name() -> None:
    model = build_tabular_model()
    loader = _Loader([make_tabular_batch(seed=3)])

    with pytest.raises(ValueError, match="not_a_real_patient"):
        patient_report(model, loader, ["not_a_real_patient"])


def test_missing_train_predictions_gives_empty_curve_with_a_reason() -> None:
    model = build_tabular_model()
    loader = _Loader([make_tabular_batch(seed=4)])

    reports = patient_report(model, loader, ["000"])

    assert reports[0].survival_times.size == 0
    assert reports[0].survival_probs.size == 0
    assert reports[0].metadata["survival_curve_computable"] is False
    assert reports[0].metadata["survival_curve_unavailable_reason"]


def test_survival_curve_is_computed_when_given_both_train_predictions_and_eval_times() -> None:
    model = build_tabular_model()
    loader = _Loader([make_tabular_batch(seed=5)])
    train_predictions = make_train_predictions(model)

    reports = patient_report(model, loader, ["000"], train_predictions=train_predictions, eval_times=[200.0, 400.0])

    assert reports[0].survival_times.tolist() == [200.0, 400.0]
    assert reports[0].survival_probs.shape == (2,)
    assert reports[0].metadata["survival_curve_computable"] is True
    assert reports[0].metadata["survival_curve_unavailable_reason"] is None


def test_patient_without_a_bag_modality_gets_none_top_patches() -> None:
    model = build_bag_model()
    loader = [make_bag_batch()]

    reports = patient_report(model, loader, ["p0", "p1"])

    assert reports[0].top_patches is not None
    assert reports[1].top_patches is None
    # Attention WAS available in this call -- p1's None is "this patient individually
    # has no bag", not "nothing here could produce attention at all".
    assert reports[0].metadata["top_patches_unavailable_reason"] is None
    assert reports[1].metadata["top_patches_unavailable_reason"] is None


def test_top_patches_reason_when_no_bag_modality_on_the_model() -> None:
    model = build_tabular_model()
    loader = _Loader([make_tabular_batch(seed=23)], feature_names={"clinical": list(FEATURES)})

    reports = patient_report(model, loader, ["000", "001"])

    for report in reports:
        assert report.top_patches is None
        assert report.metadata["top_patches_unavailable_reason"]
        assert "bag-pooling modality" in report.metadata["top_patches_unavailable_reason"]


def test_top_patches_reason_when_batch_metadata_is_missing() -> None:
    model = build_bag_model()
    batch = make_bag_batch()
    batch.metadata = {}  # no coords/slide_index/slide_ids: attention cannot be placed

    reports = patient_report(model, [batch], ["p0", "p1"])

    for report in reports:
        assert report.top_patches is None
        assert report.metadata["top_patches_unavailable_reason"]
        assert "metadata" in report.metadata["top_patches_unavailable_reason"]


def test_save_reports_round_trips(tmp_path) -> None:
    model = build_tabular_model()
    loader = _Loader([make_tabular_batch(seed=6)], feature_names={"clinical": list(FEATURES)})
    reports = patient_report(model, loader, ["000", "001"])

    out_path = tmp_path / "reports.json"
    save_reports(reports, out_path)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))

    assert len(loaded) == 2
    for original, round_tripped in zip(reports, loaded, strict=True):
        assert round_tripped["patient_id"] == original.patient_id
        assert round_tripped["risk"] == pytest.approx(original.risk)
        assert round_tripped["risk_percentile"] == pytest.approx(original.risk_percentile)
        assert round_tripped["survival_times"] == original.survival_times.tolist()
        assert round_tripped["survival_probs"] == original.survival_probs.tolist()
        assert round_tripped["modality_contributions"] == pytest.approx(original.modality_contributions)
        assert round_tripped["feature_contributions"] == pytest.approx(original.feature_contributions)
        assert round_tripped["top_patches"] == original.top_patches
        assert round_tripped["metadata"] == original.metadata


def test_feature_contributions_are_deterministic_for_a_fixed_seed() -> None:
    model = build_tabular_model()
    loader = _Loader([make_tabular_batch(seed=7)], feature_names={"clinical": list(FEATURES)})

    first = patient_report(model, loader, ["000", "001"], seed=11, n_feature_samples=20)
    second = patient_report(model, loader, ["000", "001"], seed=11, n_feature_samples=20)

    assert [r.feature_contributions for r in first] == [r.feature_contributions for r in second]


def test_training_mode_is_restored_after_patient_report() -> None:
    model = build_tabular_model()
    model.train()
    loader = _Loader([make_tabular_batch(seed=8)], feature_names={"clinical": list(FEATURES)})

    patient_report(model, loader, ["000"])

    assert model.training is True


def test_empty_modality_coalition_fallback_is_used_and_recorded(caplog: pytest.LogCaptureFixture) -> None:
    n = 6
    batch = PatientBatch(
        patient_id=[f"{i:03d}" for i in range(n)],
        modalities={
            "a": torch.arange(n, dtype=torch.float32).unsqueeze(-1),
            "b": torch.arange(n, dtype=torch.float32).unsqueeze(-1) * 2,
        },
        present={"a": torch.ones(n, dtype=torch.bool), "b": torch.ones(n, dtype=torch.bool)},
        target={"time": torch.linspace(100.0, 500.0, n), "event": torch.ones(n, dtype=torch.bool)},
    )
    model = _NanCoalitionStub({"a": 1.0, "b": 0.3}, nan_for_coalition=frozenset())

    with caplog.at_level(logging.WARNING):
        reports = patient_report(model, [batch], batch.patient_id[:2])

    assert any("empty" in record.message and "non-finite" in record.message for record in caplog.records)
    assert all(report.metadata["empty_modality_coalition_fallback_used"] is True for report in reports)


def _numel(value: torch.Tensor | list[torch.Tensor]) -> int:
    assert isinstance(value, torch.Tensor)
    return value.numel()


def test_caching_drops_the_other_modalitys_raw_tensor_and_restores_it() -> None:
    from kalecancer.interpret.report import _cache_other_modalities, _materialise, _restore_modalities

    model = build_tabular_model()
    loader = _Loader([make_tabular_batch(seed=21, n=30)])

    full_batch = _materialise(loader)
    original_quiet = full_batch.modalities["quiet"]
    assert _numel(original_quiet) > 0

    original_embedders, original_modalities = _cache_other_modalities(model, full_batch, "clinical")

    assert full_batch.modalities["quiet"] is not original_quiet
    assert _numel(full_batch.modalities["quiet"]) == 0
    assert original_modalities["quiet"] is original_quiet

    _restore_modalities(model, full_batch, original_embedders, original_modalities)

    assert full_batch.modalities["quiet"] is original_quiet


def test_feature_contributions_never_hand_predict_the_full_other_modality_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kalecancer.interpret.report import _feature_contributions, _materialise, _subset_batch

    model = build_tabular_model()
    loader = _Loader([make_tabular_batch(seed=22, n=30)])
    full_batch = _materialise(loader)
    requested_batch = _subset_batch(full_batch, [0, 1, 2])

    seen_sizes: list[int] = []
    original_predict = model.predict

    def spying_predict(batch):
        seen_sizes.append(_numel(batch.modalities["quiet"]))
        return original_predict(batch)

    monkeypatch.setattr(model, "predict", spying_predict)

    tabular_pool = full_batch.modalities["clinical"]
    assert isinstance(tabular_pool, torch.Tensor)
    _feature_contributions(
        model,
        tabular_pool,
        requested_batch,
        "clinical",
        list(FEATURES),
        n_samples=5,
        seed=0,
    )

    assert seen_sizes
    assert all(size == 0 for size in seen_sizes)
