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
