"""Tests for patient_report and save_reports.

The fitted pipeline's clinical encoder, wsi encoder and Cox head are hand-built exactly as in test_modality.py:
``log_hazard`` is an exact, deterministic function of ``age`` alone -- the clinical encoder has zero weight on
``sex``/``smoking``, the wsi branch's final layer is zeroed to a constant, and the head reads only the clinical
half of the fused embedding. That makes wsi's and sex/smoking's contributions exactly zero and age/clinical's
exactly what drives the prediction, so the "isolates the informative source" assertions are exact.
"""

from __future__ import annotations

import json
from functools import partial

import numpy as np
import pytest
import torch
from torch import nn

from kalecancer.evaluate import HarrellC
from kalecancer.interpret import modality_shapley, patient_report, save_reports
from kalecancer.model import ABMIL, CoxHead, IntermediateFusion, MaskedMean


def masked_mean_model() -> IntermediateFusion:
    """``log_hazard`` is an exact, deterministic function of ``age`` alone; wsi always contributes a constant zero."""
    clinical_encoder = nn.Linear(3, 2)
    with torch.no_grad():
        clinical_encoder.weight[:] = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        clinical_encoder.bias.zero_()
    wsi_final = nn.Linear(6, 2)
    with torch.no_grad():
        wsi_final.weight.zero_()
        wsi_final.bias.zero_()
    model = IntermediateFusion(
        encoders={
            "clinical": [clinical_encoder],
            "wsi": [ABMIL(in_dim=16, hidden_dim=6, attention_dim=4, dropout=0.0), wsi_final],
        },
        fusion=MaskedMean(),
        head=CoxHead(in_dim=2, ties="efron"),
    )
    with torch.no_grad():
        model.head.linear.weight[:] = torch.tensor([[1.0, 0.0]])
    return model


@pytest.fixture
def fitted(cohort_factory, dataset_factory, split_factory, pipeline_factory):
    cohort = cohort_factory()
    train, test = split_factory(dataset_factory(cohort))
    pipe = pipeline_factory(masked_mean_model(), optimizer=partial(torch.optim.AdamW, lr=0.0)).fit(train)
    return pipe, train, test


def test_feature_contributions_isolate_the_informative_feature(fitted):
    pipe, _train, test = fitted

    reports = patient_report(pipe, test, test.ids[:3])

    for report in reports:
        # sex/smoking never reach the clinical encoder's nonzero weight, so their risk delta is exactly zero;
        # age does, and drives the prediction, so it is not. (Unlike feature perturbation, ablating "wsi" here
        # is NOT a clean zero-vs-nonzero split: MaskedMean divides by however many modalities are present, so
        # even a modality whose own embedding is a constant changes the OTHER branch's effective weight when
        # it is marked absent -- a real rescaling effect, not an attribution bug. See the other tests below for
        # what is actually checked about modality_contributions.)
        assert report.feature_contributions[["sex", "smoking"]].abs().max() == pytest.approx(0.0, abs=1e-6)
        assert abs(report.feature_contributions["age"]) > 1e-6
        assert set(report.modality_contributions.index) == {"clinical", "wsi"}
        assert np.isfinite(report.modality_contributions.to_numpy()).all()


def test_per_patient_contributions_differ_from_the_cohort_level_numbers(fitted):
    pipe, _train, test = fitted

    [report] = patient_report(pipe, test, [test.ids[0]])
    cohort = modality_shapley(pipe, test, HarrellC())

    # a per-patient risk delta and a cohort-level metric delta are different numbers answering different
    # questions -- they must not come out equal (let alone be the same number relabelled).
    assert report.modality_contributions["clinical"] != pytest.approx(cohort["clinical"])


def test_risk_percentile_is_computed_against_the_full_cohort_regardless_of_who_else_is_requested(fitted):
    pipe, _train, test = fitted
    pid = test.ids[0]

    [alone] = patient_report(pipe, test, [pid])
    [together, _other] = patient_report(pipe, test, [pid, test.ids[1]])

    assert alone.risk == pytest.approx(together.risk)
    assert alone.risk_percentile == pytest.approx(together.risk_percentile)


def test_allow_seen_default_false_raises_on_training_patients_allow_seen_true_works(fitted):
    pipe, train, _test = fitted

    with pytest.raises(ValueError, match="were used to fit this pipeline"):
        patient_report(pipe, train, [train.ids[0]])
    [report] = patient_report(pipe, train, [train.ids[0]], allow_seen=True)
    assert np.isfinite(report.risk)


def test_unknown_patient_id_raises_naming_it(fitted):
    pipe, _train, test = fitted

    with pytest.raises(ValueError, match=r"\['not-a-real-id'\]"):
        patient_report(pipe, test, ["not-a-real-id"])


def test_survival_is_none_without_survival_times_and_populated_with_them(fitted):
    pipe, _train, test = fitted
    pid = test.ids[0]

    [without] = patient_report(pipe, test, [pid])
    assert without.survival is None

    times = sorted(float(t) for t in np.quantile(test.target.frame.loc[test.ids, "time"], [0.2, 0.4, 0.6]))
    [with_times] = patient_report(pipe, test, [pid], survival_times=times)
    assert list(with_times.survival["time"]) == times
    assert with_times.survival["survival"].between(0.0, 1.0).all()
    assert (with_times.survival["survival"].diff().dropna() <= 0).all()


def test_top_patches_has_at_most_top_k_rows_sorted_by_attention(fitted):
    pipe, _train, test = fitted

    [report] = patient_report(pipe, test, [test.ids[0]], top_k=3)

    assert report.top_patches is not None
    assert len(report.top_patches) <= 3
    assert (report.top_patches["attention"].diff().dropna() <= 0).all()


def test_save_reports_round_trips_to_readable_json(fitted, tmp_path):
    pipe, _train, test = fitted
    reports = patient_report(pipe, test, test.ids[:2])
    path = tmp_path / "reports.json"

    save_reports(reports, path)
    loaded = json.loads(path.read_text())

    assert [row["patient_id"] for row in loaded] == [r.patient_id for r in reports]
    assert loaded[0]["modality_contributions"].keys() == {"clinical", "wsi"}
    assert loaded[0]["risk"] == pytest.approx(reports[0].risk)
    assert loaded[0]["survival"] is None
