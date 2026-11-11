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
import pandas as pd
import pytest
import torch
from torch import nn

from kalecancer.evaluate import HarrellC
from kalecancer.interpret import modality_shapley, patient_report, save_reports
from kalecancer.loaddata import MultimodalDataset, PatchFeatures, Tabular, TimeToEvent
from kalecancer.model import ABMIL, Concat, CoxHead, IntermediateFusion, LateFusion, MaskedMean, Unimodal


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


@pytest.fixture
def fitted_with_optional_clinical(cohort_factory, pipeline_factory):
    """Like ``fitted``, but ``clinical`` is not required, and 5 of the patients who do have ``wsi`` have no
    clinical row at all -- the configuration FINDING 9 is about: ``MaskedMean`` can still predict such a patient
    from ``wsi`` alone.
    """
    cohort = cohort_factory()
    clinical = Tabular(cohort.clinical.drop(index=cohort.clinical.index[25:30]))
    wsi = PatchFeatures(cohort.wsi_files, multiple_files="concatenate")
    target = TimeToEvent(cohort.time, cohort.event)
    data = MultimodalDataset({"clinical": clinical, "wsi": wsi}, target=target, required_modalities=["wsi"])
    train, test = data.subset(data.ids[:22]), data.subset(data.ids[22:])
    pipe = pipeline_factory(masked_mean_model(), optimizer=partial(torch.optim.AdamW, lr=0.0)).fit(train)
    return pipe, train, test


def test_a_singleton_request_lacking_the_optional_modality_completes_with_nan_contributions(
    fitted_with_optional_clinical,
):
    """Regression test for Copilot finding C1: ``patient_ids`` containing ONLY a patient who lacks the optional
    clinical modality narrows ``requested`` down to nobody present for it. A fitted ``StandardScaler`` has no
    defined output on zero rows and must not be asked to run on this narrowed view at all; the documented
    result is NaN, not a crash.
    """
    pipe, _train, test = fitted_with_optional_clinical
    without_clinical = next(pid for pid in test.ids if not test.present.loc[pid, "clinical"])

    [report] = patient_report(pipe, test, [without_clinical], n_feature_samples=5)

    assert report.feature_contributions.isna().all()
    assert report.metadata["patients_without_tabular_modality"] == [without_clinical]


def test_a_singleton_request_lacking_the_optional_modality_saves_and_round_trips(
    fitted_with_optional_clinical, tmp_path
):
    """Same singleton case as above, through save_reports: the narrowed view must not crash building the
    report in the first place, and the NaN contributions must still serialise and read back correctly.
    """
    pipe, _train, test = fitted_with_optional_clinical
    without_clinical = next(pid for pid in test.ids if not test.present.loc[pid, "clinical"])

    [report] = patient_report(pipe, test, [without_clinical], n_feature_samples=5)
    path = tmp_path / "singleton.json"
    save_reports([report], path)
    loaded = json.loads(path.read_text())

    assert loaded[0]["patient_id"] == without_clinical
    assert all(np.isnan(v) for v in loaded[0]["feature_contributions"].values())


def test_attention_narrows_to_zero_clinical_rows_but_still_produces_top_patches(fitted_with_optional_clinical):
    """``_top_patches`` narrows separately from the coalition prediction, to the patients who have the bag
    modality (``wsi``) -- here that narrowed subset is the same single patient, who also lacks ``clinical``.
    ``attention()`` applies every fitted transform to that subset, not just wsi's own encoder, so this must not
    crash scaling zero clinical rows either.
    """
    pipe, _train, test = fitted_with_optional_clinical
    without_clinical = next(pid for pid in test.ids if not test.present.loc[pid, "clinical"])
    assert test.present.loc[without_clinical, "wsi"]  # has the bag modality, so _top_patches does not short-circuit

    [report] = patient_report(pipe, test, [without_clinical], top_k=3)

    assert report.top_patches is not None
    assert len(report.top_patches) <= 3


def test_a_requested_patient_without_the_optional_modality_gets_nan_feature_contributions(
    fitted_with_optional_clinical,
):
    pipe, _train, test = fitted_with_optional_clinical
    with_clinical = next(pid for pid in test.ids if test.present.loc[pid, "clinical"])
    without_clinical = next(pid for pid in test.ids if not test.present.loc[pid, "clinical"])

    has_report, lacks_report = patient_report(pipe, test, [with_clinical, without_clinical], n_feature_samples=5)

    assert np.isfinite(has_report.feature_contributions.to_numpy()).all()
    assert has_report.feature_contributions[["sex", "smoking"]].abs().max() == pytest.approx(0.0, abs=1e-6)

    assert lacks_report.feature_contributions.isna().all()
    # the documented value (NaN) and the documented reason (this patient_id named in metadata), not a silently
    # dropped patient or an unreported count.
    assert lacks_report.metadata["patients_without_tabular_modality"] == [without_clinical]
    assert len(lacks_report.metadata["patients_without_tabular_modality"]) == 1


def test_a_mixed_request_still_gets_real_attention_for_the_patient_who_has_clinical(
    fitted_with_optional_clinical,
):
    """No regression from over-skipping: when the narrowed attention subset DOES include a patient with
    ``clinical`` present, ``_without_zero_row_transforms`` must leave that modality's real transform alone, not
    skip it just because another patient in the same request lacks it.
    """
    pipe, _train, test = fitted_with_optional_clinical
    with_clinical = next(pid for pid in test.ids if test.present.loc[pid, "clinical"])
    without_clinical = next(pid for pid in test.ids if not test.present.loc[pid, "clinical"])

    has_report, lacks_report = patient_report(pipe, test, [with_clinical, without_clinical], top_k=3)

    assert has_report.top_patches is not None
    assert len(has_report.top_patches) <= 3
    assert lacks_report.top_patches is not None
    assert len(lacks_report.top_patches) <= 3


def test_a_mixed_request_still_gets_real_feature_and_modality_contributions_for_the_patient_who_has_clinical(
    fitted_with_optional_clinical,
):
    """ITEM D2: the same "no regression from over-skipping" guarantee as the attention test above, for
    feature_contributions and modality_contributions specifically. In this two-patient fixture neither
    coalition's survivors subset nor the bag-modality subset ever ends up with nobody present for clinical --
    ``with_clinical`` is always among the survivors whenever anyone is -- so _without_zero_row_transforms's
    "skip" branch is never actually exercised here; this test passes before and after that fix, by construction,
    and is kept anyway as the explicit, by-name coverage ITEM D2 asked for. The genuine regression coverage for
    over-skipping is the singleton-request tests above, where the requested patient IS the only one present.
    """
    pipe, _train, test = fitted_with_optional_clinical
    with_clinical = next(pid for pid in test.ids if test.present.loc[pid, "clinical"])
    without_clinical = next(pid for pid in test.ids if not test.present.loc[pid, "clinical"])

    has_report, lacks_report = patient_report(pipe, test, [with_clinical, without_clinical], n_feature_samples=5)

    assert np.isfinite(has_report.feature_contributions.to_numpy()).all()
    assert np.isfinite(has_report.modality_contributions.to_numpy()).all()
    assert lacks_report.feature_contributions.isna().all()
    assert lacks_report.modality_contributions.isna().all()


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


def test_a_patient_missing_a_coalition_modality_gets_nan_not_an_error(fitted_with_optional_clinical, tmp_path):
    """Regression test for Copilot finding 7: a patient missing one of the coalition's modalities must come back
    as a reported NaN gap in modality_contributions, through patient_report AND save_reports, not a crash that
    aborts the whole report.
    """
    pipe, _train, test = fitted_with_optional_clinical
    with_clinical = next(pid for pid in test.ids if test.present.loc[pid, "clinical"])
    without_clinical = next(pid for pid in test.ids if not test.present.loc[pid, "clinical"])

    has_report, lacks_report = patient_report(pipe, test, [with_clinical, without_clinical], n_feature_samples=5)

    assert np.isfinite(has_report.modality_contributions.to_numpy()).all()
    assert lacks_report.modality_contributions.isna().all()

    path = tmp_path / "reports.json"
    save_reports([has_report, lacks_report], path)
    loaded = json.loads(path.read_text())
    assert loaded[1]["patient_id"] == without_clinical
    assert all(np.isnan(v) for v in loaded[1]["modality_contributions"].values())


class MeanPool(nn.Module):
    """Pools a bag by mean -- not an ``AttentionEncoder``, the configuration Copilot finding 8 is about."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim

    def forward(self, bags: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack([bag.mean(dim=0) for bag in bags])


def mean_pool_model() -> IntermediateFusion:
    """Same shape as ``masked_mean_model``, but wsi's bag encoder mean-pools instead of attending."""
    return IntermediateFusion(
        encoders={"clinical": [nn.Linear(3, 4)], "wsi": [MeanPool(16), nn.Linear(16, 4)]},
        fusion=MaskedMean(),
        head=CoxHead(in_dim=4, ties="efron"),
    )


def test_a_non_attention_bag_encoder_does_not_abort_the_report(
    cohort_factory, dataset_factory, split_factory, pipeline_factory, tmp_path
):
    cohort = cohort_factory()
    train, test = split_factory(dataset_factory(cohort))
    pipe = pipeline_factory(mean_pool_model()).fit(train)

    reports = patient_report(pipe, test, test.ids[:3], n_feature_samples=5)

    for report in reports:
        assert report.top_patches is None
        assert report.metadata["attention_unavailable_reason"] is not None
        assert "wsi" in report.metadata["attention_unavailable_reason"]
        # everything else in the report is still produced, not aborted by the unsupported bag encoder.
        assert np.isfinite(report.risk)
        assert np.isfinite(report.modality_contributions.to_numpy()).all()
        assert np.isfinite(report.feature_contributions.to_numpy()).all()

    path = tmp_path / "reports.json"
    save_reports(reports, path)
    loaded = json.loads(path.read_text())
    assert all(row["top_patches"] is None for row in loaded)


def _tabicl_config() -> dict:
    """A tiny, untrained TabICL config -- small enough to build and run on CPU in a test, no download needed."""
    return dict(
        max_classes=10,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        dropout=0.0,
    )


@pytest.fixture
def fitted_with_tabicl(cohort_factory, dataset_factory, split_factory, pipeline_factory, tmp_path_factory, monkeypatch):
    """``clinical`` is encoded by a ``TabICLEncoder`` on a tiny, untrained checkpoint -- the configuration
    FINDING D1 is about. ``MaskedMean`` (not ``Concat``) so the model clears ``_require_ablatable`` and
    ``patient_report`` reaches the TabICL check instead of failing on fusion ablatability first.
    """
    pytest.importorskip("tabicl")
    from tabicl._model.tabicl import TabICL

    import kalecancer.model.encode.tabicl as tabicl_module
    from kalecancer.model import TabICLEncoder

    config = _tabicl_config()
    torch.manual_seed(0)
    tiny = TabICL(**config)
    path = tmp_path_factory.mktemp("tabicl") / "tiny.ckpt"
    torch.save({"config": config, "state_dict": tiny.state_dict()}, path)
    monkeypatch.setattr(tabicl_module, "hf_hub_download", lambda *args, **kwargs: str(path))

    clinical_encoder = TabICLEncoder(
        checkpoint="tiny.ckpt", output="row", trainable=[], context_folds=2, random_state=0
    )
    width = config["embed_dim"] * config["row_num_cls"]
    model = IntermediateFusion(
        encoders={
            "clinical": [clinical_encoder, nn.Linear(width, 2)],
            "wsi": [ABMIL(in_dim=16, hidden_dim=6, attention_dim=4, dropout=0.0), nn.Linear(6, 2)],
        },
        fusion=MaskedMean(),
        head=CoxHead(in_dim=2, ties="efron"),
    )
    cohort = cohort_factory(n_patients=40, n_with_wsi=30, seed=0)
    train, test = split_factory(dataset_factory(cohort), n_train=22)
    pipe = pipeline_factory(model).fit(train)
    return pipe, train, test


def test_a_tabicl_encoded_modality_raises_the_explicit_limitation_through_patient_report(fitted_with_tabicl):
    """FINDING D1: the same TabICLEncoder limitation attribution.py's own functions reject must surface here
    too, through patient_report's own call, rather than failing deeper in with TabICLEncoder.forward's raw
    "rows differ from the context rows" error.
    """
    pipe, _train, test = fitted_with_tabicl

    with pytest.raises(ValueError, match="TabICLEncoder"):
        patient_report(pipe, test, [test.ids[0]])


def test_late_fusion_raises_the_explicit_limitation_error(
    cohort_factory, dataset_factory, split_factory, pipeline_factory
):
    cohort = cohort_factory()
    data = dataset_factory(cohort)
    rng = np.random.default_rng(0)
    lab = Tabular(
        pd.DataFrame(
            {"a": rng.standard_normal(len(cohort.ids)), "b": rng.standard_normal(len(cohort.ids))},
            index=pd.Index(cohort.ids, name="patient_id"),
        )
    )
    three = MultimodalDataset(
        {**data.modalities, "lab": lab},
        target=data.target,
        required_modalities=[*data.required_modalities, "lab"],
    )
    train, test = split_factory(three)
    model = LateFusion(
        branches={
            "clinical_lab": IntermediateFusion(
                encoders={"clinical": [nn.Linear(3, 4)], "lab": [nn.Linear(2, 4)]},
                fusion=Concat(),
                head=CoxHead(in_dim=8, ties="efron"),
            ),
            "wsi": Unimodal(
                "wsi",
                [ABMIL(in_dim=16, hidden_dim=6, attention_dim=4, dropout=0.0), nn.Linear(6, 4)],
                CoxHead(in_dim=4, ties="efron"),
            ),
        },
        fusion=MaskedMean(),
    )
    pipe = pipeline_factory(model).fit(train)

    with pytest.raises(ValueError, match="LateFusion"):
        patient_report(pipe, test, list(test.ids[:2]))
