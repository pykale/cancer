"""Tests for permutation_importance and feature_shapley.

The fitted pipeline's clinical encoder and Cox head are hand-built, with a learning rate of 0 so training leaves
them untouched (as in ``test_pipeline.test_fit_starts_from_the_weights_the_model_was_built_with``): the clinical
encoder reads only ``age`` (zero weight on ``sex``/``smoking``), and the head reads only the clinical embedding
(zero weight on the wsi branch). That makes ``log_hazard`` an exact, deterministic function of ``age`` alone --
sex/smoking get exactly zero importance by construction, and age's importance does not depend on how training
happened to converge (with everything frozen at its random initial value, nothing converges at all; a wsi branch
with an untrained, randomly initialised combining weight would otherwise make age's sign and rank a coin flip).
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from kalecancer.evaluate import HarrellC, IntegratedBrier
from kalecancer.interpret import feature_shapley, permutation_importance
from kalecancer.loaddata import MultimodalDataset, PatchFeatures, Tabular, TimeToEvent
from kalecancer.model import ABMIL, Concat, CoxHead, IntermediateFusion, MaskedMean


def _inner_times(data, quantiles=(0.2, 0.4, 0.6)):
    """Times inside the follow-up of ``data``'s patients, where survival metrics are defined."""
    return [float(t) for t in np.quantile(data.target.frame.loc[data.ids, "time"], quantiles)]


def informative_model() -> IntermediateFusion:
    """``log_hazard`` is an exact, deterministic function of ``age`` alone.

    The clinical encoder reads only the first (scaled ``age``) column of the transformed frame -- ``sex``/
    ``smoking`` get exactly zero weight. The Cox head then reads only the clinical embedding -- the wsi branch
    gets exactly zero weight too, so its randomly initialised (and, with ``lr=0``, never trained) weights cannot
    affect which feature the model turns out to rely on.
    """
    clinical_encoder = nn.Linear(3, 1)
    with torch.no_grad():
        clinical_encoder.weight[:] = torch.tensor([[1.0, 0.0, 0.0]])
        clinical_encoder.bias.zero_()
    model = IntermediateFusion(
        encoders={
            "clinical": [clinical_encoder],
            "wsi": [ABMIL(in_dim=16, hidden_dim=6, attention_dim=4, dropout=0.0), nn.Linear(6, 1)],
        },
        fusion=Concat(),
        head=CoxHead(in_dim=2, ties="efron"),
    )
    with torch.no_grad():
        model.head.linear.weight[:] = torch.tensor([[1.0, 0.0]])
    return model


def masked_mean_model() -> IntermediateFusion:
    """``log_hazard`` is an exact, deterministic function of ``age`` alone, like ``informative_model``, but fused
    with ``MaskedMean`` rather than ``Concat`` so a patient missing the (here optional) clinical modality can
    still be scored from wsi alone -- wsi's own embedding is still a constant zero, same as ``informative_model``.
    """
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
def fitted_with_optional_clinical(cohort_factory, pipeline_factory):
    """Like ``fitted``, but ``clinical`` is not required, and 10 of the patients who do have ``wsi`` have no
    clinical row at all -- the configuration FINDING 2 is about: a dataset where the Tabular modality is not in
    ``required_modalities`` and a ``MaskedMean``-style fusion can still predict a patient who lacks it.
    """
    cohort = cohort_factory(n_patients=150, n_with_wsi=120, seed=0)
    clinical = Tabular(cohort.clinical.drop(index=cohort.clinical.index[110:120]))
    wsi = PatchFeatures(cohort.wsi_files, multiple_files="concatenate")
    target = TimeToEvent(cohort.time, cohort.event)
    data = MultimodalDataset({"clinical": clinical, "wsi": wsi}, target=target, required_modalities=["wsi"])
    train, test = data.subset(data.ids[:90]), data.subset(data.ids[90:])
    pipe = pipeline_factory(masked_mean_model(), optimizer=partial(torch.optim.AdamW, lr=0.0)).fit(train)
    return pipe, train, test


def test_a_patient_without_the_optional_tabular_modality_is_scored_but_not_perturbed(fitted_with_optional_clinical):
    pipe, _train, test = fitted_with_optional_clinical
    missing = int((~test.present["clinical"]).sum())
    assert missing == 10

    importance = permutation_importance(pipe, test, HarrellC(), n_repeats=3, seed=0)
    assert importance.attrs["n_excluded"] == missing

    shapley = feature_shapley(pipe, test, HarrellC(), n_samples=5, seed=0)
    assert shapley.attrs["n_excluded"] == missing

    # the scored cohort -- what the baseline and every perturbed coalition is evaluated over -- is the FULL test
    # set (30 patients), unchanged by restricting which rows get shuffled; it must not have silently shrunk to
    # just the 20 patients who have the modality.
    baseline_direct = pipe.evaluate(test, {"c": HarrellC()})["c"]
    assert importance["baseline"].iloc[0] == pytest.approx(baseline_direct)


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
def tabicl_model(tmp_path_factory, monkeypatch):
    """``clinical`` is encoded by a ``TabICLEncoder`` on a tiny, untrained checkpoint -- the configuration
    FINDING D1 is about. No network access and no pretrained weights needed: the same tiny-checkpoint pattern
    ``tests/test_tabicl.py`` uses, built fresh here since that file's fixtures are module-scoped and not meant
    to be imported across test files.
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
    return IntermediateFusion(
        encoders={
            "clinical": [clinical_encoder, nn.Linear(config["embed_dim"] * config["row_num_cls"], 1)],
            "wsi": [ABMIL(in_dim=16, hidden_dim=6, attention_dim=4, dropout=0.0), nn.Linear(6, 1)],
        },
        fusion=Concat(),
        head=CoxHead(in_dim=2, ties="efron"),
    )


@pytest.fixture
def fitted_with_tabicl(cohort_factory, dataset_factory, split_factory, pipeline_factory, tabicl_model):
    cohort = cohort_factory(n_patients=40, n_with_wsi=30, seed=0)
    train, test = split_factory(dataset_factory(cohort), n_train=22)
    pipe = pipeline_factory(tabicl_model).fit(train)
    return pipe, train, test


def test_a_tabicl_encoded_modality_raises_the_explicit_limitation_naming_tabicl(fitted_with_tabicl):
    """FINDING D1: perturbing a TabICLEncoder-encoded modality would hit TabICLEncoder.forward's id/value
    mismatch check (for a context row) or silently rely on it never firing (for an unseen one) -- neither is a
    real attribution. Both functions must name the limitation up front instead of failing deeper in, or
    (for an unseen patient) silently computing something never verified against a genuine counterfactual.
    """
    pipe, _train, test = fitted_with_tabicl

    with pytest.raises(ValueError, match="TabICLEncoder"):
        permutation_importance(pipe, test, HarrellC(), n_repeats=2)
    with pytest.raises(ValueError, match="TabICLEncoder"):
        feature_shapley(pipe, test, HarrellC(), n_samples=2)


def test_an_ordinary_tabular_encoder_is_unaffected_by_the_tabicl_check(fitted):
    """Same call shape as the TabICL test above, but on `fitted`'s ordinary nn.Linear clinical encoder: the
    TabICLEncoder-specific check must not raise for a model that has no TabICLEncoder at all.
    """
    pipe, _train, test = fitted

    importance = permutation_importance(pipe, test, HarrellC(), n_repeats=2)
    assert np.isfinite(importance[["mean", "std", "baseline"]].to_numpy()).all()
    shapley = feature_shapley(pipe, test, HarrellC(), n_samples=2)
    assert np.isfinite(shapley.to_numpy()).all()


def integer_columns_model() -> IntermediateFusion:
    """Reads a ``clinical`` Tabular modality with pandas' default integer columns, the configuration FINDING A2
    is about.
    """
    return IntermediateFusion(
        encoders={
            "clinical": [nn.Linear(2, 2)],
            "wsi": [ABMIL(in_dim=16, hidden_dim=6, attention_dim=4, dropout=0.0), nn.Linear(6, 2)],
        },
        fusion=Concat(),
        head=CoxHead(in_dim=4, ties="efron"),
    )


@pytest.fixture
def fitted_with_integer_columns(cohort_factory, pipeline_factory):
    """``clinical``'s frame has pandas' default integer columns (``0``, ``1``), not named ones -- already
    numeric, so no transform is needed or given.
    """
    cohort = cohort_factory()
    age = cohort.clinical["age"].to_numpy()
    clinical = Tabular(pd.DataFrame({0: age, 1: (age > 60).astype(float)}, index=cohort.clinical.index))
    wsi = PatchFeatures(cohort.wsi_files, multiple_files="concatenate")
    target = TimeToEvent(cohort.time, cohort.event)
    data = MultimodalDataset({"clinical": clinical, "wsi": wsi}, target=target, required_modalities=["clinical", "wsi"])
    train, test = data.subset(data.ids[:22]), data.subset(data.ids[22:])
    pipe = pipeline_factory(integer_columns_model(), transforms={}).fit(train)
    return pipe, train, test


def test_default_integer_columns_work_for_both_attribution_functions(fitted_with_integer_columns):
    pipe, _train, test = fitted_with_integer_columns

    importance = permutation_importance(pipe, test, HarrellC(), n_repeats=2)
    assert list(importance.index) == [0, 1]
    assert np.isfinite(importance[["mean", "std", "baseline"]].to_numpy()).all()

    shapley = feature_shapley(pipe, test, HarrellC(), n_samples=2)
    assert list(shapley.index) == [0, 1]
    assert np.isfinite(shapley.to_numpy()).all()


@pytest.fixture
def fitted(cohort_factory, dataset_factory, split_factory, pipeline_factory):
    # Bigger than the shared `cohort` fixture's default 40/30: age's signal, isolated from the (here irrelevant)
    # wsi branch, needs enough held-out patients for its true positive effect on Harrell's C to not be swamped by
    # small-sample noise -- 8 test patients (the shared fixture's default split) was not enough to reliably tell
    # the two apart.
    cohort = cohort_factory(n_patients=150, n_with_wsi=120, seed=0)
    train, test = split_factory(dataset_factory(cohort), n_train=90)
    pipe = pipeline_factory(informative_model(), optimizer=partial(torch.optim.AdamW, lr=0.0)).fit(train)
    return pipe, train, test


def test_informative_feature_ranks_first_and_noise_features_are_exactly_zero(fitted):
    pipe, _train, test = fitted

    importance = permutation_importance(pipe, test, HarrellC(), n_repeats=5, seed=0)
    assert importance["mean"].idxmax() == "age"
    assert importance.loc["age", "mean"] > 0.01
    np.testing.assert_allclose(importance.loc[["sex", "smoking"], ["mean", "std"]].to_numpy(), 0.0, atol=1e-9)

    shapley = feature_shapley(pipe, test, HarrellC(), n_samples=10, seed=0)
    assert shapley.idxmax() == "age"
    assert shapley["age"] > 0.0
    np.testing.assert_allclose(shapley[["sex", "smoking"]].to_numpy(), 0.0, atol=1e-9)


def test_deterministic_for_a_fixed_seed_and_a_different_seed_changes_the_value_not_the_rank(fitted):
    pipe, _train, test = fitted

    first = permutation_importance(pipe, test, HarrellC(), n_repeats=5, seed=0)
    again = permutation_importance(pipe, test, HarrellC(), n_repeats=5, seed=0)
    pd.testing.assert_frame_equal(first, again)
    other_seed = permutation_importance(pipe, test, HarrellC(), n_repeats=5, seed=1)
    assert other_seed.loc["age", "mean"] != pytest.approx(first.loc["age", "mean"])
    assert other_seed["mean"].idxmax() == "age"

    first_shapley = feature_shapley(pipe, test, HarrellC(), n_samples=10, seed=0)
    again_shapley = feature_shapley(pipe, test, HarrellC(), n_samples=10, seed=0)
    pd.testing.assert_series_equal(first_shapley, again_shapley)
    other_seed_shapley = feature_shapley(pipe, test, HarrellC(), n_samples=10, seed=1)
    assert other_seed_shapley["age"] != pytest.approx(first_shapley["age"])
    assert other_seed_shapley.idxmax() == "age"


def test_n_repeats_changes_std_but_not_materially_the_mean(fitted):
    pipe, _train, test = fitted

    few = permutation_importance(pipe, test, HarrellC(), n_repeats=3, seed=0)
    many = permutation_importance(pipe, test, HarrellC(), n_repeats=30, seed=0)
    assert few.loc["age", "mean"] == pytest.approx(many.loc["age", "mean"], abs=0.05)
    assert few.loc["age", "std"] != pytest.approx(many.loc["age", "std"])


def test_modality_discovery_raises_naming_what_it_found(fitted):
    pipe, _train, test = fitted

    no_tabular = MultimodalDataset({"wsi": test.modalities["wsi"]}, target=test.target, required_modalities=["wsi"])
    with pytest.raises(ValueError, match=r"found \[\] of \['wsi'\]"):
        permutation_importance(pipe, no_tabular, HarrellC())
    with pytest.raises(ValueError, match=r"found \[\] of \['wsi'\]"):
        feature_shapley(pipe, no_tabular, HarrellC())

    two_tabular = MultimodalDataset(
        {**test.modalities, "demographics": Tabular(test.modalities["clinical"].frame)},
        target=test.target,
        required_modalities=list(test.required_modalities),
    )
    with pytest.raises(ValueError, match=r"found \['clinical', 'demographics'\]"):
        permutation_importance(pipe, two_tabular, HarrellC())
    with pytest.raises(ValueError, match=r"found \['clinical', 'demographics'\]"):
        feature_shapley(pipe, two_tabular, HarrellC())


def test_sign_is_flipped_for_a_lower_is_better_metric(fitted):
    pipe, _train, test = fitted
    times = _inner_times(test)

    # IntegratedBrier has higher_is_better=False (lower is better): shuffling the informative feature must still
    # be reported as a POSITIVE drop -- this is the whole reason the sign flip exists.
    importance = permutation_importance(pipe, test, IntegratedBrier(), n_repeats=5, seed=0, survival_times=times)
    assert importance.loc["age", "mean"] > 0.0

    shapley = feature_shapley(pipe, test, IntegratedBrier(), n_samples=10, seed=0, survival_times=times)
    assert shapley["age"] > 0.0


def test_a_survival_column_metric_without_survival_times_raises_naming_the_parameter(fitted):
    pipe, _train, test = fitted

    with pytest.raises(ValueError, match="survival_times"):
        permutation_importance(pipe, test, IntegratedBrier())
    with pytest.raises(ValueError, match="survival_times"):
        feature_shapley(pipe, test, IntegratedBrier())


def test_allow_seen_default_false_raises_on_training_patients_allow_seen_true_works(fitted):
    pipe, train, _test = fitted

    with pytest.raises(ValueError, match="90 patients were used to fit"):
        permutation_importance(pipe, train, HarrellC(), n_repeats=2)
    importance = permutation_importance(pipe, train, HarrellC(), n_repeats=2, allow_seen=True)
    assert not importance.empty

    with pytest.raises(ValueError, match="90 patients were used to fit"):
        feature_shapley(pipe, train, HarrellC(), n_samples=2)
    shapley = feature_shapley(pipe, train, HarrellC(), n_samples=2, allow_seen=True)
    assert not shapley.empty
