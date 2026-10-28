"""Tests for leave_one_out_ablation and modality_shapley.

The fitted pipeline's clinical encoder, wsi encoder and Cox head are all hand-built, with a learning rate of 0 so
training leaves them untouched (as in test_attribution.py): the clinical branch reads only ``age``, the wsi
branch's final layer is zeroed so its embedding is always the constant zero vector -- no information at all,
regardless of what ABMIL computes upstream of it -- and the head's own combining weight reads only the clinical
half of the fused (``MaskedMean``) embedding. That makes wsi's contribution to the ranking exactly zero and
clinical's exactly what drives ``log_hazard``, so the "informative vs. uninformative modality" assertions are
exact rather than merely probable.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from kalecancer.evaluate import HarrellC
from kalecancer.interpret import leave_one_out_ablation, modality_shapley
from kalecancer.loaddata import MultimodalDataset, Tabular
from kalecancer.model import ABMIL, Concat, CoxHead, IntermediateFusion, LateFusion, MaskedMean, Unimodal
from kalecancer.pipeline import Pipeline


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


def concat_model() -> IntermediateFusion:
    return IntermediateFusion(
        encoders={
            "clinical": [nn.Linear(3, 2)],
            "wsi": [ABMIL(in_dim=16, hidden_dim=6, attention_dim=4, dropout=0.0), nn.Linear(6, 2)],
        },
        fusion=Concat(),
        head=CoxHead(in_dim=4, ties="efron"),
    )


@pytest.fixture
def fitted(cohort_factory, dataset_factory, split_factory, pipeline_factory):
    # Bigger than cohort_factory()'s default 40/30: age's real effect on Harrell's C, isolated from the (here
    # constant-zero) wsi branch, needs enough held-out patients for its true positive effect not to be swamped by
    # small-sample noise -- the default split was not enough to reliably tell it from a null result.
    cohort = cohort_factory(n_patients=150, n_with_wsi=120, seed=0)
    train, test = split_factory(dataset_factory(cohort), n_train=90)
    pipe = pipeline_factory(masked_mean_model(), optimizer=partial(torch.optim.AdamW, lr=0.0)).fit(train)
    return pipe, train, test


def test_uninformative_modality_is_near_zero_and_informative_is_clearly_positive(fitted):
    pipe, _train, test = fitted

    ablation = leave_one_out_ablation(pipe, test, HarrellC())
    assert ablation["wsi"] == pytest.approx(0.0, abs=1e-9)
    assert ablation["clinical"] > 0.1

    shapley = modality_shapley(pipe, test, HarrellC())
    assert shapley["wsi"] == pytest.approx(0.0, abs=1e-9)
    assert shapley["clinical"] > 0.1


def test_shapley_values_sum_to_the_documented_efficiency_statement(fitted):
    pipe, _train, test = fitted

    shapley = modality_shapley(pipe, test, HarrellC())
    ablation = leave_one_out_ablation(pipe, test, HarrellC())

    # v(empty) is defined by extrapolation from each modality's leave-one-out drop (see modality_shapley's
    # docstring), so the Shapley values sum to exactly the total of those drops.
    assert shapley.sum() == pytest.approx(ablation.sum())


def test_three_modalities_scores_exactly_8_coalitions(
    cohort_factory, dataset_factory, split_factory, pipeline_factory, monkeypatch
):
    cohort = cohort_factory()
    data = dataset_factory(cohort)
    rng = np.random.default_rng(0)
    demographics = Tabular(
        pd.DataFrame({"noise": rng.standard_normal(len(cohort.ids))}, index=pd.Index(cohort.ids, name="patient_id"))
    )
    three = MultimodalDataset(
        {**data.modalities, "demographics": demographics},
        target=data.target,
        required_modalities=[*data.required_modalities, "demographics"],
    )
    train, test = split_factory(three)
    model = IntermediateFusion(
        encoders={
            "clinical": [nn.Linear(3, 2)],
            "wsi": [ABMIL(in_dim=16, hidden_dim=6, attention_dim=4, dropout=0.0), nn.Linear(6, 2)],
            "demographics": [nn.Linear(1, 2)],
        },
        fusion=MaskedMean(),
        head=CoxHead(in_dim=2, ties="efron"),
    )
    pipe = pipeline_factory(model).fit(train)

    calls = []
    original_evaluate = Pipeline.evaluate

    def counting_evaluate(self, *args, **kwargs):
        calls.append(1)
        return original_evaluate(self, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "evaluate", counting_evaluate)

    result = modality_shapley(pipe, test, HarrellC())

    assert set(result.index) == {"clinical", "wsi", "demographics"}
    assert result.attrs["n_coalitions"] == 2**3 == 8
    # every NONEMPTY coalition is scored exactly once; the empty one is a documented convention (0), never evaluated
    assert len(calls) == 2**3 - 1 == 7


def test_concat_fusion_raises_naming_the_fusion_method(
    cohort_factory, dataset_factory, split_factory, pipeline_factory
):
    cohort = cohort_factory()
    train, test = split_factory(dataset_factory(cohort))
    pipe = pipeline_factory(concat_model()).fit(train)

    with pytest.raises(ValueError, match="Concat"):
        leave_one_out_ablation(pipe, test, HarrellC())
    with pytest.raises(ValueError, match="Concat"):
        modality_shapley(pipe, test, HarrellC())


def test_max_modalities_exceeded_names_the_coalition_count(fitted):
    pipe, _train, test = fitted

    with pytest.raises(ValueError, match=r"2\*\*2 - 1 = 3"):
        modality_shapley(pipe, test, HarrellC(), max_modalities=1)


def test_allow_seen_default_false_raises_on_training_patients_allow_seen_true_works(fitted):
    pipe, train, _test = fitted

    with pytest.raises(ValueError, match="90 patients were used to fit"):
        leave_one_out_ablation(pipe, train, HarrellC())
    ablation = leave_one_out_ablation(pipe, train, HarrellC(), allow_seen=True)
    assert not ablation.empty

    with pytest.raises(ValueError, match="90 patients were used to fit"):
        modality_shapley(pipe, train, HarrellC())
    shapley = modality_shapley(pipe, train, HarrellC(), allow_seen=True)
    assert not shapley.empty


def test_the_evaluated_cohort_is_identical_across_coalitions(
    cohort_factory, dataset_factory, split_factory, pipeline_factory
):
    cohort = cohort_factory()
    # required=("clinical",) lets patients without wsi (the last 10 of 40, per make_cohort) into the dataset, so
    # some of the test split's patients naturally lack a modality -- exactly the case comparability must handle.
    data = dataset_factory(cohort, required=("clinical",))
    train, test = split_factory(data, n_train=25)
    n_lacking_wsi = int((~test.present["wsi"]).sum())
    assert n_lacking_wsi > 0
    pipe = pipeline_factory(masked_mean_model(), optimizer=partial(torch.optim.AdamW, lr=0.0)).fit(train)

    ablation = leave_one_out_ablation(pipe, test, HarrellC())

    # only patients with BOTH modalities are defined under every coalition (baseline, ablate-clinical,
    # ablate-wsi) scored here, so exactly those lacking wsi are excluded from the comparison.
    assert ablation.attrs["n_excluded"] == n_lacking_wsi


def test_modality_shapley_with_a_single_selected_modality_satisfies_efficiency(fitted):
    pipe, _train, test = fitted

    # modalities=["clinical"] on this two-modality model: the "background" coalition (clinical absent, wsi --
    # the unselected modality -- still present) is a real, scorable coalition, not the architecturally-empty
    # one, so this must not crash looking up a score it never computed.
    single = modality_shapley(pipe, test, HarrellC(), modalities=["clinical"])
    assert set(single.index) == {"clinical"}
    assert np.isfinite(single["clinical"])

    # for k=1, the Shapley value reduces to exactly one term: v(full) - v(empty). That is the same quantity
    # leave_one_out_ablation reports for "clinical" against the same real background, so the documented
    # efficiency statement (sum(shapley) == v(full) - v(empty)) holds as this equality.
    ablation = leave_one_out_ablation(pipe, test, HarrellC(), modalities=["clinical"])
    assert single["clinical"] == pytest.approx(ablation["clinical"])


def late_fusion_model() -> LateFusion:
    """A ``Concat`` clinical+lab branch alongside a unimodal wsi branch -- the configuration FINDING 5 is about:
    the outer ``MaskedMean`` fusion combines branch PREDICTIONS, not raw modality presence, so a modality-level
    presence check cannot tell whether the clinical_lab branch (which needs BOTH clinical and lab) can actually
    predict for a given coalition.
    """
    return LateFusion(
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


def test_late_fusion_is_rejected_naming_the_limitation(
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
    pipe = pipeline_factory(late_fusion_model()).fit(train)

    with pytest.raises(ValueError, match="LateFusion"):
        leave_one_out_ablation(pipe, test, HarrellC())
    with pytest.raises(ValueError, match="LateFusion"):
        modality_shapley(pipe, test, HarrellC())
