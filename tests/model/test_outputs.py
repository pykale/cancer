import numpy as np
import pytest
import torch

from kalecancer.model import BreslowBaseline, ClassProbabilities, CoxOutput, RiskScore, VoteScores
from kalecancer.model.outputs import BaseOutput

CLASSES = ("low", "high")


def risk(values, ids=None, defined=None):
    ids = ids or [f"p{k}" for k in range(len(values))]
    defined = torch.ones(len(values), dtype=torch.bool) if defined is None else defined
    return RiskScore(ids=ids, defined=defined, risk=torch.tensor(values))


def probabilities(logits, ids=None, classes=CLASSES):
    ids = ids or [f"p{k}" for k in range(len(logits))]
    return ClassProbabilities(
        ids=ids, defined=torch.ones(len(logits), dtype=torch.bool), logits=torch.tensor(logits), classes=classes
    )


def fused(n=3):
    """A fused output with two branches, as LateFusion returns it."""
    out = risk([float(k) for k in range(n)])
    out.branches = {"a": risk([10.0 + k for k in range(n)]), "b": risk([20.0 + k for k in range(n)])}
    return out


# ---------------------------------------------------------------- select and scatter


def test_select_takes_rows_in_the_order_given_and_keeps_shared_fields():
    out = probabilities([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    picked = out.select(torch.tensor([2, 0]))
    assert picked.ids == ["p2", "p0"]
    torch.testing.assert_close(picked.logits, torch.tensor([[4.0, 5.0], [0.0, 1.0]]))
    assert picked.classes == CLASSES and picked.defined.tolist() == [True, True]


def test_select_takes_a_bool_mask_as_the_same_positions():
    out = risk([1.0, 2.0, 3.0])
    by_mask = out.select(torch.tensor([True, False, True]))
    by_position = out.select(torch.tensor([0, 2]))
    assert by_mask.ids == by_position.ids == ["p0", "p2"]
    torch.testing.assert_close(by_mask.risk, by_position.risk)


def test_select_and_scatter_apply_to_every_branch():
    picked = fused().select(torch.tensor([1, 2]))
    assert picked.branches["a"].ids == ["p1", "p2"]
    torch.testing.assert_close(picked.branches["b"].risk, torch.tensor([21.0, 22.0]))
    back = picked.scatter(torch.tensor([False, True, True]), ["p0", "p1", "p2"])
    assert back.branches["a"].defined.tolist() == [False, True, True]


def test_scatter_is_the_inverse_of_select_with_nan_rows_and_undefined_patients_elsewhere():
    mask = torch.tensor([True, False, True])
    out = probabilities([[0.0, 1.0], [4.0, 5.0]], ids=["p0", "p2"])
    full = out.scatter(mask, ["p0", "p1", "p2"])
    assert full.ids == ["p0", "p1", "p2"] and full.defined.tolist() == [True, False, True]
    assert torch.isnan(full.logits[1]).all() and full.classes == CLASSES
    torch.testing.assert_close(full.select(mask).logits, out.logits)


# ---------------------------------------------------------------- cat


def test_cat_joins_batches_and_their_branches():
    first, second = fused(2), fused(2)
    second = second.select(torch.tensor([1, 0]))
    joined = BaseOutput.cat([first, second])
    assert type(joined) is RiskScore and len(joined.ids) == 4
    torch.testing.assert_close(joined.risk, torch.tensor([0.0, 1.0, 1.0, 0.0]))
    torch.testing.assert_close(joined.branches["a"].risk, torch.tensor([10.0, 11.0, 11.0, 10.0]))


def test_cat_refuses_outputs_that_do_not_belong_together():
    with pytest.raises(ValueError, match="at least one output"):
        BaseOutput.cat([])
    with pytest.raises(ValueError, match="cannot concatenate a ClassProbabilities with a RiskScore"):
        BaseOutput.cat([risk([1.0]), probabilities([[0.0, 1.0]])])
    with pytest.raises(ValueError, match="outputs of branches"):
        BaseOutput.cat([fused(1), risk([1.0])])
    with pytest.raises(ValueError, match="different classes"):
        BaseOutput.cat([probabilities([[0.0, 1.0]]), probabilities([[0.0, 1.0]], classes=("high", "low"))])


def test_cat_refuses_outputs_with_different_baselines():
    # a baseline compares by identity: every batch of one model shares its endpoint's own
    fit = BreslowBaseline().fit(np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0]), np.array([True, False, True, True]))
    other = BreslowBaseline().fit(np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0]), np.array([True, True, True, True]))

    def cox(baseline):
        return CoxOutput(ids=["p0"], defined=torch.tensor([True]), risk=torch.tensor([0.0]), baseline=baseline)

    assert BaseOutput.cat([cox(fit), cox(fit)]).baseline is fit
    with pytest.raises(ValueError, match="different baseline"):
        BaseOutput.cat([cox(fit), cox(other)])


# ---------------------------------------------------------------- detach_cpu and views


def test_detach_cpu_detaches_casts_to_float32_and_keeps_defined_bool():
    logits = torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16, requires_grad=True)

    def make():
        return ClassProbabilities(ids=["p0"], defined=torch.tensor([True]), logits=logits * 2, classes=CLASSES)

    out = make()
    out.branches = {"a": make()}
    cpu = out.detach_cpu()
    assert cpu.logits.dtype == torch.float32 and not cpu.logits.requires_grad
    assert cpu.defined.dtype == torch.bool and cpu.branches["a"].logits.dtype == torch.float32
    assert cpu.frame().shape == (1, 2)


def test_frames_are_indexed_by_id_with_nan_for_undefined_patients():
    out = risk([1.0, 2.0]).scatter(torch.tensor([True, False, True]), ["p0", "p1", "p2"])
    frame = out.frame()
    assert frame.index.name == "id" and list(frame.columns) == ["risk"]
    assert np.isnan(frame.loc["p1", "risk"])
    votes = VoteScores(ids=["p0"], defined=torch.tensor([True]), votes=torch.tensor([[0.7, 0.3]]), classes=CLASSES)
    assert list(votes.frame().columns) == ["score[low]", "score[high]"]


def test_class_probabilities_scores_are_the_softmax_of_the_logits():
    out = probabilities([[0.0, 1.0], [2.0, -1.0]])
    torch.testing.assert_close(out.scores, torch.softmax(out.logits, dim=-1))
    torch.testing.assert_close(out.scores.sum(dim=-1), torch.ones(2))


def test_cox_survival_needs_the_fitted_baseline():
    out = CoxOutput(ids=["p0"], defined=torch.tensor([True]), risk=torch.tensor([0.0]))
    with pytest.raises(RuntimeError, match="fitted when training ends"):
        out.survival([1.0])


def test_only_outputs_on_one_scale_are_comparable_across_models():
    assert not RiskScore.comparable_across_models and not CoxOutput.comparable_across_models
    assert ClassProbabilities.comparable_across_models and VoteScores.comparable_across_models
    assert not BaseOutput.comparable_across_models
