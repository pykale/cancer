import math

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from kalecancer.loaddata import Classification, TimeToEvent
from kalecancer.model import (
    ABMIL,
    MLP,
    ClassificationHead,
    Concat,
    CoxHead,
    FusionContext,
    FusionMethod,
    InContextModule,
    MajorityVote,
    MaskedMean,
)

TARGET_IDS = pd.Index(["001", "002", "003"])
SURVIVAL = TimeToEvent(
    pd.Series([10.0, 20.0, 30.0], index=TARGET_IDS), pd.Series([True, False, True], index=TARGET_IDS)
)
BINARY = Classification(pd.Series(["low", "high", "low"], index=TARGET_IDS), classes=["low", "high"])


# ---------------------------------------------------------------- MLP and ABMIL


def test_mlp_without_hidden_layers_raises():
    with pytest.raises(ValueError, match="torch.nn.Linear"):
        MLP(in_dim=4, hidden_dims=[], out_dim=2, dropout=0.1)


def test_mlp_shape_and_stored_arguments():
    mlp = MLP(in_dim=4, hidden_dims=[8, 6], out_dim=3, dropout=0.0)
    assert mlp(torch.randn(5, 4)).shape == (5, 3)
    assert (mlp.in_dim, mlp.hidden_dims, mlp.out_dim, mlp.dropout) == (4, [8, 6], 3, 0.0)


def test_abmil_batch_equals_per_bag_and_attention_sums_to_one():
    torch.manual_seed(0)
    abmil = ABMIL(in_dim=8, hidden_dim=6, attention_dim=4, dropout=0.0).eval()
    bags = [torch.randn(n, 8) for n in (3, 17, 1)]
    batched = abmil(bags)
    single = torch.cat([abmil([bag]) for bag in bags])
    torch.testing.assert_close(batched, single)
    weights = abmil.attention(bags)
    assert [len(w) for w in weights] == [3, 17, 1]
    for w in weights:
        assert w.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_abmil_rejects_wrong_width_and_empty_bags():
    abmil = ABMIL(in_dim=8, hidden_dim=6, attention_dim=4, dropout=0.0)
    with pytest.raises(ValueError, match=r"\(N, 8\)"):
        abmil([torch.randn(3, 7)])
    with pytest.raises(ValueError, match=r"\(N, 8\)"):
        abmil([torch.randn(0, 8)])


# ---------------------------------------------------------------- in-context modules


class ContextSize(InContextModule):
    """Embeds each query as the number of context rows it sees."""

    out_dim = 1

    def __init__(self):
        super().__init__(context_label="event", context_folds=2, random_state=0)

    def embed(self, context, queries):
        return context.sum().expand(len(queries), 1).float()


EVENTS = {"event": torch.tensor([True, False] * 4)}
IDS = [f"p{k}" for k in range(8)]


def test_context_rows_see_the_context_without_their_fold_and_the_context_loads_into_a_fresh_module():
    x = torch.randn(8, 3)
    module = ContextSize()
    module.fit(x, EVENTS, IDS)
    assert module(torch.cat([x, torch.randn(1, 3)]), IDS + ["new"]).flatten().tolist() == [4.0] * 8 + [8.0]
    fresh = ContextSize()
    fresh.load_state_dict(module.state_dict())
    assert fresh(x, IDS).flatten().tolist() == [4.0] * 8


def test_in_context_guards():
    x = torch.randn(8, 3)
    module = ContextSize()
    with pytest.raises(ValueError, match="context_label must be 'event' or 'label'"):
        InContextModule(context_label="time", context_folds=2, random_state=0)
    with pytest.raises(RuntimeError, match="ContextSize.fit must be called before forward"):
        module(x, IDS)
    with pytest.raises(ValueError, match="needs a target with 'event'"):
        module.fit(x, {"label": torch.zeros(8)}, IDS)
    with pytest.raises(ValueError, match="aligned with 7 ids"):
        module.fit(x, EVENTS, IDS[:7])
    with pytest.raises(ValueError, match=r"classes \[1\] have fewer rows than context_folds=2"):
        module.fit(x, {"event": torch.tensor([True] + [False] * 7)}, IDS)
    module.fit(x, EVENTS, IDS)
    with pytest.raises(ValueError, match=r"expected x of shape \(8, 3\)"):
        module(x[:, :2], IDS)
    with pytest.raises(ValueError, match=r"ids \['p0'\] are context ids but their rows differ"):
        module(x + (torch.arange(8) == 0).float()[:, None], IDS)


# ---------------------------------------------------------------- Cox head


def reference_cox_nll(log_hazard, time, event, ties):
    """Naive negative partial log-likelihood (sum over events)."""
    total = 0.0
    for t in np.unique(time[event]):
        dead = event & (time == t)
        at_risk = time >= t
        d = dead.sum()
        risk = np.exp(log_hazard[at_risk]).sum()
        tied = np.exp(log_hazard[dead]).sum()
        total -= log_hazard[dead].sum()
        if ties == "breslow":
            total += d * math.log(risk)
        else:
            total += sum(math.log(risk - k / d * tied) for k in range(d))
    return total


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_cox_loss_matches_reference_with_ties(ties):
    rng = np.random.default_rng(0)
    n = 60
    time = rng.integers(1, 12, n).astype(np.float64)
    event = rng.random(n) < 0.6
    log_hazard = rng.normal(size=n)
    head = CoxHead(in_dim=1, ties=ties)
    loss = head.loss(
        torch.tensor(log_hazard, dtype=torch.float32).unsqueeze(1),
        {"time": torch.tensor(time, dtype=torch.float32), "event": torch.tensor(event)},
    )
    expected = reference_cox_nll(log_hazard, time, event, ties) / event.sum()
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_cox_loss_is_none_without_a_comparable_event():
    head = CoxHead(in_dim=1, ties="efron")
    output = torch.randn(4, 1)
    all_censored = {"time": torch.tensor([1.0, 2.0, 3.0, 4.0]), "event": torch.zeros(4, dtype=torch.bool)}
    only_last_dies = {"time": torch.tensor([1.0, 2.0, 3.0, 4.0]), "event": torch.tensor([False, False, False, True])}
    assert head.loss(output, all_censored) is None
    assert head.loss(output, only_last_dies) is None
    assert head.loss(output[:1], {"time": torch.tensor([1.0]), "event": torch.tensor([True])}) is None


def test_cox_loss_requires_bool_events_and_finite_values():
    head = CoxHead(in_dim=1, ties="efron")
    target = {"time": torch.tensor([1.0, 2.0]), "event": torch.tensor([1, 0])}
    with pytest.raises(TypeError, match="bool"):
        head.loss(torch.randn(2, 1), target)
    diverged = torch.tensor([[float("nan")], [0.0]])
    with pytest.raises(FloatingPointError):
        head.loss(diverged, {"time": torch.tensor([1.0, 2.0]), "event": torch.tensor([True, True])})


def test_cox_loss_is_computed_in_float32_under_autocast():
    torch.manual_seed(0)
    head = CoxHead(in_dim=3, ties="efron")
    z = torch.randn(32, 3) * 8
    target = {"time": torch.rand(32) * 100 + 1, "event": torch.rand(32) < 0.5}
    reference = head.loss(head(z), target)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = head.loss(head(z), target)
    assert mixed.dtype == torch.float32
    assert mixed.item() == pytest.approx(reference.item(), rel=1e-4)


def test_cox_head_checks_target_type():
    class Subclassed(TimeToEvent):
        pass

    head = CoxHead(in_dim=4, ties="efron")
    head.check_target(SURVIVAL)
    head.check_target(Subclassed(SURVIVAL.time, SURVIVAL.event))
    with pytest.raises(TypeError, match="time-to-event"):
        head.check_target(BINARY)
    assert head.columns(SURVIVAL) == ["log_hazard"]
    assert head.linear.bias is None


# ---------------------------------------------------------------- classification head


def test_classification_head_contract():
    head = ClassificationHead(in_dim=4, n_classes=2)
    logits = head(torch.randn(5, 4))
    labels = torch.tensor([0, 1, 1, 0, 1])
    assert head.loss(logits, {"label": labels}).item() == pytest.approx(F.cross_entropy(logits, labels).item())
    assert head.predict(logits).sum(dim=1).allclose(torch.ones(5))
    assert head.columns(BINARY) == ["probability[low]", "probability[high]"]
    with pytest.raises(ValueError, match="n_classes=2"):
        head.check_target(Classification(pd.Series(["a", "b", "c"], index=TARGET_IDS), classes=["a", "b", "c"]))
    with pytest.raises(TypeError, match="classification target"):
        head.check_target(SURVIVAL)


# ---------------------------------------------------------------- fusion methods


def scatter_nan(values: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    full = values.new_full((len(present), values.shape[1]), float("nan"))
    return full.index_put((present.nonzero().squeeze(1),), values)


def late_context(covered, kinds, width=1):
    present = pd.DataFrame(covered)
    return FusionContext("late", present, pd.Series(True, index=present.index), dict.fromkeys(present, width), kinds)


def test_concat_defines_only_patients_with_every_input():
    everyone = torch.ones(3, dtype=torch.bool)
    assert Concat()({"a": torch.randn(3, 2), "b": torch.randn(3, 4)}, {"a": everyone, "b": everyone}).shape == (3, 6)
    assert Concat().output_dim({"a": 2, "b": 4}) == 6
    present = torch.tensor([[True, True], [True, False], [False, False]])
    assert Concat().defined(present).tolist() == [True, False, False]
    assert MaskedMean().defined(present).tolist() == [True, True, False]


@pytest.mark.parametrize(
    ("method", "branch", "failing"),
    [(Concat, False, 2), (Concat, True, 1), (MaskedMean, False, 1), (MaskedMean, True, 0)],
)
def test_fusion_check_requires_the_patients_the_model_must_cover(method, branch, failing):
    present = pd.DataFrame({"clinical": [True, True, False], "wsi": [True, False, False]}, index=["001", "002", "003"])
    # a LateFusion branch must cover only the patients with some of its modalities
    required = present.any(axis=1) if branch else pd.Series(True, index=present.index)
    context = FusionContext("intermediate", present, required, {"clinical": 4, "wsi": 4})
    if failing:
        with pytest.raises(ValueError, match=f"cannot combine {failing} patients"):
            method().check(context)
    else:
        method().check(context)


def test_fusion_methods_reject_stages_they_do_not_support():
    classification = {"a": Classification, "b": Classification}
    with pytest.raises(TypeError, match="Concat is for early or intermediate fusion, not late fusion"):
        Concat().check(FusionContext.without_data("late", {"a": 2, "b": 2}, classification))
    with pytest.raises(TypeError, match="MajorityVote is for late fusion, not intermediate fusion"):
        MajorityVote(tie_break="error").check(FusionContext.without_data("intermediate", {"a": 2, "b": 2}))
    MaskedMean().check(FusionContext.without_data("early", {"a": None, "b": None}))
    MaskedMean().check(FusionContext.without_data("late", {"a": 2, "b": 2}, classification))


def test_fusion_method_subclasses_must_implement_defined():
    class Incomplete(FusionMethod):
        stages = frozenset({"intermediate"})

        def output_dim(self, widths):
            return 0

        def forward(self, values, present):
            return values

    with pytest.raises(TypeError, match=r"abstract method.*defined"):
        Incomplete()


def test_fusion_context_rejects_required_patients_indexed_unlike_present():
    present = pd.DataFrame({"a": [True, False]}, index=["001", "002"])
    with pytest.raises(ValueError, match="indexed like present"):
        FusionContext("intermediate", present, pd.Series(True, index=["001"]), {"a": 2})


def test_masked_mean_averages_present_rows_with_finite_gradients():
    torch.manual_seed(0)
    project = torch.nn.Linear(3, 2)
    present_b = torch.tensor([True, False, True, False])
    a = project(torch.randn(4, 3))
    b = scatter_nan(project(torch.randn(2, 3)), present_b)
    present = {"a": torch.ones(4, dtype=torch.bool), "b": present_b}
    fused = MaskedMean()({"a": a, "b": b}, present)
    assert torch.isfinite(fused).all()
    torch.testing.assert_close(fused[1], a[1])
    torch.testing.assert_close(fused[0], (a[0] + b[0]) / 2)
    fused.sum().backward()
    assert torch.isfinite(project.weight.grad).all()
    with pytest.raises(ValueError, match="one width"):
        MaskedMean().output_dim({"a": 2, "b": 3})
    with pytest.raises(ValueError, match="one width"):
        MaskedMean().check(FusionContext.without_data("intermediate", {"a": 2, "b": 3}))


def test_masked_mean_in_late_fusion_rejects_cox_heads_with_missing_branches():
    class Subclassed(TimeToEvent):
        pass

    cox = {"a": TimeToEvent, "b": TimeToEvent}
    MaskedMean().check(late_context({"a": [True, True], "b": [True, True]}, cox))
    with pytest.raises(ValueError, match="arbitrary offset"):
        MaskedMean().check(late_context({"a": [True, True], "b": [True, False]}, cox))
    with pytest.raises(ValueError, match="arbitrary offset"):
        MaskedMean().check(late_context({"a": [True, True], "b": [True, False]}, dict.fromkeys(cox, Subclassed)))
    classification = {"a": Classification, "b": Classification}
    MaskedMean().check(late_context({"a": [True, True], "b": [True, False]}, classification, width=2))
    with pytest.raises(TypeError, match="one kind"):
        MaskedMean().check(late_context({"a": [True], "b": [True]}, {"a": TimeToEvent, "b": Classification}))


def test_majority_vote_breaks_ties_by_mean_probability():
    p_a = torch.tensor([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
    p_b = torch.tensor([[0.3, 0.7], [0.4, 0.6], [0.1, 0.9]])
    defined = {"a": torch.ones(3, dtype=torch.bool), "b": torch.ones(3, dtype=torch.bool)}
    scores = MajorityVote(tie_break="mean_probability")({"a": p_a, "b": p_b}, defined)
    torch.testing.assert_close(scores.sum(dim=1), torch.ones(3))
    # patient 0: tie, mean probability favours class 0 (0.6 vs 0.4); patient 1: tie, mean 0.5/0.5 is exact
    assert scores[0].argmax().item() == 0
    assert scores[2].argmax().item() == 1
    with pytest.raises(ValueError, match="tied votes"):
        MajorityVote(tie_break="error")({"a": p_a, "b": p_b}, defined)


def test_majority_vote_uses_only_defined_branches_and_needs_classification_heads():
    p_a = torch.tensor([[0.9, 0.1]])
    p_b = torch.tensor([[float("nan"), float("nan")]])
    defined = {"a": torch.tensor([True]), "b": torch.tensor([False])}
    scores = MajorityVote(tie_break="error")({"a": p_a, "b": p_b}, defined)
    assert scores[0].argmax().item() == 0 and torch.isfinite(scores).all()

    class Subclassed(Classification):
        pass

    vote = MajorityVote(tie_break="error")
    vote.check(late_context({"a": [True], "b": [True]}, {"a": Classification, "b": Subclassed}, width=2))
    for kinds in ({"a": TimeToEvent, "b": TimeToEvent}, {"a": Classification, "b": None}):
        with pytest.raises(TypeError, match="classification heads"):
            vote.check(late_context({"a": [True], "b": [True]}, kinds, width=2))
