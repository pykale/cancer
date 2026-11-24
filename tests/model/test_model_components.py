import math

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kalecancer.loaddata import Classification, TimeToEvent
from kalecancer.model import (
    ABMIL,
    MLP,
    ClassificationEndpoint,
    ClassProbabilities,
    Concat,
    CoxEndpoint,
    CoxOutput,
    EncoderList,
    FittedEncoder,
    MajorityVote,
    MaskedMean,
    OutputFusion,
    RiskScore,
    TensorFusion,
    VoteScores,
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


# ---------------------------------------------------------------- fitted encoders


class Centre(FittedEncoder):
    """Subtracts the training mean, kept in a buffer whose size depends on the data."""

    mean: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.empty(0))

    def fit(self, x, target, ids):
        self.mean = x.mean(dim=0)

    def forward(self, x, ids):
        return x - self.mean


def test_a_fitted_encoder_loads_into_a_fresh_one():
    x = torch.randn(8, 3)
    fitted = Centre()
    fitted.fit(x, SURVIVAL, [f"p{k}" for k in range(8)])
    fresh = Centre()
    fresh.load_state_dict(fitted.state_dict())
    torch.testing.assert_close(fresh(x, []), x - x.mean(dim=0))


# ---------------------------------------------------------------- encoder lists


class NanToZero(torch.nn.Module):
    """An encoder that handles missing values itself."""

    def forward(self, x):
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class BagMean(torch.nn.Module):
    def forward(self, bags):
        return torch.stack([bag.mean(dim=0) for bag in bags])


class Log(torch.nn.Module):
    def forward(self, x):
        return torch.log(x)


IDS = ["001", "002", "003", "004"]


def test_encoder_list_rejects_non_finite_embeddings_and_blames_the_input():
    x = torch.randn(4, 3)
    x[1, 0], x[3, 2] = float("nan"), float("inf")
    output = r"encoders\['clinical'\] output NaN or infinite values for 2 patient\(s\) \(e\.g\. \['002', '004'\]\)"
    cause = r"input already has NaN or infinite values for 2 patient\(s\) \(e\.g\. \['002', '004'\]\)"
    with pytest.raises(ValueError, match=f"{output}.*{cause}"):
        EncoderList([torch.nn.Linear(3, 2)], "clinical")(x, IDS)
    # with no encoders the input is the embedding, and fusion and heads do not accept NaN either
    with pytest.raises(ValueError, match=output):
        EncoderList([], "clinical")(x, IDS)


def test_an_encoder_that_handles_nan_needs_no_flag():
    x = torch.randn(4, 3)
    x[1, 0] = float("nan")
    assert torch.isfinite(EncoderList([NanToZero(), torch.nn.Linear(3, 2)], "clinical")(x, IDS)).all()
    # only the embedding is checked, so an encoder later in the list may handle NaN too
    assert torch.isfinite(EncoderList([torch.nn.Linear(3, 3), NanToZero()], "clinical")(x, IDS)).all()


def test_encoder_list_blames_the_encoders_when_the_input_is_finite():
    with pytest.raises(ValueError, match="The input was finite, so the encoders produced these values"):
        EncoderList([Log()], "clinical")(-torch.rand(4, 3) - 1, IDS)


def test_encoder_list_names_the_patient_whose_bag_has_nan():
    bags = [torch.randn(5, 3), torch.randn(7, 3)]
    bags[1][2, 1] = float("nan")
    with pytest.raises(ValueError, match=r"encoders\['wsi'\] .* for 1 patient\(s\) \(e\.g\. \['002'\]\)"):
        EncoderList([BagMean()], "wsi")(bags, ["001", "002"])


class Fail(torch.nn.Module):
    def forward(self, x):
        raise RuntimeError("should not run")


def test_steps_yield_each_output_and_stopping_early_skips_the_rest():
    encoders = EncoderList([torch.nn.Linear(3, 4), torch.nn.ReLU(), Fail()], "clinical")
    x = torch.randn(4, 3)
    seen = []
    for k, encoder, out in encoders.steps(x, IDS):
        seen.append((k, type(encoder).__name__, tuple(out.shape)))
        if k == 1:
            break
    assert seen == [(0, "Linear", (4, 4)), (1, "ReLU", (4, 4))]
    with pytest.raises(RuntimeError, match="should not run") as error:
        encoders(x, IDS)
    assert error.value.__notes__ == ["in encoders['clinical'][2] Fail, called on a tensor of shape (4, 4)"]


def test_encoder_list_output_must_be_one_floating_point_row_per_patient():
    with pytest.raises(ValueError, match=r"one row per patient: 4 patients, 3 rows"):
        EncoderList([], "clinical")(torch.randn(3, 2), IDS)
    with pytest.raises(ValueError, match=r"must output an \(n, d\) tensor, got shape \(4, 2, 2\)"):
        EncoderList([], "clinical")(torch.randn(4, 2, 2), IDS)
    with pytest.raises(TypeError, match="must output a floating-point tensor, got torch.int64"):
        EncoderList([], "clinical")(torch.ones(4, 2, dtype=torch.long), IDS)


# ---------------------------------------------------------------- Cox endpoint


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


def cox_output(log_hazard):
    log_hazard = torch.as_tensor(log_hazard, dtype=torch.float32).reshape(-1)
    return CoxOutput(
        ids=[str(k) for k in range(len(log_hazard))],
        defined=torch.ones(len(log_hazard), dtype=torch.bool),
        risk=log_hazard,
    )


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_cox_loss_matches_reference_with_ties(ties):
    rng = np.random.default_rng(0)
    n = 60
    time = rng.integers(1, 12, n).astype(np.float64)
    event = rng.random(n) < 0.6
    log_hazard = rng.normal(size=n)
    endpoint = CoxEndpoint(nn.Linear(1, 1, bias=False), ties=ties)
    loss = endpoint.loss(
        cox_output(log_hazard), {"time": torch.tensor(time, dtype=torch.float32), "event": torch.tensor(event)}
    )
    expected = reference_cox_nll(log_hazard, time, event, ties) / event.sum()
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_cox_loss_is_none_without_a_comparable_event():
    endpoint = CoxEndpoint(nn.Linear(1, 1, bias=False), "efron")
    output = cox_output(torch.randn(4))
    all_censored = {"time": torch.tensor([1.0, 2.0, 3.0, 4.0]), "event": torch.zeros(4, dtype=torch.bool)}
    only_last_dies = {"time": torch.tensor([1.0, 2.0, 3.0, 4.0]), "event": torch.tensor([False, False, False, True])}
    assert endpoint.loss(output, all_censored) is None
    assert endpoint.loss(output, only_last_dies) is None
    one = {"time": torch.tensor([1.0]), "event": torch.tensor([True])}
    assert endpoint.loss(output.select(torch.tensor([0])), one) is None


def test_cox_loss_requires_bool_events_and_finite_values():
    endpoint = CoxEndpoint(nn.Linear(1, 1, bias=False), "efron")
    target = {"time": torch.tensor([1.0, 2.0]), "event": torch.tensor([1, 0])}
    with pytest.raises(TypeError, match="bool"):
        endpoint.loss(cox_output(torch.randn(2)), target)
    diverged = cox_output([float("nan"), 0.0])
    with pytest.raises(FloatingPointError):
        endpoint.loss(diverged, {"time": torch.tensor([1.0, 2.0]), "event": torch.tensor([True, True])})


def test_cox_loss_is_computed_in_float32_under_autocast():
    torch.manual_seed(0)
    endpoint = CoxEndpoint(nn.Linear(3, 1, bias=False), "efron")
    z = torch.randn(32, 3) * 8
    target = {"time": torch.rand(32) * 100 + 1, "event": torch.rand(32) < 0.5}
    ids, everyone = [str(k) for k in range(32)], torch.ones(32, dtype=torch.bool)
    reference = endpoint.loss(endpoint.to_output(endpoint(z), ids, everyone), target)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = endpoint.loss(endpoint.to_output(endpoint(z), ids, everyone), target)
    assert mixed.dtype == torch.float32
    assert mixed.item() == pytest.approx(reference.item(), rel=1e-4)


def test_cox_endpoint_checks_target_type_and_builds_cox_outputs():
    class Subclassed(TimeToEvent):
        pass

    endpoint = CoxEndpoint(nn.Linear(4, 1, bias=False), "efron")
    endpoint.check_target(SURVIVAL)
    endpoint.check_target(Subclassed(SURVIVAL.time, SURVIVAL.event))
    with pytest.raises(TypeError, match="time-to-event"):
        endpoint.check_target(BINARY)
    with pytest.raises(ValueError, match="ties"):
        CoxEndpoint(nn.Linear(4, 1, bias=False), "exact")
    output = endpoint.to_output(endpoint(torch.randn(3, 4)), list(TARGET_IDS), torch.ones(3, dtype=torch.bool))
    assert isinstance(output, CoxOutput) and output.risk.shape == (3,)
    assert list(output.detach_cpu().frame().columns) == ["log_hazard"]
    assert endpoint.output_type is CoxOutput and not endpoint.decomposable_loss


# ---------------------------------------------------------------- classification endpoint


def test_classification_endpoint_contract():
    endpoint = ClassificationEndpoint(nn.Linear(4, 2), nn.CrossEntropyLoss())
    with pytest.raises(TypeError, match="classification target"):
        endpoint.check_target(SURVIVAL)
    endpoint.check_target(BINARY)
    endpoint.bind(BINARY)
    ids = [str(k) for k in range(5)]
    output = endpoint.to_output(endpoint(torch.randn(5, 4)), ids, torch.ones(5, dtype=torch.bool))
    labels = torch.tensor([0, 1, 1, 0, 1])
    loss = endpoint.loss(output, {"label": labels})
    assert loss.item() == pytest.approx(F.cross_entropy(output.logits, labels).item())
    assert output.probabilities.sum(dim=1).allclose(torch.ones(5))
    assert list(output.detach_cpu().frame().columns) == ["probability[low]", "probability[high]"]
    assert endpoint.output_type is ClassProbabilities


def test_classification_endpoint_takes_any_torch_loss_and_declares_one_column_per_class():
    endpoint = ClassificationEndpoint(nn.Linear(4, 2), nn.CrossEntropyLoss(label_smoothing=0.1), decomposable_loss=True)
    endpoint.bind(BINARY)
    assert endpoint.out_dim == 2 and CoxEndpoint(nn.Linear(4, 1, bias=False), "efron").out_dim == 1
    assert endpoint.decomposable_loss
    assert not ClassificationEndpoint(nn.Linear(4, 2), nn.CrossEntropyLoss()).decomposable_loss


# ---------------------------------------------------------------- fusion methods


def scatter_nan(values: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    full = values.new_full((len(present), values.shape[1]), float("nan"))
    return full.index_put((present.nonzero().squeeze(1),), values)


def test_concat_defines_only_patients_with_every_input():
    everyone = torch.ones(3, dtype=torch.bool)
    assert Concat()({"a": torch.randn(3, 2), "b": torch.randn(3, 4)}, {"a": everyone, "b": everyone}).shape == (3, 6)
    present = torch.tensor([[True, True], [True, False], [False, False]])
    assert Concat().defined(present).tolist() == [True, False, False]
    assert MaskedMean().defined(present).tolist() == [True, True, False]


@pytest.mark.parametrize(
    ("method", "branch", "failing"),
    [(Concat, False, 2), (Concat, True, 1), (MaskedMean, False, 1), (MaskedMean, True, 0)],
)
def test_fusion_methods_define_the_patients_a_model_can_cover(method, branch, failing):
    present = pd.DataFrame({"clinical": [True, True, False], "wsi": [True, False, False]}, index=["001", "002", "003"])
    # a LateFusion branch must cover only the patients with some of its modalities
    required = present.any(axis=1) if branch else pd.Series(True, index=present.index)
    defined = method().defined_rows(present)
    assert defined.index.equals(present.index)
    assert int((required & ~defined).sum()) == failing


def test_fusion_methods_declare_their_stages_by_the_interfaces_they_implement():
    classification = {"a": ClassProbabilities, "b": ClassProbabilities}
    assert isinstance(Concat(), TensorFusion) and not isinstance(Concat(), OutputFusion)
    vote = MajorityVote(tie_break="error")
    assert isinstance(vote, OutputFusion) and not isinstance(vote, TensorFusion)
    assert isinstance(MaskedMean(), TensorFusion) and isinstance(MaskedMean(), OutputFusion)
    MaskedMean().check_widths({"a": 2, "b": 2})
    assert MaskedMean().output_type(classification) is ClassProbabilities


def test_fusion_method_subclasses_must_implement_defined():
    class Incomplete(TensorFusion):
        def forward(self, values, present):
            return values

    with pytest.raises(TypeError, match=r"abstract method.*defined"):
        Incomplete()


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
        MaskedMean().check_widths({"a": 2, "b": 3})


def test_masked_mean_in_late_fusion_rejects_relative_risks_with_missing_branches():
    cox = {"a": CoxOutput, "b": CoxOutput}
    complete = pd.DataFrame({"a": [True, True], "b": [True, True]})
    incomplete = pd.DataFrame({"a": [True, True], "b": [True, False]})
    MaskedMean().check_branches(complete, cox)
    with pytest.raises(ValueError, match="arbitrary offset"):
        MaskedMean().check_branches(incomplete, cox)
    with pytest.raises(ValueError, match="arbitrary offset"):
        MaskedMean().check_branches(incomplete, dict.fromkeys(cox, RiskScore))
    MaskedMean().check_branches(incomplete, {"a": ClassProbabilities, "b": ClassProbabilities})
    assert MaskedMean().output_type(cox) is RiskScore
    with pytest.raises(TypeError, match="one output type"):
        MaskedMean().output_type({"a": CoxOutput, "b": ClassProbabilities})


def probabilities(p):
    """Class probabilities ``p`` as a branch output: their logs are logits whose softmax is ``p``."""
    return ClassProbabilities(
        ids=[str(k) for k in range(len(p))],
        defined=torch.ones(len(p), dtype=torch.bool),
        logits=p.log(),
        classes=(0, 1),
    )


def test_majority_vote_breaks_ties_by_mean_probability():
    p_a = probabilities(torch.tensor([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]]))
    p_b = probabilities(torch.tensor([[0.3, 0.7], [0.4, 0.6], [0.1, 0.9]]))
    defined = {"a": torch.ones(3, dtype=torch.bool), "b": torch.ones(3, dtype=torch.bool)}
    voted = MajorityVote(tie_break="mean_probability")({"a": p_a, "b": p_b}, defined)
    assert type(voted) is VoteScores and voted.classes == (0, 1) and voted.ids == p_a.ids
    scores = voted.scores
    torch.testing.assert_close(scores.sum(dim=1), torch.ones(3))
    # patient 0: tie, mean probability favours class 0 (0.6 vs 0.4); patient 1: tie, mean 0.5/0.5 is exact
    assert scores[0].argmax().item() == 0
    assert scores[2].argmax().item() == 1
    with pytest.raises(ValueError, match="tied votes"):
        MajorityVote(tie_break="error")({"a": p_a, "b": p_b}, defined)


def test_majority_vote_uses_only_defined_branches_and_needs_class_probabilities():
    p_a = probabilities(torch.tensor([[0.9, 0.1]]))
    p_b = probabilities(torch.tensor([[float("nan"), float("nan")]]))
    defined = {"a": torch.tensor([True]), "b": torch.tensor([False])}
    scores = MajorityVote(tie_break="error")({"a": p_a, "b": p_b}, defined).scores
    assert scores[0].argmax().item() == 0 and torch.isfinite(scores).all()

    class Subclassed(ClassProbabilities):
        pass

    vote = MajorityVote(tie_break="error")
    assert vote.output_type({"a": ClassProbabilities, "b": Subclassed}) is VoteScores
    for kinds in ({"a": CoxOutput, "b": CoxOutput}, {"a": ClassProbabilities, "b": VoteScores}):
        with pytest.raises(TypeError, match="class probabilities"):
            vote.output_type(kinds)
