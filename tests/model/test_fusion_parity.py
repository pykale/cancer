"""Fixed numerical contracts of the fusion methods, whose tensor operations PyKale supplies."""

import pickle

import pytest
import torch

from kalecancer.model import ClassProbabilities, Concat, FusionMethod, MajorityVote, MaskedMean, VoteScores
from kalecancer.model.fusion import _masked_mean, _stack_defined


def _probabilities(p: torch.Tensor) -> ClassProbabilities:
    """A branch output whose probabilities are ``p``: log-probabilities are logits whose softmax is ``p``."""
    return ClassProbabilities(
        ids=[f"p{k}" for k in range(len(p))], defined=~p.isnan().any(dim=1), logits=p.log(), classes=("a", "b")
    )


def _vote(method: MajorityVote, values: dict, present: dict) -> torch.Tensor:
    output = method({name: _probabilities(p) for name, p in values.items()}, present)
    assert isinstance(output, VoteScores)
    return output.votes


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_concat_preserves_mapping_order_and_gradients(dtype):
    first = torch.tensor([[2.0, 3.0]], dtype=dtype, requires_grad=True)
    second = torch.tensor([[7.0]], dtype=dtype, requires_grad=True)
    present = {"second": torch.tensor([True]), "first": torch.tensor([True])}
    result = Concat()({"second": second, "first": first}, present)
    torch.testing.assert_close(result, torch.tensor([[7.0, 2.0, 3.0]], dtype=dtype))
    result.sum().backward()
    torch.testing.assert_close(first.grad, torch.ones_like(first))
    torch.testing.assert_close(second.grad, torch.ones_like(second))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_mean_fixed_outputs_and_gradients(dtype):
    first = torch.tensor([[2.0, 4.0], [6.0, 8.0]], dtype=dtype, requires_grad=True)
    second = torch.tensor([[10.0, 12.0], [float("nan"), float("nan")]], dtype=dtype, requires_grad=True)
    present = {"a": torch.tensor([True, True]), "b": torch.tensor([True, False])}
    result = MaskedMean()({"a": first, "b": second}, present)
    torch.testing.assert_close(result, torch.tensor([[6.0, 8.0], [6.0, 8.0]], dtype=dtype))
    result.sum().backward()
    torch.testing.assert_close(first.grad, torch.tensor([[0.5, 0.5], [1.0, 1.0]], dtype=dtype))
    torch.testing.assert_close(second.grad, torch.tensor([[0.5, 0.5], [0.0, 0.0]], dtype=dtype))


def test_empty_row_is_zero_only_at_helper_level():
    values = torch.full((2, 1, 3), float("nan"), requires_grad=True)
    mask = torch.zeros(2, 1, dtype=torch.bool)
    result = _masked_mean(values, mask)
    torch.testing.assert_close(result, torch.zeros(1, 3))
    result.sum().backward()
    torch.testing.assert_close(values.grad, torch.zeros_like(values))
    for method in (Concat(), MaskedMean(), MajorityVote(tie_break="error")):
        assert not method.defined(mask.T).item()


@pytest.mark.parametrize("method", [MaskedMean(), MajorityVote(tie_break="error")])
def test_entirely_absent_input_can_be_omitted(method):
    values = {"available": torch.tensor([[0.8, 0.2]])}
    present = {"absent": torch.tensor([False]), "available": torch.tensor([True])}
    stacked, mask = _stack_defined(values, present)
    torch.testing.assert_close(stacked, values["available"].unsqueeze(0))
    assert mask.tolist() == [[True]]
    result = method(values, present) if isinstance(method, MaskedMean) else _vote(method, values, present)
    expected = [[0.8, 0.2]] if isinstance(method, MaskedMean) else [[14 / 15, 1 / 15]]
    torch.testing.assert_close(result, torch.tensor(expected))


@pytest.mark.parametrize("tie_break", ["mean_probability", "error"])
def test_vote_scores_and_probability_gradients(tie_break):
    first = torch.tensor([[0.8, 0.2], [0.3, 0.7]], requires_grad=True)
    second_present = torch.tensor([[0.6, 0.4]], requires_grad=True)
    # as BaseOutput.scatter builds it: the absent row is a NaN constant, not computed from parameters
    second = torch.cat([second_present, torch.full((1, 2), float("nan"))])
    present = {"a": torch.tensor([True, True]), "b": torch.tensor([True, False])}
    result = _vote(MajorityVote(tie_break=tie_break), {"a": first, "b": second}, present)
    torch.testing.assert_close(result, torch.tensor([[0.94, 0.06], [0.1, 0.9]]))
    # One class avoids the constant sum of all vote scores; argmax boundaries are excluded.
    result[:, 0].sum().backward()
    assert first.grad is not None and second_present.grad is not None
    assert torch.isfinite(first.grad).all() and torch.isfinite(second_present.grad).all()
    assert first.grad.abs().sum() > 0 and second_present.grad.abs().sum() > 0


def test_vote_ties_have_fixed_scores_and_error_policy():
    values = {"a": torch.tensor([[0.9, 0.1], [0.6, 0.4]]), "b": torch.tensor([[0.3, 0.7], [0.4, 0.6]])}
    present = dict.fromkeys(values, torch.tensor([True, True]))
    result = _vote(MajorityVote(tie_break="mean_probability"), values, present)
    torch.testing.assert_close(result, torch.tensor([[0.52, 0.48], [0.5, 0.5]]))
    assert result.argmax(dim=1).tolist() == [0, 0]
    with pytest.raises(ValueError, match="2 rows have tied votes"):
        _vote(MajorityVote(tie_break="error"), values, present)


@pytest.mark.parametrize("method", [Concat(), MaskedMean(), MajorityVote(tie_break="error")])
def test_public_class_paths_and_current_pickle_roundtrip(method):
    from kalecancer.model import fusion

    assert getattr(fusion, type(method).__name__) is type(method)
    restored = pickle.loads(pickle.dumps(method))
    assert type(restored) is type(method)
    assert isinstance(restored, FusionMethod)
    assert not restored.state_dict()
    if isinstance(method, MajorityVote):
        assert restored.tie_break == method.tie_break
