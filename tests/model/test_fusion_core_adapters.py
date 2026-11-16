"""Application compatibility at the PyKale fusion boundary."""

import copy
import io
import pickle
from collections.abc import Mapping

import pandas as pd
import pytest
import torch
from torch import Tensor, nn

from kalecancer.loaddata import Classification
from kalecancer.model import Concat, FusionContext, FusionMethod, MajorityVote, MaskedMean
from kalecancer.model.fusion import _masked_mean, _stack_defined


class LearnableMean(FusionMethod):
    """Custom extension using the existing KaleCancer helper signatures."""

    stages = frozenset({"intermediate"})

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def defined(self, present: Tensor) -> Tensor:
        return present.any(dim=1)

    def forward(self, values: Mapping[str, Tensor], present: Mapping[str, Tensor]) -> Tensor:
        return self.scale * _masked_mean(*_stack_defined(values, present))


@pytest.mark.parametrize(
    ("method", "context"),
    [
        (Concat(), FusionContext.without_data("intermediate", {"a": 2})),
        (MaskedMean(), FusionContext.without_data("intermediate", {"a": 2})),
        (MajorityVote("error"), FusionContext.without_data("late", {"a": 2}, {"a": Classification})),
    ],
    ids=["concat", "masked_mean", "majority_vote"],
)
def test_adapters_keep_cancer_context_and_skip_batch_validation(method: FusionMethod, context: FusionContext) -> None:
    method.check(context)
    assert isinstance(method, FusionMethod)
    assert not method.validate_inputs
    assert method.defined_rows(pd.DataFrame({"a": [True, False]}, index=["001", "002"])).tolist() == [True, False]


def test_concat_passes_raw_missing_cells_to_encoders() -> None:
    values = {"a": torch.tensor([[1.0, float("nan")]], dtype=torch.float32), "b": torch.tensor([[3.0]])}
    result = Concat()(values, dict.fromkeys(values, torch.tensor([True])))
    torch.testing.assert_close(result, torch.tensor([[1.0, float("nan"), 3.0]]), equal_nan=True)


def test_float16_mean_avoids_reduction_overflow() -> None:
    values = {name: torch.tensor([[60000.0]], dtype=torch.float16, requires_grad=True) for name in ("a", "b")}
    result = MaskedMean()(values, dict.fromkeys(values, torch.tensor([True])))
    torch.testing.assert_close(result, values["a"])
    result.sum().backward()
    for value in values.values():
        torch.testing.assert_close(value.grad, torch.full_like(value, 0.5))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_votes_return_float32_scores(dtype: torch.dtype) -> None:
    values = {"a": torch.tensor([[0.9, 0.1]], dtype=dtype), "b": torch.tensor([[0.3, 0.7]], dtype=dtype)}
    present = dict.fromkeys(values, torch.tensor([True]))
    scores = MajorityVote("mean_probability")(values, present)
    expected = (1 + 0.5 * torch.stack(list(values.values())).float().mean(0)) / 2.5
    torch.testing.assert_close(scores, expected)
    assert scores.dtype == torch.float32
    assert scores.argmax(1).item() == 0


def test_custom_extension_keeps_parameters_gradients_and_serialization() -> None:
    method = LearnableMean()
    values = {"a": torch.tensor([[3.0]]), "b": torch.tensor([[float("nan")]])}
    present = {"a": torch.tensor([True]), "b": torch.tensor([False])}
    method.check(FusionContext.without_data("intermediate", {"a": 1, "b": 1}))
    method(values, present).sum().backward()
    torch.testing.assert_close(method.scale.grad, torch.tensor(3.0))
    stream = io.BytesIO()
    torch.save(method, stream)
    stream.seek(0)
    restored_methods = [
        copy.deepcopy(method),
        pickle.loads(pickle.dumps(method)),
        torch.load(stream, weights_only=False),
    ]
    for restored in restored_methods:
        assert isinstance(restored, FusionMethod)
        assert list(restored.state_dict()) == ["scale"]
        torch.testing.assert_close(restored(values, present), torch.tensor([[6.0]]))
