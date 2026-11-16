# Adding a fusion method

A fusion method combines named inputs, any of which a patient may lack. It goes in `kalecancer/model/fusion.py` and
implements one or both interfaces:

| Interface | Used in | Combines |
| --- | --- | --- |
| `TensorFusion` | `EarlyFusion`, `IntermediateFusion` | `(n, d)` tensors: raw inputs or embeddings |
| `OutputFusion` | `LateFusion` | the branches' typed outputs |

Models only check which interface a method has, so adding one needs no change to the models. The design is in
[ADR 0005](../adr/0005-endpoint-architecture.md).

## 1. A tensor fusion method

```python
class Max(TensorFusion):
    """Element-wise maximum of equal-width inputs, over the ones each patient has."""

    def defined(self, present: Tensor) -> Tensor:
        return present.any(dim=1)

    def check_widths(self, widths: Mapping[str, int]) -> None:
        _check_one_width(self, widths)

    def forward(self, values: Mapping[str, Tensor], present: Mapping[str, Tensor]) -> Tensor:
        stacked, mask = _stack_defined(values, present)
        fill = torch.finfo(stacked.dtype).min
        return torch.where(mask.unsqueeze(-1), stacked, torch.full_like(stacked, fill)).amax(dim=0)
```

| Method | Contract |
| --- | --- |
| `defined(present)` | which patients the method can combine, from a `(patients, inputs)` presence mask |
| `forward(values, present)` | combines the rows `defined` accepts |
| `check_widths(widths)` | optional: raises if the input widths can't be combined; runs in the dry run |

Absent inputs hold NaN, so select with `torch.where` rather than multiplying by the mask: `NaN × 0` is NaN. Learnable
parameters, such as a gate, are trained by the endpoint's loss.

## 2. A late fusion method

This example learns a weight per branch, and penalises weights that collapse onto one branch:

```python
class WeightedMean(OutputFusion):
    """A learned weighted mean of the branches' logits, over the branches each patient has."""

    def __init__(self, branches: list[str], entropy: float):
        super().__init__()
        self.branches = list(branches)
        self.entropy = entropy
        self.weight_logits = nn.Parameter(torch.zeros(len(self.branches)))

    def defined(self, present: Tensor) -> Tensor:
        return present.any(dim=1)

    def output_type(self, input_types: Mapping[str, type[BaseOutput]]) -> type[BaseOutput]:
        if not (input_types and all(issubclass(kind, ClassProbabilities) for kind in input_types.values())):
            raise TypeError(f"WeightedMean needs branches that output class probabilities, got {dict(input_types)}")
        return ClassProbabilities

    def forward(self, values: Mapping[str, BaseOutput], present: Mapping[str, Tensor]) -> ClassProbabilities:
        branches = _as(values, ClassProbabilities)
        first = next(iter(branches.values()))
        weights = torch.softmax(self.weight_logits, dim=0)
        total, norm = 0.0, 0.0
        for name, out in branches.items():
            mask, w = present[name].unsqueeze(-1), weights[self.branches.index(name)]
            total = total + torch.where(mask, w * out.logits, 0.0)
            norm = norm + w * mask
        logits = total / norm
        return ClassProbabilities(ids=first.ids, defined=_everyone(first, logits), logits=logits, classes=first.classes)

    def loss(self, output: BaseOutput, target: Mapping[str, Tensor]) -> Tensor | None:
        weights = torch.softmax(self.weight_logits, dim=0)
        return self.entropy * (math.log(len(self.branches)) + (weights * weights.log()).sum())

    @property
    def decomposable_loss(self) -> bool:
        return True  # the penalty reads only the weights, not the batch
```

| Method | Contract |
| --- | --- |
| `defined(present)` | as above, over branches |
| `output_type(input_types)` | the type it makes from these branch types, or `TypeError`; runs when the model is built |
| `forward(values, present)` | combines the branch outputs into an output of `output_type` |
| `check_branches(present, input_types)` | optional: rules about which branches patients have |
| `loss(output, target)` | optional: a regulariser, which `LateFusion` adds to its loss unweighted |
| `decomposable_loss` | whether the regulariser is independent of batching; `False` once `loss` is overridden |

- **Name the types you combine**, and declare only what fusing keeps: `MaskedMean` over Cox outputs returns a plain
  `RiskScore`, because no branch's baseline belongs to the average. Types you don't name are refused.
- **The regulariser** is computed from the output, the target and the method's own parameters, never from state
  saved during `forward`. The total loss is `fused + branch_weight × branch losses + regulariser`.

## 3. A method for both

A method that works in every stage subclasses both interfaces, as `MaskedMean(TensorFusion, OutputFusion)` does. Its
`forward` has one `@overload` per interface and checks whether it was given tensors or outputs. Its regulariser, if
any, applies in late fusion only.

## 4. Use it

```python
model = IntermediateFusion(
    encoders={"clinical": [nn.Linear(12, 64)], "wsi": [ABMIL(1024, 256, 128, 0.25), nn.Linear(256, 64)]},
    fusion=Max(),
    endpoint=CoxEndpoint(nn.Linear(64, 1, bias=False), ties="efron"),
)

model = LateFusion(
    branches={"clinical": clinical_model, "wsi": wsi_model},
    fusion=WeightedMean(["clinical", "wsi"], entropy=0.1),
    endpoint=ClassificationEndpoint(None, nn.CrossEntropyLoss()),
)
```

The method can change the width the head takes: `Concat` sums the input widths, while `Max` keeps the shared one.

## Checklist

- [ ] The method is exported from `kalecancer/model/__init__.py`.
- [ ] Tests in `tests/test_model_fusion.py` cover rows with missing inputs (finite values and gradients),
  `output_type`, each rule, and any regulariser.
- [ ] The fusion table in `docs/architecture.md` lists it.
- [ ] The [Conventions](../../CONTRIBUTING.md#conventions) are followed.
