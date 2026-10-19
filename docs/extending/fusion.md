# Adding a fusion method

> **Open to changes.** The current architecture for fusion methods does checks at construction. This complicates the process and could be fragile. Will likely change part of this architecture in the future.

A fusion method combines named inputs, any of which a patient may lack. In `EarlyFusion` and `IntermediateFusion` the
inputs are modalities. In `LateFusion` they are branches. A new method is one `FusionMethod` subclass in
`kalecancer/model/fusion.py`. Models never test for a particular method, so no model changes.

## 1. Implement it

Declare the `stages` the method supports, and implement `defined` and `forward`:

```python
class Max(FusionMethod):
    """Element-wise maximum of equal-width inputs, over the ones each patient has."""

    stages = frozenset({"early", "intermediate"})

    def defined(self, present: Tensor) -> Tensor:
        return present.any(dim=1)

    def check(self, context: FusionContext) -> None:
        super().check(context)
        _check_one_width(self, context.widths)

    def forward(self, values: Mapping[str, Tensor], present: Mapping[str, Tensor]) -> Tensor:
        stacked, mask = _stack_defined(values, present)
        fill = torch.finfo(stacked.dtype).min
        return torch.where(mask.unsqueeze(-1), stacked, torch.full_like(stacked, fill)).amax(dim=0)
```

- **`stages`**: any of `"early"`, `"intermediate"` and `"late"`. A model built with a method for another stage fails
  at construction.
- **`defined(present)`** takes an `(n_patients, n_inputs)` bool presence matrix and returns `(n_patients,)` bool: the
  patients this method can combine. The model uses the same answer in `check` and in the forward pass, so the two
  cannot disagree.
- **`forward(values, present)`** only ever receives the rows that `defined` accepts. Absent inputs hold NaN in those
  rows, so select with `torch.where`, never multiply by the mask. An input that no row has may be missing from
  `values`.

A `FusionMethod` is an `nn.Module`, so it may have learnable parameters, such as a gate or per-input weights. They
train with the rest of the model. Call `super().__init__()` in `__init__`, as `MajorityVote` does.

## 2. Add rules (optional)

Override `check(context)` and call `super().check(context)` first. The base class checks the stage, and checks that
every required patient is defined. A method's rules read only the `FusionContext` the model builds:

| Field | Holds |
| --- | --- |
| `stage` | which kind of model is fusing |
| `present` | one bool column per input, one row per patient |
| `required` | the patients who must get a combined value |
| `widths` | each input's width, or `None` when it isn't known yet |
| `head_kinds` | in late fusion, the `target_type` of each branch head |

The widths are only known in the dry run that `fit` runs before training, so width rules run there. A method never
sees the dataset.

## 3. Late fusion

In late fusion, `input_space` decides what the method receives from each branch:

- `"output"` (the default): the raw head outputs (logits or log-hazards). The first branch's head turns the fused
  result into a prediction, as with `MaskedMean`.
- `"prediction"`: the heads' predictions. The fused result is the prediction, so the method also implements
  `columns(target)` to name its columns, as `MajorityVote` does.

Check `head_kinds` if the method only makes sense for one endpoint. For example, `MaskedMean` refuses Cox branches over
patients with different branch subsets, because each branch's log-hazard has an arbitrary offset.

## 4. Use it

```python
model = IntermediateFusion(
    encoders={"clinical": [nn.Linear(12, 64)], "wsi": [ABMIL(1024, 256, 128, 0.25), nn.Linear(256, 64)]},
    fusion=Max(),
    head=CoxHead(in_dim=64, ties="efron"),
)
```

In a YAML config it is `fusion: {class_path: kalecancer.model.Max}`. Changing the method can change the width the
head takes: `Concat` sums the input widths, while `Max` and `MaskedMean` keep the shared one.

## Checklist

- [ ] The method is exported from `kalecancer/model/__init__.py`.
- [ ] Tests in `tests/test_model_fusion.py` cover `defined`, the result on rows with missing inputs (finite values
  and gradients), and each rule in `check`.
- [ ] The fusion method table in `docs/architecture.md` lists it.
- [ ] The [Conventions](../../CONTRIBUTING.md#conventions) are followed.
