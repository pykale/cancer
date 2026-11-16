# Adding an endpoint

An endpoint is what a model predicts, such as survival or a class label. Adding one takes up to four pieces:

| Piece | File | Job |
| --- | --- | --- |
| Target | `kalecancer/loaddata/targets.py` | holds the outcomes |
| Output type | `kalecancer/model/outputs.py` | says what a prediction means; only needed if no existing type fits |
| Endpoint | `kalecancer/model/endpoints.py` | wraps the head and the loss, and builds the output |
| Metric | `kalecancer/evaluate/metrics.py` | scores the output; usually needed |

The `Pipeline`, models and fusion methods don't change. The example below adds regression. The design is in
[ADR 0005](../adr/0005-endpoint-architecture.md).

## 1. The target

Subclass `BaseTarget`, check the outcomes in `__init__`, and store them in `frame`, indexed by patient id:

```python
class Continuous(BaseTarget):
    """A real-valued outcome per patient."""

    def __init__(self, values: pd.Series):
        self.values = values
        ids = _as_ids(values.index, "Continuous")
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"Continuous: {int((~np.isfinite(numeric)).sum())} values are missing or not finite")
        self.frame = pd.DataFrame({"value": numeric.astype(np.float32)}, index=ids)

    def subset(self, ids: Iterable[str]) -> Continuous:
        return Continuous(self[list(ids)]["value"])

    def tensors(self, ids: Iterable[str]) -> dict[str, Tensor]:
        return {"value": torch.tensor(self.frame.loc[list(ids), "value"].to_numpy())}

    def strata(self, ids: Sequence[str]) -> np.ndarray:
        return pd.qcut(self.frame.loc[list(ids), "value"], q=4, labels=False, duplicates="drop").to_numpy()

    def counts(self, ids: Sequence[str]) -> dict[str, int]:
        return {"patients": len(ids)}
```

| Method | Returns |
| --- | --- |
| `subset(ids)` | the same target for some patients, such as the training patients |
| `tensors(ids)` | the outcomes as the tensors the endpoint's loss reads |
| `strata(ids)` | one discrete label per patient, for stratified splits |
| `counts(ids)` | summary counts for reports, such as the number of events |

Reject missing outcomes rather than guessing: an unknown event is not a censored one.

## 2. The output type

Reuse an existing type if it means what your endpoint predicts. For example, a ranking survival loss gives a
`RiskScore`, which `HarrellC` already scores. Otherwise subclass `BaseOutput`:

```python
@dataclass(kw_only=True)
class PointPrediction(BaseOutput):
    """A predicted value per patient, on the outcome's own scale."""

    value: Tensor  # (n,)

    comparable_across_models = True  # on the outcome's scale whichever model gives it

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"value": _detach_cpu(self.value).numpy()}, index=pd.Index(self.ids, name="id"))
```

- **Per-patient fields are tensors; shared fields aren't.** The base class indexes tensors by patient and copies
  anything else, such as class names, as it is.
- **Store raw values and compute views on demand**, as `ClassProbabilities` stores logits.
- **Name the type after what it means**, not how it was trained.
- **`comparable_across_models`** says whether outputs of different models share a scale. It is `False` by default,
  as for risk scores.
- **`frame()`** is what `Pipeline.predict` returns.

## 3. The endpoint

Subclass `BaseEndpoint`:

```python
class RegressionEndpoint(BaseEndpoint):
    """A value per patient, trained with any torch-style regression loss, e.g. ``nn.MSELoss()``."""

    output_type = PointPrediction
    accepts = PointPrediction

    def __init__(self, head: nn.Module | None, loss_fn: nn.Module, decomposable_loss: bool = False):
        super().__init__(head)
        self.loss_fn = loss_fn
        self._decomposable_loss = decomposable_loss

    @property
    def decomposable_loss(self) -> bool:
        return self._decomposable_loss

    @property
    def out_dim(self) -> int:
        return 1

    def check_target(self, target: BaseTarget) -> None:
        if not isinstance(target, Continuous):
            raise TypeError(f"RegressionEndpoint needs a continuous target, got {type(target).__name__}")

    def to_output(self, raw: Tensor, ids: list[str], defined: Tensor) -> PointPrediction:
        return PointPrediction(ids=ids, defined=defined, value=raw.reshape(-1))

    def adopt(self, output: BaseOutput) -> PointPrediction:
        assert isinstance(output, PointPrediction)
        return PointPrediction(ids=output.ids, defined=output.defined, value=output.value)

    def loss(self, output: BaseOutput, target: Mapping[str, Tensor]) -> Tensor | None:
        assert isinstance(output, PointPrediction)
        with torch.autocast(device_type=output.value.device.type, enabled=False):
            return self.loss_fn(output.value.float(), target["value"])
```

| Member | Contract |
| --- | --- |
| `output_type` | the output type it produces |
| `accepts` | the type it takes as the fused endpoint of a `LateFusion` model |
| `out_dim` | the width of the head's output, which the dry run checks |
| `check_target(target)` | raises if the endpoint can't train on this target |
| `to_output(raw, ids, defined)` | builds the typed output of a batch |
| `adopt(output)` | in `LateFusion`, turns the fusion method's output into this endpoint's output |
| `loss(output, target)` | the loss of a batch, in float32; `None` if the batch has no signal |
| `decomposable_loss` | whether the loss is a mean over patients, which gradient accumulation needs; `False` by default |
| `bind(target)` | optional: takes what it needs from the training target, such as the class order |
| `fit_state(output, target)` | optional: fits state after training, such as the Breslow baseline |

- **Classification and regression take a `loss_fn=`**, so users can pick a loss without writing an endpoint.
  **Survival endpoints are one class per family**, because there the loss defines what the output means: the Cox
  partial likelihood is what makes a Breslow baseline valid.
- **Name endpoints `<Family>Endpoint`**, to keep them apart from targets.
- **Store each `__init__` argument under its own name** (or a property of that name), so configs can be saved.
- **For late fusion**, a fusion method must know how to combine the new type: `MaskedMean`, for example, would need a
  rule for averaging `PointPrediction`s (see [fusion.md](fusion.md)).

## 4. The metric

Subclass `Metric`, declare `higher_is_better` and the output type it `requires`, and implement `score`:

```python
class MeanAbsoluteError(Metric):
    higher_is_better = False
    requires = PointPrediction

    def score(self, output: BaseOutput, target: BaseTarget, train_target: BaseTarget) -> float:
        assert isinstance(output, PointPrediction)
        observed = _outcomes(output, target, Continuous, repr(self))["value"].to_numpy()
        return float(np.abs(output.value.numpy() - observed).mean())
```

`_outcomes` returns the target rows in the output's order, and raises if a patient has no prediction. `train_target`
holds the training patients' outcomes, which censoring-weighted survival metrics need and other metrics ignore.

## Checklist

- [ ] The target, output type, endpoint and metric are exported from their package's `__init__.py`.
- [ ] Tests cover the target's checks and methods (`tests/test_loaddata.py`), and the endpoint's loss against a
  reference value (`tests/test_model_components.py`).
- [ ] A short `Pipeline.fit` and `evaluate` on synthetic data runs end to end.
- [ ] The tables in `docs/architecture.md` list them.
- [ ] The [Conventions](../../CONTRIBUTING.md#conventions) are followed.
