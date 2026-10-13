# Adding an endpoint

> **Work in progress.** Targets and heads haven't been refactored yet, so this interface will change. Heads have no
> base class: models read `out_dim`, `target_type` and `decomposable_loss` with `getattr`, so nothing enforces the
> table below. The `Pipeline` also fills `EvalContext.classes` only for `Classification` targets. Expect to adapt an
> endpoint written against this guide.

An endpoint is what the model predicts, such as survival or a class label. It takes two classes: a target in
`kalecancer/loaddata/targets.py` that holds the outcomes, and a head in `kalecancer/model/heads.py` that predicts them
and owns the loss. The `Pipeline` and models stay as they are.

## 1. The target

Subclass `BaseTarget`, check the outcomes in `__init__`, store them in `frame` (indexed by patient id), and implement
three methods:

```python
class Continuous(BaseTarget):
    """A real-valued outcome per patient.

    Args:
        values: The outcome per patient, finite.
    """

    def __init__(self, values: pd.Series):
        self.values = values
        ids = _as_ids(values.index, "Continuous")
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"Continuous: {int((~np.isfinite(numeric)).sum())} values are missing or not finite")
        self.frame = pd.DataFrame({"value": numeric.astype(np.float32)}, index=ids)

    def tensors(self, ids: Sequence[str]) -> dict[str, Tensor]:
        return {"value": torch.tensor(self.frame.loc[list(ids), "value"].to_numpy())}

    def strata(self, ids: Sequence[str]) -> np.ndarray:
        return pd.qcut(self.frame.loc[list(ids), "value"], q=4, labels=False, duplicates="drop").to_numpy()

    def counts(self, ids: Sequence[str]) -> dict[str, int]:
        return {"patients": len(ids)}
```

- **`frame`** holds the checked outcomes. Metrics score predictions against it, and `fit` keeps the training rows in
  `train_target_`.
- **`tensors(ids)`** returns the outcomes of `ids`, in that order, as the tensors the head's `loss` reads. Each batch
  carries them as `batch["target"]`.
- **`strata(ids)`** returns one discrete label per patient. Stratified splits and the validation carve use it, and a
  `FittedEncoder` may read it as a label.
- **`counts(ids)`** returns the summary counts that `data.summary()`, fit reports and cross-validation folds show.
- **Reject missing outcomes** rather than guessing. `TimeToEvent` raises on a missing `event`, because an unknown
  outcome isn't a censored one.

## 2. The head

A head is an `nn.Module` that owns everything about the endpoint: its output, the prediction, the loss, the target
check and the names of the prediction columns.

```python
class RegressionHead(nn.Module):
    """Linear output trained with mean squared error."""

    target_type: Final[type[Continuous]] = Continuous
    decomposable_loss = True

    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 1
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, z: Tensor) -> Tensor:
        return self.linear(z)

    def predict(self, output: Tensor) -> Tensor:
        return output

    def columns(self, target: Continuous) -> list[str]:
        return ["value"]

    def check_target(self, target: BaseTarget) -> None:
        if not isinstance(target, self.target_type):
            raise TypeError(f"RegressionHead needs a continuous target, got {type(target)}")

    def loss(self, output: Tensor, target: Mapping[str, Tensor]) -> Tensor | None:
        with torch.autocast(device_type=output.device.type, enabled=False):
            return F.mse_loss(output.float().reshape(-1), target["value"])
```

| Member | Contract |
| --- | --- |
| `target_type` | the target class it predicts. Late fusion methods read it from `FusionContext.head_kinds`. |
| `out_dim` | the width of `forward`'s output. Patients with no prediction get a row of NaN of this width. |
| `forward(z)` | the raw output (logits, log-hazards) from the fused `(n, in_dim)` embedding. |
| `predict(output)` | the prediction, with one column per entry of `columns`. |
| `columns(target)` | the names of the prediction columns, read from the target where needed (e.g. class order). |
| `check_target(target)` | raises if the head can't train on this target. Accept `target_type` or a subclass. |
| `loss(output, target)` | the loss of a batch, or `None` when the batch carries no signal. For Cox, that's a batch with no event that has anyone else at risk. The `Pipeline` skips such batches and reports them. |
| `decomposable_loss` | `True` only if the loss is a mean over patients, so gradient accumulation reproduces a larger batch. Cox isn't: its risk set is the batch. |

Compute the loss in float32, outside autocast, as the stock heads do, so that mixed precision doesn't overflow it.

## 3. Metrics

An endpoint usually needs a metric too. Subclass `Metric` in `kalecancer/evaluate/metrics.py`. Set
`higher_is_better`, and implement `__call__(prediction, target, context)`: it scores the prediction frame against the
target's `frame`, matched by patient id. Use scikit-learn or TorchSurv for the computation where they have it.

## Checklist

- [ ] The target is exported from `kalecancer/loaddata/__init__.py`, the head from `kalecancer/model/__init__.py`, and
  any metric from `kalecancer/evaluate/__init__.py`.
- [ ] Tests in `tests/test_loaddata.py` cover the target's checks, `tensors`, `strata` and `counts`. Tests in
  `tests/test_model_components.py` cover the head's loss against a reference value, and `check_target`.
- [ ] A short `Pipeline.fit` and `evaluate` on synthetic data runs end to end.
- [ ] The target and head tables in `docs/architecture.md` list them.
- [ ] The [Conventions](../../CONTRIBUTING.md#conventions) are followed.
