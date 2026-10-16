# Adding an encoder

An encoder is any `nn.Module` whose `forward(x)` takes a batch and returns a batch. There is no `Encoder` base class,
so torch, timm and Hugging Face modules work unchanged, and a new encoder needs no code anywhere else in the
library.

## 1. Write the module

Its input is the modality's batch, or the previous encoder's output:

| Modality kind | `x` for the first encoder |
| --- | --- |
| `FixedShapeModality` | an `(n, *shape)` tensor |
| `BagModality` | a list of `(N_i, d)` tensors, one per patient |

```python
class Scale(nn.Module):
    """Learns one scale per feature."""

    def __init__(self, in_dim: int, init: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.init = init
        self.weight = nn.Parameter(torch.full((in_dim,), init))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"Scale expects (n, {self.in_dim}), got {tuple(x.shape)}")
        return x * self.weight
```

The rules:

- **Check your own input in `forward`** if you want a clear error. Nothing reads declared widths. Before training,
  `fit` runs the model on up to two training patients per modality (the dry run), so a wrong input fails early, with a
  note naming the encoder:

  ```text
  ValueError: Scale expects (n, 6), got (2, 5)
  in encoders['rna'][0] Scale, called on a tensor of shape (2, 5)
  ```

- **The last encoder in a list must return the embedding**: a finite, floating-point `(n, d)` tensor, one row per
  patient. `EncoderList` checks this on every pass. A bag encoder therefore has to pool somewhere in the list.
- **Follow the [Conventions](../../CONTRIBUTING.md#conventions)**, which cover weights and seeding.
- **Sample a bag's instances at random only when `self.training` is true.** A fixed selection belongs in the
  modality.
- **Handle non-finite input yourself.** An encoder receives one modality, or the fused vector in `EarlyFusion`, for
  only the patients who have it. `EncoderList` doesn't stop NaN or inf in that input, so an encoder that can't handle
  them should raise.

## 2. Add an ability (optional)

Two base classes in `model/encode/base_encoders.py` add abilities. Subclass one only when you need it.

**`FittedEncoder`**: fitted on the training patients before training starts, e.g. a PCA, a codebook or TabICL's
context.

```python
class Centre(FittedEncoder):
    mean: Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.empty(0))

    def fit(self, x: Tensor, target: BaseTarget, ids: list[str]) -> None:
        self.mean = x.mean(dim=0)

    def forward(self, x: Tensor, ids: list[str]) -> Tensor:
        return x - self.mean
```

- It may only be the **first** encoder in a list. The input of a later encoder changes as the encoders before it
  train.
- `fit` receives the training patients who have the modality, their ids and the dataset's `BaseTarget`. Read what you
  need from the target, e.g. `target.strata(ids)`. Only a `FittedEncoder` receives `ids` in `forward`.
- Keep fitted state in buffers, or in `get_extra_state` and `set_extra_state` for anything that isn't a tensor, so
  checkpoints load under `weights_only=True`. Buffers may have a size that depends on the data. The base class resizes
  them when a state dict is loaded.
- Embed each patient independently of the rest of the batch. If a patient's own label is part of what you fitted, as
  in an in-context model, keep it from leaking into that patient's embedding.

**`AttentionEncoder`**: a bag encoder that reports what it attends to, which `kalecancer.interpret.attention` reads.
Implement `attention(bags)`, returning one tensor per bag whose last dimension is aligned with the bag's instances.
Export currently reads `(N_i,)` weights. Encoders before it must not change the number of instances. See `ABMIL` in
`model/encode/encoders.py`.

## 3. Use it

```python
model = IntermediateFusion(
    encoders={"rna": [Scale(5), nn.Linear(5, 8)], "wsi": [ABMIL(1024, 256, 128, 0.25), nn.Linear(256, 8)]},
    fusion=Concat(),
    head=CoxHead(in_dim=16, ties="efron"),
)
```

In a YAML config it is `{class_path: kalecancer.model.Scale, init_args: {in_dim: 5}}`.

## Where it goes

- An encoder with no new dependencies goes in `kalecancer/model/encode/encoders.py`, exported from
  `kalecancer/model/__init__.py`.
- One that needs an optional extra gets its own file under `model/encode/`. Add the extra to `pyproject.toml`, and
  export the encoder lazily through the `__getattr__` in `kalecancer/model/__init__.py`, as `TabICLEncoder` is, so
  the rest of the package imports without the extra.

## Checklist

- [ ] The encoder is in the right file and exported from `kalecancer/model/__init__.py`.
- [ ] Tests in `tests/test_model_components.py` cover its output shape and its input error and, for a
  `FittedEncoder`, that a fresh instance loads a fitted one's `state_dict`. A short `Pipeline.fit` on the `cohort`
  fixture from `tests/conftest.py` covers the dry run.
- [ ] The encoder list in `docs/architecture.md` includes it.

See [ADR 0002](../adr/0002-encoder-interface.md) for why encoders are designed this way.
