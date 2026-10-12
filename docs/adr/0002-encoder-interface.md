# ADR 0002: Encoder interface

- **Date:** 2026-10-08
- **Scope:** what an encoder is, who checks what it accepts and when, the extra abilities an encoder can have, and what encoding things are called

## TL;DR

- An encoder is any `nn.Module` whose `forward(x)` takes a batch and returns a batch. There is no `Encoder` base
  class, so stock torch, timm and Hugging Face modules work unchanged.
- The encoder checks its own input in `forward`. Nothing reads declared widths. `fit` runs a dry run on a few training
  patients before training, so a wrong input fails before the first optimizer step.
- Extra abilities are named types: `FittedEncoder` for encoders fitted on training rows, and `AttentionEncoder` for
  encoders that report attention.
- `EncoderList` checks its output, not its input. This supersedes the input check of ADR 0001 §8.
- Names are consistent: an **encoder**, an **`EncoderList`**, models take **`encoders`**, and the output is an **embedding**.
- The Pipeline no longer re-initialises weights.

## 1. An encoder is any `nn.Module`

**Context:** Seven optional attributes were found with `getattr` (`needs_ids`, `allow_nan`, declared widths,
`keep_weights`, `reset_parameters`, `attention`) in three places, so a new encoder's author had to read all three.
ADR 0001 fixed this for modalities with a base class, but `Modality` is the library's own abstraction and `nn.Module`
is torch's.

**Options:**
- hidden attributes (before)
- an `Encoder` base class, which would need a wrapper around every stock module
- a `Stage(module, ...)` wrapper
- any `nn.Module`, with each extra ability a small named type (chosen)

**Choice:** An encoder only needs `forward(x)`. Its input is the modality's batch (an `(n, ...)` tensor for a
`FixedShapeModality`, a list of `(N_i, d)` tensors for a `BagModality`) or the previous encoder's output.

## 2. Naming

**Context:** A module in the list was a "stage", a "block" or an "encoder"; the model argument was `encoding`; and
"stage" also meant the fusion point.

**Choice:** Call each module an encoder and name everything else after it. "Stage" now means only the fusion point.

| Before | After |
| --- | --- |
| stage, block | encoder |
| `StageList` | `EncoderList` |
| `encoding` (argument, attribute, YAML key, `param_groups` prefix, error messages) | `encoders` |
| `stages_for` | `encoders_for` |
| `run_modality` | `run_encoders` |
| `InContextModule` | `FittedEncoder` |
| "fusion input" | embedding |

Encoders live in `model/encode/`:

- `base_encoders.py`: `FittedEncoder` and `AttentionEncoder`, in their own file so that every encoder file can import
  them without circular imports;
- `encoders.py`: `MLP` and `ABMIL`;
- `encoder_list.py`: `EncoderList`;
- `tabicl.py`: `TabICLEncoder`, kept separate so that importing the other encoders doesn't import the optional
  `tabicl` extra.

## 3. The encoder owns its rules

**Context:** `StageList` compared declared widths. It couldn't see the batch format or a modality's width after its
transform, and it skipped silently whenever a module declared no width.

**Options:**

- a central checker of declared widths (before)
- specs on both sides, which grow into a type system and
need an adapter for every stock module
- shape propagation on the meta device, which breaks on code that depends on the data, such as TabICL
- the encoder checks its own input (chosen)

**Choice:** Only the encoder knows what it accepts, so it checks that in `forward` if it wants a clear message, as
`ABMIL` does. A stock module raises its own error. `EncoderList` adds a note naming the encoder and its input to any
error, for example "in `encoders['wsi'][0]` Linear, called on a list of 2 tensors shaped (812, 1024), …".
`EncoderList` never inspects what kind of encoder it holds, and a new encoder needs no code anywhere else.

## 4. Each check runs as soon as the information it needs exists

**Context:** A modality's width is only known once its transform has been fitted in `fit`, so a complete check can't
run when the model is built. Checking only in the first training batch is late.

**Options:**

- Check declared widths at construction (before). This is incomplete, and it reports false errors: a module that
  declares no width, such as `nn.Sequential`, a convolution or `nn.LazyLinear`, breaks the walk.
- Keep a corrected construction-time walk alongside the dry run. It finds nothing the dry run doesn't, a few seconds
  earlier, and brings back the `getattr` widths.
- Add a public `Pipeline.check(data)`. This is a second entry point for what `fit` does in its first few seconds.
- Do a dry run inside `fit` (chosen).

**Choice:** After the transforms and any `FittedEncoder` are fitted, and before the optimizer is built, `fit` runs the
model once:

- in eval mode under `no_grad`;
- on up to two training patients per modality, plus up to two who have every modality.

Each model's `dry_run` runs the encoder lists and calls `FusionMethod.check` with the real embedding widths. It then
runs fusion and the head, and the error notes name the head and, in late fusion, the branch. Width rules run only
here. `model.check(data)` at prediction checks only the modalities, the target and presence. `FusionMethod.output_dim`
and the declared-width checks are removed.

**Costs:**

- A width mismatch fails a few seconds into `fit` instead of at construction.
- A stock module's shape error is less direct than "expects width 64".
- Code paths that only run in training mode aren't exercised.

## 5. Fitted encoders are a general kind

**Context:** `InContextModule` was the only encoder fitted on training rows, and `needs_ids` existed only for it.
Other fitting inside a fold (PCA, a k-means codebook, label-aware selection) has the same leakage concerns.

**Options:**

- Keep `InContextModule` (before).
- Add an `InContextEncoder` between `FittedEncoder` and TabICL. It would have only one subclass.
- Allow fitted encoders anywhere in the list. They would be fitted on features that change during training.
- Use one general `FittedEncoder` that TabICL subclasses directly (chosen).

**Choice:** `FittedEncoder` declares `fit(x, target, ids)` and `forward(x, ids)`:

- `target` is the training dataset's `BaseTarget`, so each encoder chooses what it reads. TabICL reads
  `target.strata(ids)`, so `context_label` goes.
- Only the first encoder may be fitted, and only a `FittedEncoder` receives `ids`.
- The model loads the training rows only when `EncoderList.needs_fit` says so.

Fitted state must load under `torch.load(weights_only=True)`:

- Store it in buffers, or in `extra_state` for anything that isn't a tensor, never as pickled objects. TabICL
  reimplements its preprocessing transform from tensors for this reason.
- `FittedEncoder` resizes data-dependent buffers when it loads a state dict.

A fitted encoder owns the rows it is fitted on, including any NaN in them, and must embed each patient independently
of its batch.

`TabICLEncoder` embeds a training patient against the context without the patient's stratified fold. Leaving out only
the patient would let the class counts give its label away. A row whose id is in the context must match the stored
row. Its tests run on a tiny, randomly initialised TabICL checkpoint, so they need no download. TabICL's context is
the training patients' rows and labels, so it is saved in every checkpoint.

## 6. The encoder list's output is checked; its input is reported

**Context:** Torch passes NaN through without raising, so ADR 0001 §8 checked that the input was finite, with an
`allow_nan` opt-out. That check had three gaps:

- encoders differ in whether they accept NaN, inf or both;
- it missed non-finite values the encoders produce themselves;
- it passed a `(ΣN_i, d)` output.

The input differs from encoder to encoder. The output doesn't, because every fusion method and head needs one finite
vector per patient.

**Options:**

- an input check with `allow_nan` (before)
- separate flags for NaN and inf
- no check
- an output check (chosen)

**Choice:** `EncoderList` checks every embedding, in this order:

1. it is a tensor;
2. it has two dimensions;
3. it has one row per patient;
4. it is floating point (on an integer tensor, `isfinite` always passes);
5. every row is finite.

The error names the patients and says whether their input was already non-finite or the encoders produced the
values. `allow_nan` and `check_input` go.

**The warning:** an encoder can turn NaN into a finite value, on purpose or by accident, and the output check can't
see that. So the Pipeline collects the patients whose input has NaN or inf, from every batch of `fit`, `predict` and
`run_encoders`. At the end of the pass it raises one `NonFiniteInputWarning` per modality, and `fit` records the
counts in `fit_report_["non_finite_inputs"]`. Other places to raise the warning were rejected:

- in `EncoderList`: Python's deduplication makes it either silent or repeated every batch;
- in the modality: worker processes hide it;
- when the transforms are fitted: it sees the data before imputation;
- in the first epoch only: the warning would depend on epochs.

## 7. Encoders that report attention subclass `AttentionEncoder`

**Context:** Attention export found the attending encoder with `getattr`, and nothing said what `attention()`
returned. Future encoders attend in different shapes, such as per instance and class (CLAM) or per instance and head.

**Options:**

- `getattr` (before).
- A `Protocol` in `interpret`. Structural matching gains nothing here.
- An `AttentionPool` base class. It fits only encoders whose attention is their final weighted sum.
- Forward hooks with typed axis maps. Far more code than needed.
- A small abstract base class with a fixed return contract (chosen).

**Choice:** `AttentionEncoder` declares `attention(bags) -> list[Tensor]`: one tensor per bag whose last dimension is
aligned with the bag's instances. Export checks `isinstance` and that last dimension, and for now it reads only
`(N_i,)` weights. The method is called `attention` because in torch "weights" means parameters. Attention inside
fusion methods (MCAT) is left for its own decision.

## 8. One way to run an encoder list

**Context:** Attention export ran the encoders by hand, repeating how `ids` are passed. The output check was written
twice.

**Choice:** `EncoderList.steps(x, ids)` is a generator. It calls each encoder, adds the error note, and yields
`(index, encoder, output)`. `forward` is built on it. Attention export stops iterating before the attending encoder,
and checks that no earlier encoder changed the bag sizes. A `run(x, ids, stop=k)` method was rejected because it
hides each encoder's output from that check.

## 9. The Pipeline no longer re-initialises weights

**Context:** With `random_state` set, `fit` called `reset_parameters()` on every module unless the module set
`keep_weights`. `nn.MultiheadAttention` has no public `reset_parameters()`, so any transformer encoder failed.

**Options:** re-initialise (before); rebuild the model from its constructor arguments, which fails for modules whose
configs can't be dumped; follow torch's convention (chosen).

**Choice:** A module is initialised when it is built, and `fit` trains a copy of it. `random_state` seeds only what
`fit` does. To vary the initial weights, seed before building the model. `load_pipeline` builds the model under the
config's `random_state`. An encoder's own randomness, such as TabICL's fold assignment, is seeded by that encoder's
own argument. `_reinitialise`, `keep_weights` and the `reset_parameters` contract go.

## Consequences

- **Removed:**
  - classes and helpers: `StageList`, `_declared`, `output_width`, `_check_head_width`, `_vectors`, `_run_stages`,
    `_reinitialise`, `InContextModule`;
  - flags and checks: `needs_ids`, `allow_nan`, `check_input`, `keep_weights`;
  - `TabICLEncoder`'s `context_label` and `out_dim`, and `FusionMethod.output_dim`.
- **Added:** `EncoderList.steps`, the output check, `FittedEncoder`, `AttentionEncoder`, the dry run, and
  `NonFiniteInputWarning`.
- **Superseded:** the input check of ADR 0001 §8.
