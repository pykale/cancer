# ADR 0005: Endpoint architecture

> **Status:** implemented on the branch for issue #23. Decisions may still change in review.

- **Date:** 2026-10-30
- **Scope:** what a model outputs, who declares and checks it, where heads, losses and fitted state live, how models
  are structured around them, when checks run, and how metrics and predictions read model outputs

## TL;DR

- Models return **typed outputs**, such as `CoxOutput` or `ClassProbabilities`, whose type says what a prediction means.
- An **endpoint** wraps the head and the loss, checks the target, builds the typed output and holds state fitted after
  training, such as the Breslow baseline.
- Endpoints, fusion methods and metrics declare output types, and one rule checks them: does the produced type satisfy
  the required one? This replaces column-name parsing and `isinstance` checks on targets.
- `LateFusion` takes an endpoint for its fused output, so the fused prediction is trained. Its loss is
  `fused + branch_weight × (sum of the branch losses) + the fusion method's regulariser`.
- Each check runs as soon as its information exists: when the model is built, when `fit` starts, or in the dry run.
- Fitted state, and the censoring distribution of IPCW metrics, use the training patients only.

## Context

An endpoint such as Cox survival was spread across six modules that talked through string keys, column names and
`isinstance` checks (issue #23). This caused several problems:

- **Unchecked string contracts.** Losses read target tensors by name, and metrics worked out what a prediction was
  from its column names. A mismatch showed up only during training.
- **The fused output of `LateFusion` was never trained.** Its loss was the sum of the branch losses.
- **Metrics weren't checked against the model**, so a wrong early-stopping metric failed after the first epoch.
- **Nothing recorded the output space.** Cox log-hazards have an arbitrary offset, and vote scores aren't
  probabilities, but nothing told consumers so.
- **The head was fused to the loss**, so changing the loss meant writing a new head.
- **The fitted population was inconsistent.** The baseline used training and validation patients, while IPCW
  metrics used training patients only.

## 1. Models return typed outputs

**Options:** keep the untyped `ModelOutput` (before); store outputs as attributes on the model, which makes modules
stateful and breaks when two batches run at once; return typed outputs (chosen).

**Choice:** `forward` returns a subclass of `BaseOutput`, which holds the patient ids, which patients are defined, and
the raw values:

| Output | Means |
| --- | --- |
| `RiskScore` | risk scores with an arbitrary offset: only the ranking within one model means anything |
| `CoxOutput` | a `RiskScore` of log-hazards, and a `SurvivalFunction` once its baseline is fitted |
| `ClassProbabilities` | the softmax of trained logits |
| `VoteScores` | vote scores that rank patients but are not probabilities |

`SurvivalFunction` and `ClassScores` are capabilities that several outputs provide.

- **Outputs are named after what they mean, not how they were trained.** A ranking loss also gives a `RiskScore`, and
  `HarrellC` scores it.
- **Risk scores and survival curves are separate capabilities**, because each occurs without the other: averaged Cox
  log-hazards have no baseline, and DeepHit gives curves without a risk score. Likewise, class probabilities and vote
  scores are separate types, so `predict_proba` and calibration metrics can refuse vote scores when the model is
  built.
- **Outputs store raw values** (logits, log-hazards) and compute views (probabilities, curves) when asked.
- **Typed outputs are always model outputs.** A head returns a plain tensor, and everything before the endpoint stays
  tensors, as ADR 0002 chose.
- **A field is per patient exactly when it is a tensor**, so `select`, `cat` and the other helpers work for any new
  type. Tagging fields explicitly was rejected as more verbose. Revisit this when the first type with a shared
  tensor, such as discrete-time bin edges, is added.

## 2. Producers and consumers declare output types

**Options:** target types on heads, plus metrics parsing column names (before); a mixin forcing every component to
define `output_type`, which needs signatures that don't fit together; declared types where outputs are produced and
consumed (chosen).

**Choice:** compatibility is one rule, `issubclass(produced, required)`.

- An endpoint declares the type it produces, and the type it `accepts` as `LateFusion`'s fused endpoint.
- A late fusion method declares the type it makes from the branch types, keeping only what fusing preserves:
  `MaskedMean` over `CoxOutput`s gives a `RiskScore`, because no branch's baseline belongs to the average.
- A metric declares the weakest type it needs: `HarrellC` needs a `RiskScore`, `IntegratedBrier` a
  `SurvivalFunction`, `AUROC` `ClassScores`.
- **How to combine a type belongs to the fusion method, not the type.** Averaging logits rather than probabilities is
  `MaskedMean`'s choice, and another method could combine the same type differently. So each method names the types
  it combines, and refuses the rest. This reverses an earlier version of this ADR, in which types declared how they
  were averaged. A registry was rejected as an `isinstance` chain with global state.
- A type declares only what holds whoever combines it: `comparable_across_models`, `False` by default, so risk scores
  aren't mixed across patients with different branches.

## 3. Endpoints wrap the head and the loss

**Options:** the head owns the loss (before); the model takes a free `loss=` argument, which would let a Cox model with
a ranking loss claim valid survival curves; users write an endpoint to change the loss, which is too much for a common
need; an endpoint that wraps the head and the loss (chosen).

**Choice:** a model takes an endpoint, never a bare head:

```python
Unimodal("clinical", [MLP(...)], ClassificationEndpoint(nn.Linear(64, 3), loss_fn=nn.CrossEntropyLoss()))
```

- **The head is any `nn.Module`**, sized by the user, so its initial weights are drawn when the model is built.
  Building it lazily from the target was rejected, because `fit` would no longer start from the weights the model
  was built with.
- **Classification (and later regression) takes any torch-style `loss_fn`**, because the output means the same
  whichever reasonable loss trains it.
- **Survival endpoints are one class per family** (`CoxEndpoint`, later DeepHit), because there the loss defines what
  the output means: the Cox partial likelihood is what makes a Breslow baseline valid.
- **`decomposable_loss`** says whether the loss is a mean over patients, which gradient accumulation needs. A
  user-supplied loss takes it as an argument, defaulting to `False`: only the user knows, and a wrong `False` only
  refuses accumulation, while a wrong `True` gives silently wrong gradients. `CoxEndpoint` is always `False`, because
  its risk set is the batch.

## 4. Models

```text
BaseModel
├── SingleEndpointModel     encoders → [fusion of tensors] → endpoint
│   ├── Unimodal
│   ├── EarlyFusion
│   └── IntermediateFusion
└── LateFusion              branches → fusion of outputs → [endpoint]
```

**`LateFusion(branches, fusion, endpoint, branch_weight=1.0)`** takes an endpoint for the fused output, with no head,
such as `CoxEndpoint(None, ties="efron")`.

- The argument has no default but may be `None`, so the user always says what owns the fused output.
- Types decide which endpoints are valid. No fusion method is named: `MajorityVote` gives `VoteScores`, which no
  classification endpoint accepts, so it takes `None`.
- Each branch's endpoint holds its own state, and the fused endpoint holds the fused state, such as a baseline fitted
  on the averaged log-hazards. Copying the first branch's endpoint instead was rejected: it hides what owns the fused
  output.

**Fusion methods** implement one of two interfaces, or both:

- `TensorFusion` for early and intermediate fusion, which combines tensors;
- `OutputFusion` for late fusion, which combines typed outputs.

Models check the interface when they are built, and never test for a particular method. `MaskedMean` implements both,
with one typed `forward` overload per interface. This replaces `stages` and an untyped `forward`.

Each rule has its own hook, called when its information exists: the output type when the model is built, rules about
which branches patients have for each dataset, and widths in the dry run. This replaces `FusionContext`, which passed
one bag of partly filled information to a single `check` at all three moments. The model itself checks that every
patient it must predict for can be combined, so a method can't skip that rule.

Rejected: two public classes for the one method that works in both settings, and a factory that swaps the user's
method for a hidden stage-specific one.

## 5. Losses

**Options:** late fusion sums the branch losses (before); it trains the fused output only, so branches are no longer
useful on their own for `branch=` predictions; the fused loss plus the branch losses as auxiliary terms (chosen).

**Choice:** the late fusion loss is

```text
fused loss + branch_weight × (sum of the branch losses) + the fusion method's regulariser
```

- The fused loss exists only with a fused endpoint. With `endpoint=None`, only the branches train.
- **A late fusion method may add a regulariser**, a penalty on itself such as one on its learned weights. It is
  computed from the output, the target and the method's parameters, never from state saved during `forward`, because
  validation computes its loss on a whole epoch at once.
- **Only late fusion has a regulariser.** In early and intermediate fusion, the output comes from the head several steps
  after the fusion method, so the method's useful terms need values only `forward` sees. A trainable tensor fusion
  method is still trained there by the endpoint's loss.

## 6. Fitted endpoint state

**Options:** the `Pipeline` owns the Breslow baseline (before), so `branch=` couldn't give survival curves; the output
fits its own state, so outputs would both hold data and fit things; the endpoint fits and holds its state (chosen).

**Choice:** the endpoint takes what it needs from the target when `fit` starts (such as the class order), and fits
state on the training patients' outputs once training ends (such as the baseline). Fitted state and the IPCW
censoring distribution use the **training patients only**: validation patients chose the model, they didn't fit it.
This matches pycox and sksurv (checked 2026-11-16).

## 7. Each check runs as soon as its information exists

This extends ADR 0002 §4 from widths to targets, outputs and metrics.

**Options:** check everything at `fit`; check everything when the model is built, before targets and widths exist;
add a separate dummy `fit`; check each rule at the first point its information exists (chosen).

| When | Checks |
| --- | --- |
| Model built | output types fit together, and each fusion method has the interface its model needs |
| `fit` starts | the endpoint suits the target, missing modalities can be handled, the early-stopping metric suits the output |
| Dry run | widths, and the loss on real target tensors |

`Pipeline` is an sklearn estimator, whose `__init__` must not validate, so it checks its metric when `fit` starts.
A metric that needs survival curves can't drive early stopping, because the baseline is only fitted after training.

## 8. Predictions and metrics

**Choice:**

- `predict` returns a DataFrame. `predict_proba` and `predict_survival_function(times)` (sksurv's name) give
  probabilities and curves, and refuse outputs that can't provide them. The `survival_times` argument is removed.
- Metrics receive the typed output and the target, instead of frames plus `EvalContext`. Metrics that need times
  hold them, as `IntegratedBrier(times)` does.

**The training target.** IPCW metrics (`UnoC`, `TimeDependentAUC`, `IntegratedBrier`) estimate censoring from the
training patients, so they need the training outcomes too.

**Options:** pass it on every call (chosen); fit metrics first, which makes them stateful, so they must be refitted in
every fold; attach it to the output, which puts evaluation logic in the model.

**Choice:** `metric(output, target, train_target)`, which is sksurv's convention. Metrics stay stateless, and those that
don't need the training target ignore it.

## 9. Batching

Training batches are shuffled by default. The design leaves room for an endpoint to supply a sampler later, such as
one that guarantees every Cox batch has an event. Classification isn't rebalanced by default, because balanced
sampling distorts the predicted probabilities; class weights in the loss are the usual answer.

## 10. Names

**Decided (2026-11-16):** the current names are final.

- Outputs: `BaseOutput`, `RiskScore`, `CoxOutput`, `SurvivalFunction`, `ClassScores`, `ClassProbabilities`,
  `VoteScores`.
- Models: `BaseModel`, `SingleEndpointModel`, `LateFusion`.
- Fusion: `TensorFusion` and `OutputFusion`.
- Endpoints: named `<Family>Endpoint`, to keep them apart from targets such as `Classification`. Their loss method is
  `loss`, so a user-supplied loss is the argument `loss_fn`.

## Open questions

- **Default endpoints.** Users always give a head, and a loss for classification. Defaults, such as a linear head and
  cross-entropy, would cover the common case.
- **Class weights from the training target**, as sklearn's `class_weight="balanced"` computes them.
- **A `loss_fn=` for `CoxEndpoint`**, left out until someone needs it.
- **Regularisers for tensor fusion methods**, such as attention entropy or contrastive alignment. They need values only
  `forward` sees, which would have to be carried on the output.
- **Labels for fitted encoders.** TabICL labels its context with the event flag, so a patient censored at 3 months
  looks like a long-term survivor. The endpoint should decide what labels an encoder sees.

## Out of scope

- Separating `strata` from splitting, and splitting by groups such as centre.
- The naming of fusion methods (`Concat` or `FusionConcat`), which belongs with kale core.
- Saving a fitted pipeline. Fitted state, such as the baseline, isn't in the model's state dict yet. To be raised as
  its own issue.

The design must not rule out several targets per dataset (multi-task), or survival beyond right censoring
(competing risks, left truncation, interval censoring).

## Consequences

**Breaking changes:**

- Models take `endpoint=` instead of `head=`. `CoxHead` and `ClassificationHead` become `CoxEndpoint` and
  `ClassificationEndpoint`, whose head is any `nn.Module`.
- `LateFusion` takes a required `endpoint`, which may be `None`, and trains the fused output. No branch may be called
  `fused` or `regulariser`, the names of its loss parts.
- The Breslow baseline uses training patients only, so survival curves change for pipelines with a validation split.
- Metrics are called as `metric(output, target, train_target)`.
- `MaskedMean` over Cox branches without a fused endpoint gives a `RiskScore`, with a `risk` column and no survival
  curve.

**Removed:** `ModelOutput`, `EvalContext`, `FusionContext`, `FusionMethod.stages` and `input_space`, head
`target_type` and `columns`, `Pipeline.baseline_`, and the `survival_times` argument.

**Added:** `predict_output`, `predict_proba`, `predict_survival_function`, `BaseTarget.subset`, and
`OutputFusion.loss`.

The head contract in the old `docs/extending/endpoints.md` is superseded, and ADR 0002 §4 is extended.
