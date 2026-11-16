# ADR 0005: Endpoint architecture

> **Work in progress.** This records what was agreed while working on issue #23, before implementation. Any decision
> here may change as the work goes on. Names in code are placeholders. Sketches of the interfaces are in
> `kalecancer/model/outputs.py` and `kalecancer/model/endpoints.py`; nothing imports them yet.

- **Date:** 2026-10-30
- **Scope:** what a model outputs, who declares and checks it, where heads, losses and fitted endpoint state live,
  how models are structured around them, when checks run, and how metrics and predictions read model outputs

## TL;DR

- `ModelOutput` is replaced by typed outputs: a `BaseOutput` and subclasses for each output space, such as relative
  risk, survival functions, class probabilities and class scores. Typed outputs are always model outputs.
- Everything before the endpoint is tensors, so any `nn.Module` works as an encoder or a head. Everything after it
  is a typed output.
- An **endpoint** wraps the head and the loss. It declares the output type, checks and binds the target, builds the
  typed output, computes the loss, and holds the state fitted after training, such as the Breslow baseline.
  Classification takes any torch-style `loss_fn=`; each family of survival models is its own endpoint.
- Endpoints, late fusion methods and metrics declare output types. One rule (does the producer's type satisfy the
  consumer's?) replaces `target_type`, `head_kinds`, `input_space`, column name parsing and the special cases built
  on them.
- `Unimodal`, `EarlyFusion` and `IntermediateFusion` end in one endpoint. `LateFusion` is a composite of those
  models, and takes a required `endpoint` argument for the fused output, which may be `None`.
- Each check runs at the earliest point its information exists: when the model is built, when `fit` starts, or in
  the dry run. The dry run also runs the loss.
- Late fusion trains the fused output, with the branch losses as auxiliary terms, and fusion methods may add loss
  terms of their own.
- Fitted endpoint state is fitted on the training patients only, as is the censoring distribution for IPCW metrics.
- `EvalContext` and the `survival_times` argument are removed. Predictions come from `predict`, `predict_proba` and
  `predict_survival_function(times)`.

## Context

An endpoint, such as Cox survival, was spread across six modules that talked through string keys, column names and
`isinstance` checks (issue #23):

- `loaddata/targets.py` held the outcomes, tensor keys, strata and counts.
- `model/heads.py` held the loss, the output, the column names and the target check.
- `Pipeline._fit_baseline` fitted the Breslow baseline, and decided a model was Cox by testing
  `columns == ["log_hazard"]`.
- `Pipeline.fit`, `Pipeline.evaluate` and `cross_validate` each built an `EvalContext` with
  `isinstance(..., Classification)`.
- Metrics worked out what a prediction was from its column names.
- `MaskedMean.check` keyed a Cox-specific rule on `TimeToEvent`.

Several problems followed from this:

- **Nothing confirmed the string contracts.** `head.loss` read `target["event"]` or `target["label"]` by name, and
  the dry run never called the loss, so a mismatch showed up at the first training step.
- **The fused output of `LateFusion` was never trained.** Its loss was the unweighted sum of the branch losses. The
  fused prediction users get had no loss, and early stopping on `"loss"` watched the branch sum.
- **`ModelOutput.output` meant two things.** It was the raw head output, except under `input_space="prediction"`,
  where it held the vote scores.
- **Metrics were never checked against the model before training.** A mismatched early-stopping metric failed after
  the first epoch, and a mismatched `evaluate` metric failed after training. `EarlyStopping` hard-coded a ban on
  `IntegratedBrier`.
- **Output space mattered beyond fusion.** Cox log-hazards have an arbitrary offset, which matters for out-of-fold
  pooling in `cross_validate` and for `bootstrap_ci` as well as for `MaskedMean`. Nothing told class probabilities
  from `MajorityVote`'s vote scores, so a calibration metric would have accepted vote scores.
- **The head was fused to the loss.** Every head was a linear layer with a loss method, so changing the loss meant
  writing a new head, and a head could not be a stock module.
- **The fitted population was inconsistent.** The baseline was fitted on training and validation patients, while
  `train_target_`, which IPCW metrics read, held training patients only.
- **Column names were lossy.** Classes can be any `Hashable`, but columns format them as strings, so classes `1` and
  `"1"`, or a label containing `]`, broke the `endswith(f"[{c}]")` check.

## 1. Models return typed outputs

**Options:**

- Keep `ModelOutput` (before). It is untyped, and its fields mean different things in different models.
- Store outputs as attributes on the model. This makes every module stateful, keeps the autograd graph and device
  memory alive between calls, breaks when two batches run at once (validation inside training, distributed training),
  and hides data flow.
- Return typed outputs from `forward` (chosen).

**Choice:** A model's `forward` returns an instance of a subclass of `BaseOutput`. `BaseOutput` holds the patient
`ids`, which of them are `defined`, the `branches` of a `LateFusion` model, and `frame()`, the DataFrame that
`predict` returns. It implements `select`, `scatter`, `cat` and `detach_cpu` once for every subclass. Subclasses name
the output space:

| Output | Field | Meaning |
| --- | --- | --- |
| `RiskScore` | `risk` | a risk score with an arbitrary offset: only the ranking within one model is meaningful |
| `CoxOutput` | `risk`, `baseline` | a `RiskScore` whose risk is a log-hazard, and a `SurvivalFunction` once its baseline is fitted |
| `SurvivalFunction` | | a capability, not an output: anything that gives absolute survival probabilities S(t) |
| `ClassScores` | `classes` | abstract: `scores`, one per class, that rank patients for each class |
| `ClassProbabilities` | `logits` | a `ClassScores` whose scores are the softmax of its logits, meant as calibrated probabilities |
| `VoteScores` | `votes` | a `ClassScores` whose scores are `MajorityVote`'s vote scores, which are not probabilities |

Outputs are named after what they mean, not how they were trained: `RiskScore`, not `LogHazards`, because a ranking
loss or a survival SVM also gives risk scores that are not log-hazards, and `HarrellC` scores them all. A `CoxOutput`'s
risk happens to be a log-hazard, and its `frame()` still names the column `log_hazard`.

Risk scores and survival functions are separate capabilities, because each occurs without the other: `MaskedMean`
over Cox branches without a fused endpoint gives risk scores with no baseline, and DeepHit or discrete-time models give
survival curves with no canonical risk score. `CoxOutput` has both. Merging them would make survival curves a runtime
property, so `IntegratedBrier` could no longer be refused by type. Likewise, class probabilities and vote scores stay
two types under `ClassScores`, because `predict_proba`, calibration metrics and a fused `ClassificationEndpoint`
must refuse vote scores; one class with a flag would move those checks from build time to run time.

Each output stores only its raw values: logits, not probabilities, and log-hazards, not curves. Views are computed
when asked for, so `ModelOutput`'s split between `output` and `prediction` goes, and no softmax runs on every
training step. A field is never named after a view it is not: `ClassProbabilities` keeps `logits`, and its `scores`
are its probabilities. Prediction columns stay strings such as `probability[<class>]` (Consequences).

**Typed outputs are always model outputs.** A head returns a raw tensor. It never sees `ids` or `defined`, because it
runs only on the patients with a prediction, and it holds no state. The endpoint (§3) builds the typed output.

**Before the endpoint, everything is tensors.** Raw inputs and embeddings are not typed. ADR 0002 §1 and §3 chose
plain tensors between encoders so that any stock `nn.Module` works there, and nothing downstream needs to know what an
embedding means, only its shape. Typed outputs earn their keep because metrics, fusion and `predict` need to know what
a prediction means.

## 2. Producers and consumers declare output types

**Options:**

- `target_type` on heads and `head_kinds` in `FusionContext`, plus metrics parsing column names (before). Target type
  is the wrong level of detail: what fusion and metrics need is the output space.
- A mixin that forces heads, fusion methods and models to define `output_type`. The three need different signatures
  (a class attribute, a function of the input types, a derived property), and heads, now any `nn.Module`, and early
  and intermediate fusion methods, which output tensors, have no output type.
- Declared output types where outputs are produced and consumed (chosen).

**Choice:**

- An endpoint declares the output type it produces, as a class attribute.
- A late fusion method declares `output_type(input_types)`, the type it produces from the branch output types, and
  raises `TypeError` for types it cannot fuse. It declares only what fusing keeps: `MaskedMean` over `CoxOutput`s
  produces a `RiskScore`, because each branch's baseline belongs to its own log-hazards and none belongs to their
  average; over `ClassProbabilities` it produces `ClassProbabilities`. `MajorityVote` over `ClassProbabilities`
  produces `VoteScores`.
- What combining means belongs to the fusion method, not the output type, so each late fusion method names the types
  it combines, as a metric names the types it scores. `MaskedMean` averages the `risk` of `RiskScore`s, including
  `CoxOutput`s, into a `RiskScore`, and the `logits` of `ClassProbabilities` into `ClassProbabilities`, by `isinstance`;
  `MajorityVote` votes with the `probabilities` of `ClassProbabilities`. Averaging logits rather than probabilities is
  `MaskedMean`'s choice: another method, such as a weighted mean or a product of experts, could combine the same type
  differently, which no single declaration on the type could describe. A new output type that should be averaged
  means a change to `MaskedMean`.

  The type declares only what is true of its output space whatever combines it: `comparable_across_models`, `False`
  for risk scores, whose offset differs from model to model, so `MaskedMean` refuses patients without every branch.
  It is `False` by default, so a type that does not declare it is never mixed patient by patient by mistake;
  `ClassProbabilities` and `VoteScores` set it to `True`.

  This reverses an earlier choice (this ADR, before 2026-11-03), under which each output type declared `raw`,
  `combined_type` and `from_raw` so that late fusion methods need not change when a type is added. Only `MaskedMean`
  ever read them, and they fixed one method's notion of combining on the type. A registry, in which each type
  registers its rule with `MaskedMean`, was rejected: registered next to the type, it makes the type's module import
  the fusion method; registered next to `MaskedMean`, it is the `isinstance` chain with more steps, plus global
  mutable state.
- An endpoint also declares `accepts`, the type it takes as the fused endpoint of a `LateFusion` model (§4):
  `CoxEndpoint` accepts a `RiskScore` and makes it a `CoxOutput`, `ClassificationEndpoint` accepts
  `ClassProbabilities` (logits).
- A model's `output_type` is abstract on `BaseModel`: its endpoint's, or for `LateFusion` its fused endpoint's or its
  fusion method's. Endpoints declare their type as a class attribute, because their class fixes it; models compute
  theirs as a property, because it depends on the parts they were built with.
- A metric declares the weakest type it needs, and reads only what that type promises: `HarrellC` needs a
  `RiskScore` and reads `risk`, `IntegratedBrier` a `SurvivalFunction` and calls `survival`, `AUROC` `ClassScores`
  and reads `scores` (it only ranks), and a calibration metric such as log-loss `ClassProbabilities`. A new output
  type that provides one of these, such as a DeepHit output giving a risk score, is scored with no change to the
  metrics. That much coupling is inherent: a metric depends on what it consumes.

Compatibility is one rule: the producer's type must satisfy the consumer's, by `issubclass`. This replaces
`target_type`, `head_kinds`, `input_space`, the column regexes and the `TimeToEvent` rule in `MaskedMean.check`.
`MaskedMean` over incomplete branches is refused for risk scores because of the offset, not because the target is
`TimeToEvent`. `LateFusion` no longer uses the first branch's head to build its prediction and name its columns.

Out-of-fold pooling and `bootstrap_ci` can read the same declaration to refuse or warn about pooling relative outputs.

## 3. Endpoints wrap the head and the loss

**Options:**

- The head owns the loss (before). The head must be a special class, not any `nn.Module`, and the loss is fused to
  a linear layer.
- The model takes a free `loss=` argument. The loss determines what the output means, so this separates the loss
  from the output type it implies. A Cox model with a ranking loss would still claim valid survival curves.
- Users write an endpoint to change the loss. Too complicated for a common need.
- An endpoint that wraps the head and the loss, and takes a torch-style `loss_fn=` where the output type does not depend
  on the loss (chosen).

**Choice:** A model takes an endpoint, never a head on its own:

```python
Unimodal(
    "clinical",
    encoders=[MLP(...)],
    endpoint=ClassificationEndpoint(head=nn.Linear(64, 3), loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1)),
)
```

`BaseEndpoint` owns, in the order `fit` uses them:

1. `output_type`, known when the model is built (§2).
2. `check_target` and `bind`, when `fit` starts: raise if the target does not suit it, then take what it needs from
   the target, such as the class order.
3. `forward`, which runs the head on the embedding, and `to_output`, which builds the typed output and attaches the
   endpoint's state. As a fused endpoint, with no head, `adopt` takes the fusion method's output instead, which must
   be an `accepts`.
4. `loss`, the loss of a batch, or `None` when the batch has no signal.
5. `fit_state`, once training ends (§6).

The head is any `nn.Module` from the embedding to the raw output, as encoders are in ADR 0002. The user still sizes it
(`nn.Linear(64, 3)`), so its initial weights are drawn when the model is built, and the dry run checks its width
against the target. Creating the last layer lazily from the target was rejected: its initial weights would depend on
the RNG state at `fit`, breaking the promise that `fit` starts from the weights the model was built with.

**Choosing a loss.** In classification, and later regression, the output means the same whatever reasonable loss
trains it: cross-entropy with weights, label smoothing or a focal loss all train logits; MSE, L1 or Huber all train a
point prediction. So these endpoints take any `loss(input, target)` in torch's convention, such as
`nn.CrossEntropyLoss(...)` or `nn.HuberLoss()`, and jsonargparse builds them from YAML.

**`decomposable_loss`** says whether the loss is a mean over patients, so that accumulating gradients reproduces a
larger batch. Only the training setup reads it, to refuse `accumulate_grad_batches` when it would change the
gradients. It is about how the loss behaves, not what the output means, so it is not part of the output type. It is
an instance property on `BaseEndpoint`, because with a user-supplied loss it depends on the instance:

- An endpoint whose loss the user supplies takes it as an argument, `decomposable_loss: bool = False`. Only the user
  knows whether their loss averages over patients; inspecting the loss (its `reduction` attribute) only approximates
  that, and says nothing about a plain function or a custom loss. The default is `False` because most users never
  accumulate gradients: they are not asked a question about a feature they don't use, and a user who accumulates is
  refused, with a message to pass `decomposable_loss=True` if their loss is a mean over patients. A wrong `True` would
  give silently wrong gradients; a wrong `False` only refuses accumulation.
- An endpoint with a fixed loss knows the answer and takes no argument. `CoxEndpoint` always returns `False`, because its
  risk set is the batch.
- The endpoint does not check the loss's `reduction`. `"none"` fails loudly, because the loss is not a scalar, and
  `"sum"` is a legitimate choice that only changes the loss scale.

A model combines its endpoints' answers with its own: late fusion over patients who lack some modalities averages
each branch over a different subset, so it is not decomposable even when every endpoint is.

In survival, the loss is the statistical model: the Cox partial likelihood is what makes a Breslow baseline valid, and
DeepHit's loss gives a different output type. So each family is its own endpoint, `CoxEndpoint(head, ties=...)` and later
`DeepHit(...)` or a discrete-time endpoint, which knows its loss. The Cox loss also returns `None` when no event has
anyone else at risk, and computes in float32 outside autocast.

## 4. Models

**Context:** `models.py` is to be refactored. `Unimodal`, `EarlyFusion` and `IntermediateFusion` share one shape:
inputs, encoders, fusion of raw inputs or embeddings, one endpoint. `LateFusion` has another: whole models, their
typed outputs, fusion of those outputs.

**Choice:**

```text
BaseModel                      forward(batch) -> BaseOutput, output_type, loss, fit_state
├── SingleEndpointModel        encoders → [fusion of tensors] → endpoint
│   ├── Unimodal
│   ├── EarlyFusion
│   └── IntermediateFusion
└── LateFusion                 branches: dict[str, SingleEndpointModel] → fusion of outputs → [endpoint]
```

- A `LateFusion` branch is any single-endpoint model, as now. `LateFusion` doesn't nest for now; the structure
  allows it later.
- `LateFusion(branches, fusion, endpoint, branch_weight=1.0)` takes the endpoint of the fused output, with no head,
  such as `CoxEndpoint(None, ties="efron")`. The argument has no default, so the user always says what owns the fused
  output, but it may be `None`.
- Which endpoints are valid is decided by types, never by naming a fusion method, as `models.py` requires: when the
  model is built, `fusion.output_type(branch types)` must satisfy the endpoint's `accepts`. `MajorityVote` produces
  `VoteScores`, which no classification endpoint accepts because vote scores are not logits, so it can only take
  `None`; a future voting method gets the same rule for free.
- With `endpoint=None`, the model outputs what the fusion method makes, with no fused loss and no fused state.
  `MaskedMean` over Cox branches then outputs a `RiskScore`: `HarrellC` works on it, and `IntegratedBrier` or
  `predict_survival_function` are refused by type.
- Each branch's endpoint holds that branch's state and the fused endpoint holds the fused state, so `branch=` and the
  fused prediction work the same way. This answers who fits the Breslow baseline of a fused Cox output.
- Deriving the fused endpoint, by copying the first branch's without its head, was rejected: it hides what owns and
  trains the fused output, and makes the first branch special.

**Fusion methods** implement one of two interfaces under a shared `FusionMethod`, which holds only `defined`:

- `TensorFusion`, for early and intermediate fusion: `forward(Mapping[str, Tensor], present) -> Tensor`, and
  `check_widths(widths)`, called in the dry run with the real widths.
- `OutputFusion`, for late fusion: `output_type(input_types)`, called when the model is built,
  `check_branches(present, input_types)`, called for each dataset, and `forward(Mapping[str, BaseOutput], present) ->
  BaseOutput`, a typed output for the rows it is given.

The hooks accept anything by default, so a method overrides only the rules it has, with no `super()` call. The model,
not the method, checks that the method defines every patient the model must predict for: `required` is the model's
idea (a `LateFusion` branch need not cover patients with none of its modalities), and the method's part is
`defined`.

`FusionContext` is removed. It let one hook, `check(context)`, run when the model was built, for each dataset and in
the dry run, so each call padded the information it lacked with placeholders (an empty presence table, `None` widths,
no output types), each rule had to tell which call it was in, and a method that overrode `check` without calling
`super().check(context)` silently skipped the rule that every required patient is defined. One hook per moment
receives only real values. Keeping `check` with keyword arguments instead of the dataclass was rejected: it unpacks
the same bag. No rule needs widths and presence together, since widths are the same for every patient.

A method that works in every stage subclasses both. `MaskedMean(TensorFusion, OutputFusion)` declares `forward` with
two `@overload`s, one per interface, and its implementation dispatches on whether its inputs are tensors or outputs.
That `isinstance` tests only which of the two interfaces it was called through, not any particular type. mypy checks
the overloads against both bases, and refuses a class that subclasses both but implements one signature. Calls
through `nn.Module.__call__` are typed `Any` by torch, so the overloads constrain implementers more than callers.
`Concat` is a `TensorFusion` and `MajorityVote` an `OutputFusion`.

The interface replaces `stages`: `LateFusion` takes an `OutputFusion` and `EarlyFusion` and `IntermediateFusion` a
`TensorFusion`, checked when the model is built, and jsonargparse checks the same annotations in YAML configs.
No current method is allowed in one of early and intermediate fusion but not the other, so they share one interface.
`input_space` goes, because a late fusion method reads the view it needs from the type (§2).

Rejected: one interface whose `forward` takes `Any` (before), which cannot be typed and made every method carry an
`output_type`; two public classes for the one method that works in both settings, which gives users a second name
for the same mean; and a factory that turns a user's `MaskedMean` into a hidden stage-specific class, which moves the
dispatch into the models or a spec object rather than removing it, and leaves the user holding an object that is not
the one in the model.

**Repetition between models.** The models' constructors repeated the same checks. `SingleEndpointModel.__init__` now
checks and stores the endpoint, each fusion model names its `stage` as a class attribute instead of repeating the
string, and one helper checks a fusion method and runs its construction-time check. A mixin holding the fusion
method was rejected: `LateFusion` and the single-endpoint fusion models have different bases, and a cooperative
`__init__` across `nn.Module` subclasses is easy to get wrong for little gain.

## 5. Losses

**Options:**

- Late fusion sums the branch losses (before).
- Late fusion trains the fused output only. The branches are then not trained to be useful on their own, which
  `branch=` relies on.
- The fused loss plus the branch losses as auxiliary terms (chosen).

**Choice:**

- The endpoint computes the loss on the output it builds, a branch's or a fused one (§3).
- A fusion method may add its own loss terms, so that future trainable fusion methods can be trained and regularised.
- Late fusion's loss is the fused endpoint's loss plus `branch_weight` (default 1) times the sum of the branch
  losses, which is the usual deep supervision pattern. Each part is reported as `loss/fused` and `loss/<branch>`, so
  no branch may be called `fused`.
- The fused loss exists exactly when there is a fused endpoint. With `endpoint=None`, only the branch losses train the
  model, as before; the user chose that by writing `None`.

## 6. Fitted endpoint state

**Options:**

- The `Pipeline` owns the Breslow baseline, fitted on training and validation patients (before). `survival_times`
  can't be used with `branch=`, and a model that predicts curves itself, such as DeepHit or discrete-time models,
  has no way to provide them.
- The output type fits its own state, and the model stores it. Outputs would then both carry data and fit things.
- The endpoint fits and holds its state, fitted on the training patients only (chosen).

**Choice:**

- State is bound in two phases, both on the endpoint: from the target when `fit` starts (`bind`, e.g. the class
  order), and from the training patients' outputs once training ends (`fit_state`, e.g. `CoxEndpoint.baseline_`).
  `to_output` attaches it to every output, so outputs fit nothing themselves.
- The fitted population is `train_ids_`, for the baseline and for the IPCW censoring distribution alike. Validation
  patients were used to choose the model, not to fit it. To confirm against the libraries before implementing: pycox's
  `compute_baseline_hazards` and sksurv's `CoxPHSurvivalAnalysis` fit the baseline on the training data, and sksurv's
  `concordance_index_ipcw` estimates censoring from the training data, as in Uno et al.

## 7. Each check runs at the earliest point its information exists

This extends ADR 0002 §4 from widths to targets, outputs and metrics.

**Options:**

- Check everything at `fit`, including what is known when the model is built.
- Check everything when the model is built. Targets, presence and real widths don't exist yet.
- Add a separate dummy `fit`. The dry run already is one.
- Check each rule at the first stage where its information exists (chosen).

**Choice:**

| Stage | Information that exists | Checks |
| --- | --- | --- |
| Model built (`__init__` of models, endpoints, fusion methods) | output types, fusion interfaces | late fusion accepts the branch output types and its fused endpoint's, the fusion method has the interface its model needs |
| `fit` starts, before training | target, modalities, presence, metrics given to the `Pipeline` | the endpoint suits the target, the fusion method defines every required patient and accepts the branches each patient has (`check_branches`), the early-stopping metric suits the output type |
| Dry run | real tensors | widths (`check_widths`), the head's width against the target, and the loss on the real target tensors |

`Pipeline` is an sklearn `BaseEstimator`, whose `__init__` must not validate because `clone` and `set_params` rely on
that. So `Pipeline` checks its early-stopping metric when `fit` starts, not when it is built. Model components are
`nn.Module`s, not sklearn estimators, so they validate when built, as `LateFusion` does with
`fusion.output_type(branch types)`. Metrics passed to `evaluate` or `cross_validate` are checked when those are called, before predicting
or before the first fold.

A declared type can promise more than is available during training: a `CoxOutput` has no baseline until training
ends, so `survival` raises before then. Early stopping on `IntegratedBrier` passes the type check, so it still needs a
rule about timing (refuse it, or fit a provisional baseline each epoch).

## 8. Predictions and metrics

**Choice:**

- `predict` returns every prediction as a DataFrame, from the output's `frame()`, as users are used to.
  `predict_proba` gives class probabilities and `predict_survival_function(times)` gives survival curves, the latter
  following sksurv's name. The `survival_times` argument of `predict`, `evaluate` and `cross_validate` goes.
- Metrics receive the typed output and the target instead of frames plus `EvalContext`, so they stop parsing column
  names and re-checking what the target guarantees. `EvalContext` is removed.
- Metrics hold their own times, as `UnoC(tau)` and `TimeDependentAUC(time)` already do: `IntegratedBrier(times=...)`
  asks the output for S(t).

**The training target.** `UnoC`, `TimeDependentAUC` and `IntegratedBrier` weight each test patient by 1/G(t), where
G is the censoring survival estimated with Kaplan–Meier on the training patients (IPCW). So they need the training
outcomes as well as the test ones, which `EvalContext.train_target` used to carry.

**Options:**

- Pass the training target on every call, `metric(output, target, train_target)` (chosen). This is sksurv's
  convention (`concordance_index_ipcw(survival_train, survival_test, estimate, tau)`), and metrics stay stateless,
  so one metric object serves every fold. Metrics that don't need it, such as `HarrellC` and `AUROC`, ignore it.
- Fit IPCW metrics first, `UnoC(tau).fit(train_target)`, then call them. G is estimated once rather than on every
  bootstrap resample, but Kaplan–Meier on a cohort takes milliseconds, and a stateful metric must be refitted in every
  fold of `cross_validate` and fails if a user forgets to fit it before `bootstrap_ci`.
- The endpoint fits G with its other state and attaches it to the output, so metrics take only `(output, target)`.
  This puts evaluation logic in the model, and predictions that didn't come from a kalecancer model couldn't be
  scored.

**Choice:** `Metric.__call__(output, target, train_target)`. `Pipeline.evaluate`, `cross_validate` and early stopping
pass the training target of the fitted model, and users pass it to `bootstrap_ci` themselves.

**How it fits together:**

- `target` and `train_target` are `BaseTarget`s, not frames. `target` must hold at least the output's patients, so a
  dataset's whole target can be passed; `train_target` holds the training patients only, which the new
  `BaseTarget.subset(ids)` gives. `Pipeline.train_target_` is now such a target, `data.target.subset(train_ids_)`.
- A metric declares `requires`, the weakest output type it scores, and `check_output_type(output_type)` raises
  `TypeError` against it. `__call__` checks the output and then calls `score`, which subclasses implement.
  `Pipeline.fit` checks the early-stopping metric against the model before training, and `evaluate` and
  `cross_validate` check every metric before predicting and before the first fold.
- Metrics read class scores through `ClassScores.scores`, which `ClassProbabilities` gives as probabilities, not
  logits: logits rank the classes of one patient, but not one class across patients, which AUROC needs.
- `Pipeline.predict_output(data, branch=None)` returns the typed output, for scoring outside `evaluate`, e.g.
  `bootstrap_ci(metric, output, target, train_target, baseline=None, baseline_train_target=None, ...)`.
  `bootstrap_ci` resamples in the target's order, so its result does not depend on the order of the output.
- A metric that needs a `SurvivalFunction` cannot be monitored by early stopping: the state that gives survival
  functions, such as a Breslow baseline, is fitted only once training ends (§7). The rule reads `requires`, not
  the metric's class.

## 9. Batching

**Choice:** Training batches are shuffled by default. The design leaves a hook for the endpoint to supply a sampler,
to be implemented later:

- For Cox, the only hard requirement is that each batch has an event with someone else at risk, which an event-aware
  sampler can guarantee.
- For classification, the convention is no rebalancing by default, because balanced sampling distorts the predicted
  probabilities. Class weights in the loss are the usual answer to imbalance.

## Open questions

- **Class weights from the training target.** `nn.CrossEntropyLoss(weight=...)` takes fixed weights. Weights computed
  at `fit`, as sklearn's `class_weight="balanced"` does, would need an endpoint option that builds them in `bind`.
- **A `loss_fn=` for `CoxEndpoint`,** taking functions with torchsurv's signature `(log_hz, event, time)`. Left out until
  someone needs it.
- **A default head.** Users always size the head now. A convenience such as `CoxEndpoint.linear(in_dim)` could cover the
  common case.
- **The names** of `BaseOutput`, its subclasses, `BaseModel` and `SingleEndpointModel`. Endpoints are named
  `<Family>Endpoint` (`CoxEndpoint`, `ClassificationEndpoint`), which keeps them apart from targets such as
  `Classification`. Their loss method is `loss`, as on models, so a user-supplied loss is the argument `loss_fn`:
  configs store each `__init__` argument under its own name, so the two can't share one.
- **Labels for fitted encoders.** TabICL labels its context with `strata`, which for `TimeToEvent` is the event flag,
  so a patient censored at 3 months looks like a long-term survivor. The endpoint, not the splitter, should decide
  what labels an encoder sees.
- **How an output marks its per-patient fields.** `BaseOutput` implements `select`, `scatter`, `cat` and
  `detach_cpu` once, so it must tell per-patient fields (`risk`, `logits`, `defined`), which are indexed by
  patient, from fields shared by the batch (`classes`, `baseline`), which are copied as they are. The sketch in
  `kalecancer/model/outputs.py` treats a field as per-patient exactly when it holds a `Tensor`. A new output type
  then needs no code for these methods, but nothing enforces the rule: a shared tensor, such as discrete-time bin
  edges, would be indexed by patient, and a per-patient field that is not a tensor would be silently left out. The
  alternative is to tag each per-patient field (`field(metadata={"per_patient": True})`), which is explicit and
  can't misfire on a shared tensor, but is more verbose and fails the same way when a tag is forgotten.
  **Decided for now (2026-11-04):** keep "a `Tensor` is per patient", which holds for every output type so far.
  Revisit it when the first type with a shared tensor, such as discrete-time bin edges, is added.
- we should provide some default endpoints with default paramerters for all endpoints. i.e. classificationendpoint  where no loss or head is needed to be provided, and simple default ones, chosen by us, are used.

## Out of scope

- Separating `strata` from splitting, and splitting by groups such as centre, unless it falls out of the labels
  question above.
- The naming of fusion methods (`Concat` or `FusionConcat`). It is unrelated, and belongs with kale core.
- `dump_config` for state bound at `fit`.
- Saving a fitted pipeline. Fitted endpoint state (class order, the baseline, later class weights or calibration)
  must eventually be saved with the weights; the Breslow baseline is not a tensor, so it is not in the state dict
  today. To be raised as its own issue.

The design must not rule out:

- several targets per dataset (multi-task, for example overall survival with recurrence, or Cox with an auxiliary
  classifier);
- survival targets beyond right censoring: competing risks, left truncation, interval censoring.

## Consequences

To be filled in as the implementation lands. Expected so far:

- `model/heads.py` is replaced by `model/endpoints.py`. `CoxHead` and `ClassificationHead` become `CoxEndpoint` and
  `ClassificationEndpoint`, and models take `endpoint=` instead of `head=`.
- `ModelOutput`, `EvalContext`, `FusionContext`, `FusionMethod.check`, `FusionMethod.stages`, `FusionMethod.input_space`, head `target_type` and
  `columns(target)`, `Pipeline.baseline_` and the `survival_times` argument are removed or replaced.
- `LateFusion` takes a third, required argument, `endpoint`, and trains the fused output when it is given. Late fusion
  methods take and return typed outputs, and `MajorityVote` predictions are `score[c]` columns.
- The Breslow baseline is fitted on the training patients only, not training and validation, so baselines and
  survival curves change for pipelines with a validation split.
- The `Metric` call signature changes, so every metric changes.
- Prediction frames keep string columns, such as `probability[<class>]`: metrics no longer read them, so only
  display depends on them, and `predict_proba` gives the classes themselves as columns. `(prefix, class)` columns
  were dropped as churn for no remaining gain.
- `predict`, `evaluate` and `cross_validate` lose `survival_times`; survival curves come from
  `predict_survival_function`, and `IntegratedBrier(times)` holds its own times. `EvalContext` is removed, and
  `bootstrap_ci` takes typed outputs and targets.
- `docs/extending/endpoints.md` and `docs/extending/fusion.md` are rewritten, and `docs/architecture.md` updated.
- Supersedes the head contract in `docs/extending/endpoints.md` and extends ADR 0002 §4.
