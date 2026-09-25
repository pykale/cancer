# Multimodal fusion

How modalities are combined into one prediction. Fusion is a choice of model and fusion method, so comparing
approaches changes the model dictionary, not the encoders, heads or `Pipeline`.

## Models

| Model | Combines | Structure |
| --- | --- | --- |
| `Unimodal` | nothing | stages, then head |
| `EarlyFusion` | raw vectors | fuse, then one stage list, then head |
| `IntermediateFusion` | encoded features | stages per modality, then fuse, then head |
| `LateFusion` | branch predictions | branch models trained jointly, then a combiner |

Early fusion needs modalities that are already vectors, such as tables.

A `LateFusion` branch is a `Unimodal`, `EarlyFusion` or `IntermediateFusion` model. The loss is the unweighted sum of
the branch losses, and `predict(..., branch=name)` returns a single branch's predictions.

## Fusion methods

| Method | Used by | Combines | Missing modalities |
| --- | --- | --- | --- |
| `Concat` | early, intermediate | vectors, concatenated | Not handled: every patient needs every modality |
| `MaskedMean` | early, intermediate | equal-width vectors, averaged over those present | Handled |
| `MeanLogits` | late | branch outputs (logits or log-hazards), averaged over present branches | Handled, except Cox with differing subsets |
| `MajorityVote` | late | each branch's most probable class, ties broken by mean probability | Handled |

`MeanLogits` refuses Cox branches unless every patient has every branch: each branch's log-hazard has an arbitrary
offset, so averaging different subsets would reorder patients. `MajorityVote` needs classification heads.

## Usage

```python
from torch import nn

from kalecancer.model import (
    ABMIL, MLP, ClassificationHead, Concat, CoxHead, IntermediateFusion, LateFusion, MeanLogits, Unimodal,
)
from kalecancer.pipeline import Pipeline

# Intermediate: encode each modality, concatenate, one Cox head.
intermediate = IntermediateFusion(
    encoding={
        "clinical": [MLP(in_dim=12, hidden_dims=[64], out_dim=64, dropout=0.1)],
        "wsi": [ABMIL(in_dim=1024, hidden_dim=256, attention_dim=128, dropout=0.25), nn.Linear(256, 64)],
    },
    fusion=Concat(),
    head=CoxHead(in_dim=128, ties="efron"),
)

# Late: a classifier per modality, logits averaged.
late = LateFusion(
    branches={
        "clinical": Unimodal("clinical", [MLP(12, [64], 64, 0.1)], ClassificationHead(64, n_classes=2)),
        "wsi": Unimodal("wsi", [ABMIL(1024, 256, 128, 0.25), nn.Linear(256, 64)], ClassificationHead(64, n_classes=2)),
    },
    combine=MeanLogits(),
)

pipeline = Pipeline(model=intermediate, ...)
```

The dataset supplies the modality names and the target. Runnable versions are in
[examples/hancock/](../examples/hancock/): `survival_intermediate.py` (intermediate, Cox), `classification_late.py`
(late, classification) and `configs/intermediate_cox.yaml` (the same model as YAML).

Widths are checked at construction: a stage list must end in `(n, d)` vectors, adjacent `in_dim`/`out_dim` must
match, and the head's `in_dim` must match the fusion output (`Concat` sums the widths, `MaskedMean` needs them equal).

## Missing modalities

Every model follows one rule, so absence needs no placeholder values:

1. Stages run only on patients who have the modality.
2. Outputs are scattered back into the batch with NaN for the others.
3. Fusion selects the defined rows. It never multiplies by a mask, since `NaN * 0` is `NaN` and would corrupt
   gradients.
4. Heads run on defined rows only. Patients with no prediction get NaN.

Which patients a model can handle is decided by the fusion method and `MultimodalDataset(required_modalities=...)`.
`model.check(data)` runs at the start of `fit` and raises if the data has missing patterns the fusion cannot handle,
or patients with none of the modalities. The library does not include modality dropout or learned placeholder
embeddings.

## Extending

 A new fusion method is an `nn.Module` with `handles_missing`, `output_dim(widths)` and
`forward(z, present)`. A late combiner instead sets `input_space` (`"output"` or `"prediction"`) and implements
`check_branches`, with `columns` if it works on predictions.

