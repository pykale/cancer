# Multimodal fusion API

Fusion components for combining modalities into a single survival prediction. These
are staged ahead of the tabular merge: the WSI pipeline in
[`examples/hancock_wsi_survival/`](../examples/hancock_wsi_survival/) is unimodal and does not use
them yet.

The design goal is that a researcher changes **one config value** to compare fusion
approaches, without touching encoders, heads, or training code.

## Two layers

**Strategy** — *what* gets combined (`kalecancer.model.multimodal`):

| Strategy | Mechanism | When to use |
| --- | --- | --- |
| `early` | Each modality encoded → **features fused** → one shared head | Learns cross-modal structure before any prediction is made |
| `late` | Each modality encoded → **predicts independently** → risks combined | The baseline fusion has to beat; fully interpretable per modality |
| `hybrid` | **Both**: an early-fusion trunk plus late-style per-modality heads | Cross-modal learning while every encoder stays supervised |

```
early                          late                        hybrid
─────                          ────                        ──────
enc_a ─┐                       enc_a → head_a ─┐           enc_a ─┬─→ head_a ─┐
       ├→ fuse → head → risk                   ├→ risk            │           │
enc_b ─┘                       enc_b → head_b ─┘           enc_b ─┼─→ head_b ─┤
                                                                  └→ fuse → head → risk
```

Early fusion is **feature-level, not raw-input-level**: modalities are encoded first,
because a gigapixel slide and a clinical table share no common input space. What makes
it *early* is that fusion happens **before any prediction**, so the shared head learns
from cross-modal structure rather than from separate verdicts.

Hybrid is literally early + late — `HybridFusionSurvival` subclasses
`EarlyFusionSurvival` and adds the per-modality heads.

**Method** — *how* features combine, used by `early` and `hybrid`
(`kalecancer.model.embed.multimodal_fusion`):

| Method | Mechanism | Notes |
| --- | --- | --- |
| `concat` | Concatenate then project | Simple, strong baseline |
| `poe` | Product of Gaussian experts | **Preferred when modalities can be missing** — an absent expert drops out of the product without retraining |
| `lowrank` | Low-rank tensor factorisation | Models multiplicative interactions at linear parameter cost |

Every method returns the same `output_dim`, so the survival head is unchanged when
the method changes.

### End-to-end vs pre-extracted features

Encoders here are trained end to end with the survival loss. Some taxonomies reserve
"early fusion" for fusing features from *separately trained* extractors and call the
end-to-end version *joint fusion*. Freeze the encoders to get the strict form — which
is already the case for WSI, whose patch features come from a frozen foundation model.

## Usage

All three strategies take the **same arguments**, so switching is a one-word change:

```python
from kalecancer.model import build_multimodal_survival, multimodal_cox_loss
from kalecancer.model.embed import AttentionMIL, BagEncoder

encoders = {
    "wsi": BagEncoder(AttentionMIL(input_dim=1024, hidden_dim=256)),
    "clinical": TabularEncoder(...),  # any module returning (batch, dim)
}
latent_dims = {"wsi": 256, "clinical": 64}

# Early: encode each modality, fuse the features, predict once.
model = build_multimodal_survival("early", encoders=encoders, latent_dims=latent_dims, fusion="poe", fused_dim=64)

# Late: encode and predict per modality, then combine the risks.
model = build_multimodal_survival("late", encoders=encoders, latent_dims=latent_dims)

# Hybrid: the early trunk plus a head on every modality.
model = build_multimodal_survival("hybrid", encoders=encoders, latent_dims=latent_dims, fusion="poe", fused_dim=64)

output = model({"wsi": bags, "clinical": table}, mask)
loss = multimodal_cox_loss(output, event, duration, auxiliary_weight=0.3)
```

`auxiliary_weight` only bites when the model produces per-modality risks, so the same
loss call works for all three strategies.

`output.risk` is the patient-level log partial hazard; `output.modality_risk` holds
the per-modality risks from late fusion or the hybrid auxiliary heads.

Encoders are **injected, not built here**, so any modality plugs in as long as it
returns `(batch, latent_dim)`. `BagEncoder` adapts `AttentionMIL` to that interface
while keeping attention available on `.last_attention` for interpretation.

## Configuration

```yaml
FUSION:
  STRATEGY: hybrid          # early | late | hybrid
  METHOD: poe               # concat | poe | lowrank   (early and hybrid)
  FUSED_DIM: 64
  RANK: 4                   # lowrank only
  AUXILIARY_HEADS: True     # hybrid only
  AUXILIARY_WEIGHT: 0.3     # hybrid only
  COMBINE_RISKS: False      # hybrid only: also merge decisions, not just features
  MODALITY_DROPOUT: 0.2
```

`COMBINE_RISKS` decides how completely hybrid merges the two strategies. Left `False`,
the fused trunk makes the prediction and the per-modality heads only supply auxiliary
supervision and interpretability. Set `True`, the per-modality risks are blended into
the final risk as well, combining early and late at the **decision** level too;
absent modalities do not vote.

## Missing modalities

Missing modalities are a design requirement, not an edge case: real cohorts rarely
have every modality for every patient. A `(batch, num_modalities)` mask (1 = present)
is carried into every strategy and fusion block.

| Mechanism | Behaviour |
| --- | --- |
| Modality mask | Marks which modalities each patient actually has |
| Learned placeholder | `concat`, `lowrank` and `early` substitute a learned embedding for an absent modality — a zero vector would be indistinguishable from a genuine all-zero latent |
| Precision zeroing | `poe` gives an absent expert negligible precision, so it leaves the product entirely |
| Prior expert | `poe` includes a prior so a patient missing *every* modality still yields a defined result instead of `0/0` |
| No vote | `late` renormalises its weights over present modalities only |
| Modality dropout | Randomly marks modalities absent during training; always keeps at least one |

## Scope

These are model-level APIs. Patient matching across modalities and a multimodal
trainer are provided by the multimodal cohort loader, not by this module.

Fusion operates on encoded representations rather than raw inputs, since a 3D volume,
a variable-size patch bag, and a ~50-dimensional vector share no common input space.
The Cox risk set spans the mini-batch, so the batch-size requirements of the unimodal
pipeline apply unchanged.
