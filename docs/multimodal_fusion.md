# Multimodal fusion API

Fusion components for combining modalities into a single survival prediction. These
are staged ahead of the tabular merge: the WSI pipeline in
[`examples/wsi_survival/`](../examples/wsi_survival/) is unimodal and does not use
them yet.

The design goal is that a researcher changes **one config value** to compare fusion
approaches, without touching encoders, heads, or training code.

## Two layers

**Strategy** — *where* the modalities meet (`kalecancer.model.multimodal`):

| Strategy | Mechanism | When to use |
| --- | --- | --- |
| `early` | Features concatenated at the input, one joint encoder | Commensurable inputs only; a slide must already be pooled to a vector |
| `late` | Independent per-modality model and head, predictions combined | The baseline fusion has to beat; fully interpretable per modality |
| `hybrid` | Per-modality encoders → latent fusion → shared head, plus auxiliary per-modality heads | Learns cross-modal interactions while keeping every encoder supervised |

**Method** — *how* latents combine inside `hybrid`
(`kalecancer.model.embed.multimodal_fusion`):

| Method | Mechanism | Notes |
| --- | --- | --- |
| `concat` | Concatenate then project | Simple, strong baseline |
| `poe` | Product of Gaussian experts | **Preferred when modalities can be missing** — an absent expert drops out of the product without retraining |
| `lowrank` | Low-rank tensor factorisation | Models multiplicative interactions at linear parameter cost |

Every method returns the same `output_dim`, so the survival head is unchanged when
the method changes.

## Usage

```python
from kalecancer.model import build_multimodal_survival, multimodal_cox_loss
from kalecancer.model.embed import AttentionMIL, BagEncoder

encoders = {
    "wsi": BagEncoder(AttentionMIL(input_dim=1024, hidden_dim=256)),
    "clinical": TabularEncoder(...),          # any module returning (batch, dim)
}

model = build_multimodal_survival(
    "hybrid",
    encoders=encoders,
    latent_dims={"wsi": 256, "clinical": 64},
    fusion="poe",
    fused_dim=64,
)

output = model({"wsi": bags, "clinical": table}, mask)
loss = multimodal_cox_loss(output, event, duration, auxiliary_weight=0.3)
```

`output.risk` is the patient-level log partial hazard; `output.modality_risk` holds
the per-modality risks from late fusion or the hybrid auxiliary heads.

Encoders are **injected, not built here**, so any modality plugs in as long as it
returns `(batch, latent_dim)`. `BagEncoder` adapts `AttentionMIL` to that interface
while keeping attention available on `.last_attention` for interpretation.

## Configuration

```yaml
FUSION:
  STRATEGY: hybrid          # early | late | hybrid
  METHOD: poe               # concat | poe | lowrank   (hybrid only)
  FUSED_DIM: 64
  RANK: 4                   # lowrank only
  AUXILIARY_HEADS: True
  AUXILIARY_WEIGHT: 0.3
  MODALITY_DROPOUT: 0.2
```

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

## Relationship to PyKale

`ConcatFusion` and `ProductOfExpertsFusion` build directly on
`kale.embed.multimodal_fusion.Concat` and `.ProductOfExperts`.

`LowRankFusion` re-implements low-rank tensor fusion instead of reusing PyKale's
`LowRankTensorFusion`, which has two defects that make it unusable here:

- its factors and fusion weights are stored in a plain list of device-moved tensors,
  so they are **not registered as module parameters** — `.parameters()` is empty and
  an optimiser would never train them;
- its device is hardcoded to `cuda:0`, so it returns CUDA tensors even for CPU inputs.

`tests/model/embed/test_multimodal_fusion.py` asserts against both failure modes for
every registered method. Worth upstreaming a fix to PyKale.

## Limitations

- **Untested on real multimodal data.** These APIs are verified against synthetic
  tensors only; no tabular encoder or multimodal cohort loader exists yet.
- **Early fusion needs vectors.** A gigapixel slide must be pooled before it can be
  concatenated at the input, which limits what early fusion can learn from pathology.
- **The Cox risk set still spans the mini-batch**, so the batch-size guidance from the
  unimodal pipeline carries over.
- **No multimodal trainer or cohort loader yet** — these are model-level APIs;
  matching patients across modalities comes with the tabular merge.
