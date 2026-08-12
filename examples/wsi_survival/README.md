# WSI primary-tumour survival prediction

Time-to-event survival prediction from **precomputed whole-slide-image patch
features**, using attention multiple-instance learning with a Cox proportional-hazards
head.

The pipeline starts *after* patch encoding: slides have already been tiled and passed
through a pathology foundation model (UNI via CLAM for the HANCOCK cohort), so tiling,
stain normalisation and foundation-model inference are out of scope here.

```
Precomputed UNI patch features (HDF5)
          ↓
     Attention MIL
          ↓
patient-level representation
          ↓
 Cox proportional-hazards head
          ↓
      survival risk
          ↓
evaluation + attention interpretation
```

This is a **unimodal** pipeline: primary-tumour pathology only. Lymph-node slides are
deliberately excluded — they represent a different tumour microenvironment, and a
node may show metastasis without representing the primary tumour.

## Input data

### Patch features

Any directory tree of HDF5 files; nesting by subsite or batch is fine, and files are
discovered recursively:

```
<FEATURE_ROOT>/
├── WSI_PrimaryTumor_Larynx/h5_files/PrimaryTumor_HE_002.h5
├── WSI_PrimaryTumor_OralCavity/h5_files/PrimaryTumor_HE_004.h5
└── ...
```

Each file must contain two row-aligned datasets:

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `features` | `(num_patches, feature_dim)` | Patch embeddings (1024-d for UNI) |
| `coords` | `(num_patches, 2)` | Patch `(x, y)` coordinates in the slide |

Files are rejected — and listed in `cohort_summary.json` — when a key is missing, the
feature array is empty, the two arrays disagree in length, or the feature dimension
differs from `MODEL.INPUT_DIM`. Nothing is dropped silently.

### Patient identifiers

The patient id is parsed from the filename stem: everything after the last underscore
group of digits, keeping the original zero padding so it matches the clinical records
exactly.

| Filename | Patient | Slide |
| --- | --- | --- |
| `PrimaryTumor_HE_036.h5` | `036` | `PrimaryTumor_HE_036` |
| `PrimaryTumor_HE_036_a.h5` | `036` | `PrimaryTumor_HE_036_a` |

A trailing letter marks an **additional slide for the same patient**. All of a
patient's slides are pooled into one bag, so the model produces exactly one risk score
per survival label. Each patch keeps a `slide_index`, so attention remains traceable to
its source slide.

Override the pattern by passing a different `slide_pattern` to `build_cohort` if your
naming differs; it only needs a `patient_id` named group.

### Clinical data

A JSON file holding a list of patient objects. For the default overall-survival
endpoint:

| Field | Use |
| --- | --- |
| `patient_id` | Matching key, read as a string |
| `days_to_last_information` | Time to event or censoring, in days |
| `survival_status` | `"deceased"` → event, `"living"` → censored |

Internally the label contract is always **`event = 1` observed, `0` censored**.

Two endpoints are provided (`SURVIVAL.ENDPOINT`):

- **`OS`** (default) — overall survival, death from any cause.
- **`DSS`** — disease-specific survival, from `survival_status_with_cause`. Records
  reading only `"deceased"` carry no cause and are **excluded**, not assumed censored;
  they appear under `clinical_exclusions` in the cohort summary.

Records with missing, zero or negative time, or missing status, are excluded with a
recorded reason. Duplicate patient ids raise an error.

## Patient matching

Matching is deterministic and happens entirely at patient level. `cohort_summary.json`
reports both sides of every mismatch:

```json
{
  "num_clinical_patients": 763,
  "num_wsi_patients": 701,
  "num_matched_patients": 701,
  "num_events": 192,
  "num_censored": 509,
  "patients_without_wsi": ["005", "..."],
  "unmatched_wsi_patients": [],
  "invalid_feature_files": [],
  "patients_with_multiple_slides": {"036": 2, "239": 2}
}
```

## Running

```bash
python examples/wsi_survival/main.py
python examples/wsi_survival/main.py --cfg examples/wsi_survival/configs/hancock_primary_tumour.yaml
```

### Provided configurations

| Config | Run |
| --- | --- |
| `hancock_primary_tumour.yaml` | Overall survival, single seeded train/val/test split |
| `hancock_primary_tumour_cv.yaml` | Overall survival, 5-fold patient-level cross-validation |
| `hancock_primary_tumour_dss.yaml` | Disease-specific survival endpoint |
| `hancock_primary_tumour_quick.yaml` | Heavily subsampled smoke run for checking wiring |

Override any setting as trailing `KEY VALUE` pairs:

```bash
python examples/wsi_survival/main.py \
    DATASET.FEATURE_ROOT "E:/WSI_UNI_encodings/WSI_PrimaryTumor" \
    DATASET.CLINICAL_PATH "E:/StructuredData/StructuredData/clinical_data.json" \
    SOLVER.MAX_EPOCHS 30
```

To change dataset paths permanently, edit `configs/hancock_primary_tumour.yaml` — no
path is hard-coded in the library.

### Configuration

| Section | Key settings |
| --- | --- |
| `DATASET` | `FEATURE_ROOT`, `CLINICAL_PATH`, `MAX_PATCHES`, `NUM_WORKERS`, `TRAIN_RATIO` / `VAL_RATIO` / `TEST_RATIO`, `NUM_FOLDS`, `VALIDATE_FEATURES` |
| `MODEL` | `INPUT_DIM`, `HIDDEN_DIM`, `ATTENTION_DIM`, `DROPOUT`, `GATED` |
| `SOLVER` | `SEED`, `BASE_LR`, `WEIGHT_DECAY`, `MAX_EPOCHS`, `BATCH_SIZE`, `EARLY_STOP`, `OPTIMIZER` |
| `SURVIVAL` | `ENDPOINT`, `TIES`, `EVAL_TIMES` |
| `OUTPUT` | `OUT_DIR`, `TOP_K` |

Setting `DATASET.NUM_FOLDS` to a non-zero value runs patient-level stratified
cross-validation instead of a single split, writing one subdirectory per fold.

## Evaluation

| Metric | Notes |
| --- | --- |
| Harrell's C-index | Primary metric; ranking of patients by risk |
| Uno's (IPCW) C-index | Reweighted for censoring |
| Time-dependent AUC | At `SURVIVAL.EVAL_TIMES`, by default 1, 3 and 5 years |
| Brier score + integrated Brier score | Calibration over follow-up |

Training loss (Cox partial likelihood), the validation metric (C-index, which drives
early stopping and checkpoint selection) and the final test metrics are reported
separately.

The censoring distribution and the Breslow baseline hazard are estimated from the
**training** split only, then applied to the split being scored, so no test information
leaks into its own evaluation. Horizons beyond the observed follow-up are dropped
rather than extrapolated.

## Outputs

Written under `OUTPUT.OUT_DIR`:

```
config.yaml              effective configuration for the run
cohort_summary.json      matching provenance and every exclusion
history/                 per-epoch training and validation logs
checkpoints/best.ckpt    best model by validation C-index
predictions.csv          patient_id, split, risk_score, duration, event
metrics.json             metrics per split
attention/<patient>.csv  patient_id, slide_id, x, y, attention
attention/top_patches.csv the most attended patches per patient
```

Every prediction is traceable to its patient identifier.

## Attention interpretation

`attention/<patient_id>.csv` gives one row per patch, pairing its attention weight with
the coordinate and slide it came from. Because attention is aligned index-for-index
with `coords`, these files are enough to render a heatmap over the original slide:

```python
import pandas as pd

attention = pd.read_csv("outputs/wsi_survival/attention/036.csv")
hotspots = attention.nlargest(20, "attention")[["slide_id", "x", "y", "attention"]]
```

Attention is exported from full bags (no patch subsampling), so it covers whole slides.

No heatmap image is produced: raw whole-slide images are not part of this pipeline, and
rendering one without them would mean inventing the underlying tissue.

## Limitations and assumptions

- **Relative risk, not absolute.** The Cox head predicts a ranking. Survival
  probabilities come from a Breslow baseline fitted on the training split and inherit
  its case mix.
- **Attention marks risk, not diagnosis.** High attention indicates regions associated
  with *higher predicted risk*, not a calibrated probability of death, and is not a
  histopathological finding.
- **Batch-level risk set.** The Cox partial likelihood is computed within each
  mini-batch. Batches containing no events carry no gradient and are skipped, so
  `SOLVER.BATCH_SIZE` should stay large enough to contain events — with a ~27 % event
  rate, 16 is a reasonable floor.
- **Tied event times** use Efron's approximation, which matters at day-resolution
  follow-up.
- **`MAX_PATCHES` applies to training only**, bounding memory for large slides;
  evaluation and interpretation always use the full bag.
- **Survival time units** follow the source field (days for HANCOCK), so
  `SURVIVAL.EVAL_TIMES` must use the same unit.
- Patients without usable primary-tumour features are excluded, not imputed. On
  HANCOCK this is 62 of 763 clinical patients.
