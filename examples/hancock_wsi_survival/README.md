# WSI survival pipeline

Time-to-event prediction from **precomputed whole-slide patch features**, using
attention multiple-instance learning with a Cox proportional-hazards head.

```
Precomputed patch features (HDF5)
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

The pipeline begins *after* patch encoding: slides are already tiled and passed
through a pathology foundation model (UNI via CLAM for HANCOCK). Tiling, stain
normalisation and foundation-model inference are out of scope.

Primary-tumour pathology only. Lymph-node slides are excluded by default: they
represent a different tumour microenvironment, and a node may show metastasis without
representing the primary tumour.

To run this, see the [quickstart](../../docs/quickstart.md). This page is the
reference for inputs, configuration and outputs.

## Input requirements

### Patch features

Any directory tree of HDF5 files; files are discovered recursively, so nesting by
subsite or batch is fine.

```
<FEATURE_ROOT>/
├── WSI_PrimaryTumor_Larynx/h5_files/PrimaryTumor_HE_002.h5
├── WSI_PrimaryTumor_OralCavity/h5_files/PrimaryTumor_HE_004.h5
└── ...
```

Each file must contain two row-aligned datasets:

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `features` | `(num_patches, feature_dim)` | Patch embeddings, 1024-d for UNI |
| `coords` | `(num_patches, 2)` | Patch `(x, y)` coordinates in the slide |

A file is rejected — and listed in `cohort_summary.json` — when a key is missing, the
feature array is empty, the arrays disagree in length, or the feature dimension does
not match `MODEL.INPUT_DIM`. Nothing is dropped silently.

### Patient identifiers

The patient id is read from the filename stem, keeping its original zero padding so it
matches the clinical records.

| Filename | Patient | Slide |
| --- | --- | --- |
| `PrimaryTumor_HE_036.h5` | `036` | `PrimaryTumor_HE_036` |
| `PrimaryTumor_HE_036_a.h5` | `036` | `PrimaryTumor_HE_036_a` |

A trailing letter marks an additional slide for the same patient. All of a patient's
slides are pooled into one bag, so the model produces exactly one risk score per
survival label, and each patch keeps a `slide_index` so attention stays traceable to
its source slide.

For a different naming scheme, pass a `slide_pattern` with a `patient_id` named group
to `build_cohort`.

### Clinical records

A JSON file holding a list of patient objects. For overall survival:

| Field | Use |
| --- | --- |
| `patient_id` | Matching key, read as a string |
| `days_to_last_information` | Time to event or censoring |
| `survival_status` | `"deceased"` → event, `"living"` → censored |

Internally the contract is always **`event = 1` observed, `0` censored**.

Endpoints (`SURVIVAL.ENDPOINT`):

- **`OS`** — overall survival, death from any cause.
- **`DSS`** — disease-specific survival, from `survival_status_with_cause`. Records
  reading only `"deceased"` carry no cause and are excluded rather than assumed
  censored; they appear under `clinical_exclusions`.

Records with missing, zero or negative time, or missing status, are excluded with a
recorded reason. Duplicate patient ids raise an error.

## Patient matching

Matching is deterministic and happens at patient level. `cohort_summary.json` reports
both sides of every mismatch:

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

## Configuration

Every setting has a default in `kalecancer/config.py`; `--print-config` shows the
resolved values.

| Section | Settings |
| --- | --- |
| `DATASET` | `SOURCE`, `FEATURE_ROOT`, `CLINICAL_PATH`, `REGION`, `PATIENTS`, `CACHE_DIR`, `MAX_PATCHES`, `NUM_WORKERS`, `TRAIN_RATIO` / `VAL_RATIO` / `TEST_RATIO`, `NUM_FOLDS`, `VALIDATE_FEATURES` |
| `MODEL` | `INPUT_DIM`, `HIDDEN_DIM`, `ATTENTION_DIM`, `DROPOUT`, `GATED` |
| `SOLVER` | `SEED`, `BASE_LR`, `WEIGHT_DECAY`, `MAX_EPOCHS`, `BATCH_SIZE`, `EARLY_STOP`, `OPTIMIZER`, `DEVICES` |
| `SURVIVAL` | `ENDPOINT`, `TIES`, `EVAL_TIMES` |
| `FUSION` | Multimodal only, see [fusion](../../docs/multimodal_fusion.md) |
| `OUTPUT` | `OUT_DIR`, `TOP_K` |

Ready-made configurations in [`configs/`](configs/). The `hancock_*` files fetch the
published archives, so they need no local copy of the data; `local_primary_tumour.yaml`
reads files already on disk instead.

| File | Source | Run |
| --- | --- | --- |
| `hancock_primary_tumour.yaml` | archive | Overall survival, single split |
| `hancock_primary_tumour_cv.yaml` | archive | 5-fold patient-level cross-validation |
| `hancock_primary_tumour_dss.yaml` | archive | Disease-specific survival |
| `hancock_primary_tumour_quick.yaml` | archive | Heavily subsampled smoke run, 50 patients |
| `local_primary_tumour.yaml` | local files | Overall survival from your own copy |

Setting `DATASET.NUM_FOLDS` above zero runs cross-validation, writing one
subdirectory per fold plus an aggregate mean and standard deviation.

## Evaluation

| Metric | Notes |
| --- | --- |
| Harrell's C-index | Primary metric; ranking of patients by risk |
| Uno's (IPCW) C-index | Reweighted for censoring |
| Time-dependent AUC | At `SURVIVAL.EVAL_TIMES`, by default 1, 3 and 5 years |
| Brier score and its integral | Calibration over follow-up |

Training loss (Cox partial likelihood), the validation metric (C-index, which drives
early stopping and checkpoint selection) and the final test metrics are reported
separately.

The censoring distribution and the Breslow baseline hazard are estimated from the
**training** split alone, then applied to the split being scored, so no test
information enters its own evaluation. Horizons beyond the observed follow-up are
dropped rather than extrapolated, and a split with no events reports its metrics as
undefined rather than failing.

## Attention interpretation

`attention/<patient_id>.csv` gives one row per patch, pairing its attention weight
with the coordinate and slide it came from:

```python
import pandas as pd

attention = pd.read_csv("outputs/wsi_survival/attention/036.csv")
hotspots = attention.nlargest(20, "attention")[["slide_id", "x", "y", "attention"]]
```

Attention is exported from full bags, so it covers whole slides. Because it is aligned
index-for-index with `coords`, these files are sufficient to render a heatmap over the
original slide.

Attention indicates regions associated with higher predicted risk. It is not a
calibrated probability and not a histopathological finding. No heatmap image is
produced, as raw whole-slide images are not inputs to this pipeline.

## Model behaviour

The Cox head predicts relative risk. Absolute survival probabilities come from a
Breslow baseline fitted on the training split and reflect that split's case mix.

The Cox partial likelihood is computed within each mini-batch. Batches containing no
events carry no gradient and are skipped, so `SOLVER.BATCH_SIZE` must be large enough
to contain events; at the HANCOCK event rate of about 27 %, 16 is the practical
minimum. Tied event times use Efron's approximation.

`DATASET.MAX_PATCHES` applies to training only, bounding memory for large slides.
Evaluation and interpretation always use the complete bag.

Survival times carry the unit of the source field, days for HANCOCK, and
`SURVIVAL.EVAL_TIMES` uses the same unit. Patients without usable primary-tumour
features are excluded rather than imputed: 62 of 763 clinical patients on HANCOCK.
