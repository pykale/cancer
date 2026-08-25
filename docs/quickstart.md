# Quickstart

Pick the route that matches how you work. All three run the same pipeline with the
same settings.

| Route | For |
| --- | --- |
| [Command line](#command-line) | Running predictions without writing code |
| [Python](#python) | Notebooks, custom experiments, reusing single stages |
| [Coding agent](#coding-agent) | Delegating setup and runs to Copilot, Claude Code, Cursor, Codex, Jules |

## Install

```bash
pip install -e .
```

## Get the data

Either use your own files, or fetch the published
[HANCOCK dataset](https://hancock.research.fau.eu/) (CC BY 4.0), which needs no
account.

Downloading belongs to the experiment that knows the dataset's layout, so the fetcher
lives with the example rather than in the library. Running the example fetches what
the configuration asks for:

```bash
python examples/wsi_survival/main.py \
    --cfg examples/wsi_survival/configs/hancock_primary_tumour_quick.yaml \
    DATASET.SOURCE hancock DATASET.PATIENTS 50
```

The feature archive is 9.65 GB, but supports HTTP range requests, so only the selected
patients are transferred — about 640 MB for 60 patients. Files are cached, so repeat
runs fetch nothing.

| Setting | Purpose |
| --- | --- |
| `DATASET.PATIENTS N` | Patients to fetch; `0` fetches all 709 |
| `DATASET.REGION primary\|lymph_node` | Anatomical region |
| `DATASET.CACHE_DIR DIR` | Cache location, default `~/.cache/kalecancer` |

Patients are selected in sorted identifier order, so a given `DATASET.PATIENTS` always
gives the same cohort, and a patient's slides are always kept together.

> A 50-patient cohort leaves fewer than 10 patients in the test split. That is enough
> to confirm the pipeline runs, but not to produce a meaningful metric.

## Command line

`kalecancer` runs on data already on disk:

```bash
kalecancer wsi-survival \
    --features /path/to/WSI_PrimaryTumor \
    --clinical /path/to/clinical_data.json \
    --preset quick
```

Point `--features` at the cache directory above to reuse an already-fetched cohort.

### Presets

| Preset | Run |
| --- | --- |
| `quick` | 3 epochs on subsampled patches, to check a setup |
| `default` | Single seeded train/validation/test split |
| `cv` | 5-fold patient-level cross-validation |
| `dss` | Disease-specific survival instead of overall survival |

### Common options

| Option | Purpose |
| --- | --- |
| `--out DIR` | Where results are written |
| `--epochs N` | Training epochs |
| `--batch-size N` | Patients per batch; 16 or more is recommended |
| `--folds N` | Cross-validation folds; `0` uses a single split |
| `--endpoint OS\|DSS` | Survival endpoint |
| `--print-config` | Print the resolved settings and exit |

`--print-config` outputs a complete YAML file. Save it, edit it, and pass it back with
`--cfg` for a fully reproducible run. Anything can also be overridden directly:

```bash
kalecancer wsi-survival --cfg my_run.yaml MODEL.DROPOUT 0.1 SOLVER.BASE_LR 0.0003
```

## Python

```python
from kalecancer.config import get_cfg_defaults
from kalecancer.loaddata.clinical_access import endpoint_from_config
from kalecancer.pipeline.wsi_survival_runner import run

cfg = get_cfg_defaults()
cfg.DATASET.FEATURE_ROOT = "/path/to/WSI_PrimaryTumor"
cfg.DATASET.CLINICAL_PATH = "/path/to/clinical_data.json"
cfg.SOLVER.MAX_EPOCHS = 30
cfg.freeze()

metrics = run(cfg, endpoint=endpoint_from_config(cfg))
print(metrics["test"]["c_index"])
```

To fetch instead of reading local files, set `cfg.DATASET.SOURCE` and pass the
example's fetcher, which is what `examples/wsi_survival/main.py` does:

```python
from hancock import fetch_for  # examples/wsi_survival/hancock.py

metrics = run(cfg, endpoint=endpoint_from_config(cfg), fetch=lambda: fetch_for(cfg))
```

Individual stages work on their own:

```python
from kalecancer.evaluate.cohort_report import cohort_summary, log_cohort_summary
from kalecancer.loaddata import build_cohort, train_val_test_split
from kalecancer.loaddata.clinical_access import endpoint_from_config

cohort = build_cohort(feature_root, clinical_path, endpoint=endpoint_from_config(cfg))
log_cohort_summary(cohort_summary(cohort))  # matched patients and every exclusion

split = train_val_test_split(cohort, group_key="patient_id", stratify_keys=["event"], seed=2026)
```

## Coding agent

[`AGENTS.md`](../AGENTS.md) is read by Copilot, Cursor, Codex, Claude Code, Gemini CLI
and Jules, and `.devcontainer/` makes a GitHub Codespace start ready to run. Since the
data comes from a public archive, an agent needs no local files and no credentials.

**1. Set up**

> Set up this repository following AGENTS.md. Create the virtual environment, install
> the package with the dev extra, and confirm by running the test suite. Report
> anything that fails.

**2. Run it**

> Run `examples/wsi_survival/main.py` with the quick config, overriding
> `DATASET.SOURCE hancock DATASET.PATIENTS 50` so it fetches its own data. Report the
> cohort summary and the test C-index.

**3. Check the cohort**

> Read `cohort_summary.json` from the output directory. Tell me how many patients
> matched, how many were excluded and why, and whether any feature files were invalid.

**4. Run the experiment**

> Run again with `DATASET.NUM_FOLDS 5` and `SOLVER.MAX_EPOCHS 30`. Report the
> cross-validated C-index and its standard deviation from `metrics.json`.

**5. Interpret**

> From the attention export, list the 10 highest-attention patches for the three
> highest-risk patients in `predictions.csv`, with slide IDs and coordinates.

To make a change rather than a run:

> Add <component> to the appropriate `kalecancer/` subpackage following AGENTS.md. Add
> tests in the mirrored `tests/` path using synthetic data, then run the format, lint
> and test commands and fix anything that fails.

A Codespace has no GPU. Use it for development, tests and small demonstration runs;
train full cohorts on a GPU machine.

## Results

| File | Contents |
| --- | --- |
| `cohort_summary.json` | Patients matched, and every exclusion with its reason |
| `metrics.json` | Metrics per split |
| `predictions.csv` | Risk score per patient |
| `attention/<patient>.csv` | Attention weight per patch, with coordinates |
| `checkpoints/best.ckpt` | Best model by validation C-index |
| `config.yaml` | Exact settings used |

## Next

- [WSI survival pipeline](../examples/wsi_survival/) — input formats, full configuration, interpretation
- [Multimodal fusion](multimodal_fusion.md) — combining modalities
