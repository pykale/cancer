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
python -m examples.hancock_wsi_survival.main \
    --cfg examples/hancock_wsi_survival/configs/hancock_primary_tumour_quick.yaml \
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

## Running an experiment

Experiments are the examples; there is no library command, because a run names a
dataset, an endpoint and an output layout, none of which generalise. Each example is
driven by a YAML config:

```bash
python -m examples.hancock_wsi_survival.main     --cfg examples/hancock_wsi_survival/configs/hancock_primary_tumour_quick.yaml
```

To read data already on disk instead of fetching it, use the local config:

```bash
python -m examples.hancock_wsi_survival.main     --cfg examples/hancock_wsi_survival/configs/local_primary_tumour.yaml     DATASET.FEATURE_ROOT /path/to/WSI_PrimaryTumor     DATASET.CLINICAL_PATH /path/to/clinical_data.json
```

Any setting can be overridden as trailing `KEY VALUE` pairs:

```bash
python -m examples.hancock_wsi_survival.main --cfg my_run.yaml MODEL.DROPOUT 0.1 SOLVER.BASE_LR 0.0003
```

### Configurations

| Config | Run |
| --- | --- |
| `hancock_primary_tumour_quick.yaml` | 3 epochs on subsampled patches, to check a setup |
| `hancock_primary_tumour.yaml` | The full published split |
| `hancock_primary_tumour_cv.yaml` | 5-fold patient-level cross-validation |
| `hancock_primary_tumour_dss.yaml` | Disease-specific survival instead of overall survival |
| `local_primary_tumour.yaml` | Files already on disk |

The test set comes from HANCOCK's published assignment by default;
`DATASET.SPLIT_MODE cv` cross-validates instead. See
[examples/README.md](../examples/README.md).

## Python

```python
from kalecancer.config import get_cfg_defaults
from kalecancer.loaddata.clinical_access import endpoint_from_config
from examples.hancock_wsi_survival.runner import run

cfg = get_cfg_defaults()
cfg.DATASET.FEATURE_ROOT = "/path/to/WSI_PrimaryTumor"
cfg.DATASET.CLINICAL_PATH = "/path/to/clinical_data.json"
cfg.SOLVER.MAX_EPOCHS = 30
# Local files carry no published assignment, so draw a split rather than apply one.
cfg.DATASET.SPLIT_MODE = "random"
cfg.freeze()

metrics = run(cfg, endpoint=endpoint_from_config(cfg))
print(metrics["test"]["c_index"])
```

To fetch instead of reading local files, set `cfg.DATASET.SOURCE` and pass the
example's fetcher, which is what `examples/hancock_wsi_survival/main.py` does:

```python
from examples.hancock import fetch_for, split_for

metrics = run(
    cfg,
    endpoint=endpoint_from_config(cfg),
    fetch=lambda: fetch_for(cfg),
    splits=lambda: split_for(cfg),  # the published assignment
)
```

Or build the model directly, without the runner. There is one trainer, and what it
predicts is the task you hand it:

```python
from kalecancer.model.embed import AttentionMIL, BagEncoder, MLPEmbedder
from kalecancer.pipeline import ClassificationTask, CohortTrainer, SurvivalTask

embedders = {
    "wsi": BagEncoder(AttentionMIL(input_dim=1024, hidden_dim=256)),
    "clinical": MLPEmbedder(in_dim=64, out_dim=256),
}
survival = CohortTrainer(embedders, task=SurvivalTask())
binary = CohortTrainer(embedders, task=ClassificationTask(pos_weight=3.0))
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

> Run `examples/hancock_wsi_survival/main.py` with the quick config, overriding
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

- [WSI survival pipeline](../examples/hancock_wsi_survival/) — input formats, full configuration, interpretation
- [Multimodal fusion](multimodal_fusion.md) — combining modalities
