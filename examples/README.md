# Examples

Each example is named `<dataset>_<modality>_<task>`, so what it does is readable from
the directory name alone. The task suffix is the important one: `survival` predicts a
right-censored time to event with a Cox partial likelihood, `classification` predicts
a binary outcome with cross-entropy.

| Example | Modality | Task | Endpoint |
| --- | --- | --- | --- |
| [`hancock_tabular_survival/`](hancock_tabular_survival/) | Structured tables | Time-to-event (Cox) | Overall survival |
| [`hancock_wsi_survival/`](hancock_wsi_survival/) | Whole-slide imaging | Time-to-event (Cox) | Overall / disease-specific survival |
| [`hancock_multimodal_survival/`](hancock_multimodal_survival/) | Tables + imaging | Time-to-event (Cox) | Overall survival |
| [`hancock_multimodal_classification/`](hancock_multimodal_classification/) | Tables + imaging | Binary classification | Recurrence, survival status |
| [`synthetic_survival.py`](synthetic_survival.py) | Synthetic | Time-to-event (Cox) | — (no data needed) |

[`hancock/`](hancock/) is not an example. It holds the shared HANCOCK archive access
every HANCOCK example imports, so the dataset layout is described once.

## Survival and classification are the same pipeline

The two multimodal examples are a matched pair: same fusion, same two modalities, same
splits. They differ only in the task handed to the trainer.

```python
CohortTrainer(embedders, task=SurvivalTask())  # time-to-event
CohortTrainer(embedders, task=ClassificationTask(pos_weight=3.0))  # binary
```

The task carries the head, the loss, the epoch metric, and whether a batch can be
learned from at all. Everything else — encoders, fusion, splitting, the training
loop — is shared. See [`kalecancer/pipeline/task.py`](../kalecancer/pipeline/task.py).

The whole-slide example uses the same `CohortTrainer` with a single bag modality,
so there is no whole-slide trainer either. Each example owns its own orchestration
— [`hancock_wsi_survival/runner.py`](hancock_wsi_survival/runner.py) is the fullest
one — because naming a dataset, an endpoint and an output layout is what an
experiment is, and none of it generalises to the next study.

## Running one

All examples run as modules from the repository root, so their imports resolve without
any `sys.path` handling:

```bash
python -m examples.synthetic_survival
python -m examples.hancock_wsi_survival.main --cfg examples/hancock_wsi_survival/configs/hancock_primary_tumour_quick.yaml
python -m examples.hancock_multimodal_survival.main --cfg examples/hancock_multimodal_survival/configs/multimodal.yaml
python -m examples.hancock_multimodal_classification.main --cfg examples/hancock_multimodal_classification/configs/quick.yaml
```

## Splitting

Every example uses HANCOCK's **published train/test assignment** by default, because
re-drawing the test set is what makes a number incomparable with everything else
reported on this cohort. Validation is always carved out of the training half, so the
published test set is never touched during model selection.

Cross-validation is a configuration change, for when a single held-out split is too
small to separate arms:

```bash
python -m examples.hancock_wsi_survival.main --cfg <cfg> DATASET.SPLIT_MODE cv DATASET.NUM_FOLDS 5
```

| `DATASET.SPLIT_MODE` | Test set | Use when |
| --- | --- | --- |
| `published` (default) | The dataset's own assignment | Reporting a comparable number |
| `cv` | Each of `NUM_FOLDS` folds in turn | The held-out split is too small |
| `random` | A fresh stratified draw | The cohort publishes no split |

`DATASET.SPLIT_FILE` picks which published assignment to apply, for a dataset offering
several. The tabular example uses its own YAML key, `split.mode`, for the same choice.

Anything fitted on data — scalers, one-hot encoders, the TabICL context, the IPCW
censoring distribution and the Breslow baseline hazard — is fitted on the training
rows alone, and rebuilt per fold when cross-validating.

## Data

Every HANCOCK example fetches what it needs over HTTP range requests from the published
archives (CC BY 4.0, no credentials), transferring only the patients selected by
`DATASET.PATIENTS`. Keep that number small when verifying a run: a cohort of 50–60
patients leaves fewer than 10 in the test split, which is enough to check the pipeline
executes but not to produce a meaningful metric. Report such runs as verification, not
as results.

`synthetic_survival.py` needs no data and no network.
