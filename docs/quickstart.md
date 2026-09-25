# Quickstart

Install with `uv sync --all-extras`, then run an example from the repository root:

```bash
uv run python examples/<dir>/<script>.py
```

Each example is a standalone script. Start with the synthetic one, then read a HANCOCK script to see the same
`Pipeline` on real data. For how the pieces fit together, see [architecture.md](architecture.md) and
[multimodal_fusion.md](multimodal_fusion.md).

## Synthetic

[examples/synthetic/synthetic_survival.py](../examples/synthetic/synthetic_survival.py) is the smallest end-to-end
run. It generates a cohort of a clinical table, patch features and censored survival times, and trains an
intermediate-fusion Cox model. It runs on CPU in under a minute and downloads nothing. The notebook
[synthetic_survival.ipynb](../examples/synthetic/synthetic_survival.ipynb) is the same example.

## HANCOCK

These need a local copy of the HANCOCK data in `data/hancock/`, which git ignores. Full cohorts should be trained on a
GPU machine; the scripts fall back to CPU when no GPU is found. Each one takes a published split (`in`, `out` or
`Oropharynx`) from `SPLIT_NAME` at the top of the file.

All four use a clinical table (encoded by TabICL) and primary-tumour patch features (attention MIL), and require
patients to have both.

| Script | Task | Fusion | What it shows |
| --- | --- | --- | --- |
| [survival_intermediate.py](../examples/hancock/survival_intermediate.py) | Survival (Cox) | Intermediate | The main example: build the model in Python, train on a split, report Harrell's C and Uno's C. A [notebook](../examples/hancock/survival_intermediate.ipynb) has the output already captured. |
| [from_config.py](../examples/hancock/from_config.py) | Survival (Cox) | Intermediate | The same pipeline loaded from [configs/intermediate_cox.yaml](../examples/hancock/configs/intermediate_cox.yaml) with `load_pipeline`, instead of built in Python. |
| [cross_validation.py](../examples/hancock/cross_validation.py) | Survival (Cox) | Intermediate | Five-fold `cross_validate` on the full cohort, refitting everything learned from data in each fold, with per-fold Harrell's C. |
| [classification_late.py](../examples/hancock/classification_late.py) | Classification (deceased vs living) | Late | A clinical branch and a WSI branch trained jointly, with logits averaged. Reports AUROC and balanced accuracy for the fused model and for each branch. |

Choosing between them:

- **Learning the API:** the synthetic example, then `survival_intermediate.py`.
- **Configuring without code:** `from_config.py`.
- **A more reliable estimate than one split:** `cross_validation.py`.
- **A different endpoint or fusion:** `classification_late.py`. It changes the target, head and fusion, and the
  `Pipeline` stays the same.
