# HANCOCK outcome classification

Reproduces the experimental design of [Dörrich et al., *A multimodal dataset for
precision oncology in head and neck cancer*, Nat Commun 16:7163
(2025)](https://doi.org/10.1038/s41467-025-62386-6) with the two modalities this
repository holds, as an external check that the KaleCancer APIs work outside the
survival setting they were built for.

```
structured tables (clinical + pathological + blood)     UNI patch features
                    |                                          |
              MLPEmbedder                               AttentionMIL
                    |                                          |
                    +---------------- fusion ------------------+
                                        |
                              linear logit + BCE
                                        |
                                    ROC-AUC
```

## What is reproduced

| Paper | Here |
| --- | --- |
| Two endpoints: `survival_status`, `recurrence` | Same, with the same filters |
| Three published splits: in / out / Oropharynx | Same files, read directly |
| Early fusion of per-modality encodings | `MultimodalFusion`, stage and method configurable |
| Random Forest + SMOTE | `CohortTrainer` with a `ClassificationTask`, plus a RandomForest baseline |
| ROC-AUC, mean ± std over 5 repeats | Same, plus a bootstrap CI |
| Fig 3C most-attended patches | `top_patches.csv` per imaging arm |
| Fig 3F attention by modality | Leave-one-modality-out AUC |
| Fig 2B–D UMAP of patient vectors | `umap_embedding`, needs the `interpret` extra |

## Running it

```bash
# One cell, three epochs: checks the wiring on real data
python -m examples.hancock_multimodal_classification.main --cfg examples/hancock_multimodal_classification/configs/quick.yaml

# The full matrix: 2 endpoints x 3 splits x 3 modality settings x 5 repeats
python -m examples.hancock_multimodal_classification.main --cfg examples/hancock_multimodal_classification/configs/full.yaml

# Shard it, then collect
python -m examples.hancock_multimodal_classification.main --splits dataset_split_in.json
python -m examples.hancock_multimodal_classification.main --collect
```

Nothing needs to be downloaded by hand: the structured tables, the split files and
the patch features are fetched from the published archives on first use and cached.
Cells that already have a `summary.json` are skipped unless `--force` is given.

## What the two endpoints mean

Both are binary, and both are defined by *filtering* before labelling — a patient
whose outcome cannot be determined is dropped rather than called negative.

**`survival_status`** — living (0) against deceased (1), excluding the 89 patients
recorded as *deceased not tumor specific*. They did die, so calling them survivors
would be wrong, and the tumour did not kill them, so calling them positive would be
wrong too. 674 patients, 18.4 % positive.

**`recurrence`** — recurrence within three years. This follows the authors' **code**,
which is looser than their prose: the negative class admits any living,
recurrence-free patient regardless of follow-up length, not only those followed for
three years. That roughly doubles the negative class (667 patients rather than 487).
We match the code because that is what produced the published numbers, and
`tests/examples/test_hancock_multimodal_classification.py` pins the discrepancy so nobody
"corrects" it to the prose without noticing every recurrence figure move.

## What "tabular" means here

Not demographics. The paper's patient vector is built from the clinical **and**
pathological **and** blood tables, and this example does the same:

| Role | Columns | Encoding |
| --- | --- | --- |
| binary flags | 7 | most-frequent impute, one-hot |
| nominal | 7 | most-frequent impute, one-hot |
| discrete | 3 | mean impute, standardise |
| ordinal stage | 2 | integer rank, then as above |
| blood | 16 | mean impute, standardise |

That is 67 encoded columns against the paper's 104. The gap is their 40 ICD
bag-of-words columns and 4 TMA cell densities, which come from archives outside
`StructuredData.zip`. Expect our tabular AUC to sit below theirs for that reason
alone.

Missing categories become a level of their own rather than being imputed away:
whether a field was assessed is informative here — perinodal invasion is unrecorded
for the 365 patients who had no neck dissection — and the reference implementation
treats it the same way.

The encoded width is split-dependent, because the one-hot encoder is fitted on the
training rows and a category absent from them contributes no column. It is logged,
not asserted.

## Where our numbers legitimately differ

1. **Fewer features** — 67 against 104, as above.
2. **Different model** — an MLP into fused features and a linear logit, against their
   tuned RandomForest. The RandomForest baseline row is the like-for-like check.
3. **Class weighting, not SMOTE** — both counter imbalance; they are not equivalent.
4. **Different imaging pipeline** — `AttentionMIL` over the published UNI encodings
   with a patch cap, against CLAM over ~3 % of patches at 256x256.
5. **± means something different** — the split is fixed, so our repeats vary only
   initialisation and the inner validation split. Their spread also carries
   RandomForest and SMOTE randomness. The bootstrap CI is the comparable interval,
   and on the Oropharynx split it is what should be read first.

The direction is what should reproduce, not the value: tabular above imaging, and
fusion at least as good as the better single modality.

## Outputs

```
outputs/hancock_multimodal_classification/
├── results.csv                 one row per cell, with the paper's AUC alongside
├── results.json
└── <split>/<target>/
    ├── umap.csv                the cohort in 2D, with the split marked   (Fig 2B-D)
    └── <arm>/
        ├── summary.json        mean, std, bootstrap CI, ablation
        ├── roc_mean.csv        the mean ROC curve and its band           (Fig 2E)
        ├── top_patches.csv     imaging arms only                         (Fig 3C)
        └── repeat_<i>/metrics.json
```

`umap.csv` is written once per split and endpoint rather than per arm, because it
describes the patient vectors rather than any one model. The projection is fitted on
the whole cohort, as the paper's is, so read it as a map of the data and of where the
published split falls -- not as evidence that the classes separate.

## Figures

```bash
python -m examples.hancock_multimodal_classification.figures
```

Reads back what `main.py` wrote and renders it to `outputs/.../figures/`. Nothing is
recomputed, so the figures cannot disagree with the reported numbers, and rerunning
is instant.

| File | Reproduces |
| --- | --- |
| `fig2e_roc_curves.png` | Fig 2E — mean ROC per split and endpoint, shaded +/-1 SD |
| `fig2bd_umap.png` | Fig 2B-D — the cohort in 2D, coloured by outcome and by split |
| `fig3e_modality_comparison.png` | Fig 3E — per-repeat AUC by modality, with the paper's values marked |
| `fig3f_modality_ablation.png` | Fig 3F — what each modality was worth |
| `fig3c_attention_maps.png` | Fig 3C — where the imaging model looked |

Two honest limits. The attention figure draws only the `OUTPUT.TOP_K` patches that
were exported, and draws no tissue, because the source slides are outside this
pipeline; their Fig 3C shows the tiles themselves. And the paper reports one WSI-only
AUC for the whole cohort rather than one per split, so that value is drawn as a
reference line rather than a per-split marker.

## Reading the results

`train_auc_mean` near 0.5 while the loss falls means the labels never reached the
objective; a test AUC well below 0.5 means the label polarity is inverted. A tabular
AUC near 1.0 means leakage — the first thing to check is that the preprocessor was
fitted on training identifiers only, which `tabular.encode` asserts.
