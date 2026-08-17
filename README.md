# cancer — Multimodal Cancer AI for PyKale

Multimodal machine learning for oncology: whole-slide pathology, radiology, and
clinical records, with time-to-event prediction.

**cancer** is the oncology domain package of the [PyKale](https://github.com/pykale/pykale)
ecosystem, installed as **`kalecancer`**. It follows PyKale's verb-oriented pipeline
(`loaddata` → `prepdata` → `model` → `evaluate` → `interpret`), adding a dedicated
**`survival`** stage. MONAI covers 3D imaging well but provides neither multimodal
fusion nor survival modelling; this package supplies both for oncology.

## Quickstart

No data or credentials needed — patches are fetched from the published
[HANCOCK dataset](https://hancock.research.fau.eu/) (CC BY 4.0):

```bash
pip install -e ".[survival]"
kalecancer wsi-survival --source hancock --patients 50 --preset quick
```

## Documentation

| If you want to | Read |
| --- | --- |
| Run a prediction without writing code | [Quickstart](docs/quickstart.md) |
| Use your own data, or change the experiment | [WSI survival pipeline](examples/wsi_survival/) |
| Combine modalities | [Multimodal fusion](docs/multimodal_fusion.md) |
| Contribute code | [Contributing](#contributing) and [AGENTS.md](AGENTS.md) |
| Understand the design decisions | [Architecture](docs/architecture.md) |

## What works today

| Capability | Status |
| --- | --- |
| WSI survival from precomputed patch features | Attention MIL + Cox head, with metrics and attention export |
| Survival modelling | Cox loss, Harrell and IPCW C-index, time-dependent AUC, Brier score |
| Multimodal fusion | Early, late, and hybrid strategies; `concat` / `poe` / `lowrank` |
| Data access | Local files, or streamed from the public HANCOCK archives |
| CT / MRI and tabular encoders | Planned |
| Preprocessing transforms, `auto` classes | Planned |

## Installation

Requires **Python ≥ 3.10**.

```bash
git clone https://github.com/pykale/cancer.git
cd cancer
pip install -e ".[survival]"
```

Heavy libraries are kept out of the core install:

| Extra | Packages | When to use |
| --- | --- | --- |
| `survival` | torchsurv | Time-to-event losses and metrics |
| `imaging` | monai, nibabel, pydicom, SimpleITK | DICOM / NIfTI workflows |
| `pathology` | openslide-python, tifffile | Reading whole-slide images |
| `interpret` | shap, captum | Model explanation |
| `dev` | pytest, ruff, black, isort, mypy | Development and CI |

## Package layout

```
kalecancer/
├── loaddata/    Patch features, clinical labels, cohort matching, splitting
├── prepdata/    Transforms — planned
├── model/       Encoders (attention MIL) and multimodal fusion
├── pipeline/    Trainers and end-to-end runners
├── survival/    Cox head, loss, metrics, baseline hazard
├── evaluate/    Metrics and prediction reports
├── interpret/   Attention export
├── auto/        High-level construction — planned
└── utils/       Seeding and artefact writing
```

`survival/` is deliberately cancer-agnostic: it must not import from other
`kalecancer` modules, so it can move into PyKale core. CI enforces this
(`tests/test_survival_boundary.py`).

## Status

Early development. A preliminary version is targeted for **31 August 2026**.

## Contributing

1. Fork the repository and create a feature branch.
2. `pip install -e ".[dev,survival]"`.
3. Run the same checks as CI:

   ```bash
   black --check . && isort --check-only . && ruff check . && pytest
   ```

4. Open a pull request against `main`.

Conventions and architectural constraints are documented in [AGENTS.md](AGENTS.md),
which applies to human and automated contributors alike.

## License

MIT — see [LICENSE](LICENSE). The HANCOCK dataset is licensed separately under CC BY 4.0.
