# cancer — Multimodal Cancer AI for PyKale

Accessible multimodal machine learning for oncology — combining radiology volumes, whole-slide pathology, and clinical records, with time-to-event prediction.

## What this is

**cancer** is the oncology domain package of the [PyKale](https://github.com/pykale/pykale) ecosystem. It is installable on PyPI and in development environments as **`kalecancer`**.

The package follows PyKale’s verb-oriented pipeline API (`loaddata` → `prepdata` → `model` → `evaluate` → `interpret`), extended with cancer-specific data types and a dedicated **`survival`** stage for time-to-event modelling.

## Why it exists

Clinical collaborators — medical physicists, pathologists, and research software engineers — often already use **MONAI** for medical imaging workflows. MONAI excels at 3D radiology preprocessing and model zoo access, but it does **not** provide:

- multimodal **fusion** across imaging and tabular clinical data, or
- built-in **time-to-event / survival** modelling.

**PyKale** is designed for multimodal learning and transfer learning across modalities. **kalecancer** bridges that gap for oncology: reusing familiar imaging tooling where appropriate, while adding clinical tabular integration and survival analysis as first-class concerns.

## Supported modalities

| Modality | Formats | Description |
| --- | --- | --- |
| CT / MRI | DICOM, NIfTI | 3D radiology volumes *(planned loaders)* |
| Whole-slide pathology | `.svs`, TIFF | Gigapixel 2D slide images *(planned loaders)* |
| Clinical tabular | CSV | Patient features, outcomes, missing values *(planned loaders)* |

> **Note:** Modality support is being built out module by module. See [Status](#status) below.

## Supported tasks

| Task | Examples | Status |
| --- | --- | --- |
| **Classification** | Tumour vs benign, molecular subtype | Planned |
| **Regression** | Continuous biomarkers, tumour burden | Planned |
| **Time-to-event / survival** | Overall survival (OS), disease-free survival (DFS) | **Primary focus** — planned |

Survival analysis is the main research driver for this package: jointly modelling imaging, pathology, and clinical variables to predict when an event (e.g. progression or death) occurs.

## Package layout

```
kalecancer/
├── auto/          High-level selection and construction (AutoCancer* classes) — planned
├── loaddata/      Datasets and records: DICOM/CT, WSI, tabular clinical — planned
├── prepdata/      Reusable transforms: CT windowing, WSI tiling, stain normalisation — planned
├── model/         Modality encoders, fusion, and task heads — planned
├── survival/      Cox head, losses, and metrics — planned (cancer-agnostic; destined for PyKale core)
├── evaluate/      Performance metrics — planned
├── interpret/     SHAP, Grad-CAM, modality contribution — planned
└── utils/         Shared helpers — planned
```

The **`survival`** submodule is intentionally **cancer-agnostic**: it must not import from other `kalecancer` modules, so it can later move into PyKale core. This boundary is enforced in CI (`tests/test_survival_boundary.py`).

## Installation

Requires **Python ≥ 3.10**. Install PyTorch for your platform first, then:

```bash
git clone https://github.com/pykale/cancer.git
cd cancer
pip install -e .
```

### Optional extras

Install only what you need — heavy imaging and pathology libraries are kept out of the core install.

| Extra | Packages | When to use |
| --- | --- | --- |
| `imaging` | monai, nibabel, pydicom, SimpleITK | DICOM / NIfTI radiology workflows |
| `pathology` | openslide-python, tifffile | Whole-slide image reading |
| `interpret` | shap, captum | Model explanation and attribution |
| `dev` | pytest, ruff, black, isort, mypy, … | Development and CI |

```bash
pip install -e ".[dev]"                        # development tools
pip install -e ".[imaging,pathology]"          # radiology + pathology
pip install -e ".[imaging,pathology,interpret,dev]"  # everything
```

## Planned API (not yet implemented)

The following illustrates the intended high-level workflow. **None of this is available yet** — it documents the design direction only.

```python
# Planned API — subject to change
import kalecancer as kc

pipeline = kc.auto.AutoCancerSurvival.from_config("configs/os_lung.yaml")
pipeline.fit()
metrics = pipeline.evaluate()
pipeline.interpret(modality="clinical")
```

## Status

**Early development.** The package skeleton, CI, and architectural boundaries are in place; loaders, transforms, models, and metrics are not yet implemented.

A **preliminary version** is targeted for **31 August 2026**.

## Contributing

Contributions are welcome from PyKale maintainers and clinical collaborators.

1. Fork the repository and create a feature branch.
2. Install with `pip install -e ".[dev]"`.
3. Run the linters and tests locally (same as CI):

   ```bash
   black --check .
   isort --check-only .
   ruff check .
   pytest --cov=kalecancer
   ```

4. Open a pull request against `main`.

For architectural questions — especially around the `survival` boundary — see `tests/test_survival_boundary.py` and the module docstrings under `kalecancer/survival/`.

## License

MIT — see [LICENSE](LICENSE).
