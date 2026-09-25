# KaleCancer

> *Multimodal machine learning for oncology: whole-slide pathology, radiology, and clinical records, with time-to-event prediction.*

-----------------------------------------

<!-- Keep badges to just ONE line, i.e. only the most important badges! -->
[![Built on PyKale](https://img.shields.io/badge/built%20on-PyKale-5699C6)](https://github.com/pykale/pykale)
[![CI](https://github.com/pykale/cancer/actions/workflows/ci.yml/badge.svg)](https://github.com/pykale/cancer/actions/workflows/ci.yml)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/pykale/cancer/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)

[Getting Started](https://github.com/pykale/cancer#how-to-use) |
[Documentation](https://github.com/pykale/cancer/tree/main/docs) |
[Examples](https://github.com/pykale/cancer/tree/main/examples) |
[Contributing](https://github.com/pykale/cancer#step-2-building-and-contributing) |
[Architecture](https://github.com/pykale/cancer/blob/main/docs/architecture.md)

KaleCancer is an oncology library built on [PyKale](https://github.com/pykale/pykale), a library in the [PyTorch ecosystem](https://pytorch.org/ecosystem/), aiming to make cancer machine learning more accessible to interdisciplinary research by bridging gaps between clinical data, software, and end users. Both machine learning experts and clinical researchers can do better research with our accessible, scalable, and sustainable design, guided by green machine learning principles. KaleCancer inherits PyKale's unified *pipeline-based* API and extends it where oncology needs more: [multimodal learning](https://en.wikipedia.org/wiki/Multimodal_learning) across pathology, radiology, and clinical records, and [survival analysis](https://en.wikipedia.org/wiki/Survival_analysis) for time-to-event outcomes such as overall and disease-specific survival.

Cancer prediction is rarely single-modality. A prognosis draws on a whole-slide image, a CT or MRI scan, and a clinical record at once, and its outcome is a *time* to an event that is usually censored rather than a class label. [MONAI](https://monai.io/) covers 3D medical imaging well but provides neither multimodal fusion nor survival modelling, while general survival libraries provide neither the imaging encoders nor a shared workflow. KaleCancer supplies both in one pipeline, so that a unimodal baseline and a fused multimodal model are the same code path under a different configuration.

KaleCancer enforces the same *standardization* and *minimalism* as PyKale, via green machine learning concepts of *reducing* repetitions and redundancy, *reusing* existing resources, and *recycling* learning models across areas. Modules are built on PyKale rather than beside it, and the library stays free of dataset-specific choices: anything naming a dataset, an endpoint column, a file pattern or a published split belongs in `examples`.

#### Pipeline-based API

- `loaddata` loads data as input: a `MultimodalDataset` of tables and patch features keyed by patient id, with time-to-event and classification targets and leakage-safe, patient-level splitting
- `prepdata` preprocesses data to fit machine learning modules below (`TableTransform`, fitted on the training patients only)
- `model` embeds and predicts: encoders (attention multiple-instance learning, MLP, TabICL), unimodal, early, intermediate and late fusion models, and the `CoxHead` and `ClassificationHead` with the losses that train them
- `evaluate` evaluates the performance using some metrics, including Harrell's and Uno's C, IPCW time-dependent AUC, AUROC and balanced accuracy, and patient-level `cross_validate`
- `interpret` interprets the outputs via post-prediction analysis, currently exporting attention weights with their patch coordinates
- `pipeline` specifies a machine learning workflow: one `Pipeline` takes a model, transforms and training settings, and works the same for every model and target. It can be built in Python or loaded from a YAML config

#### Example usage

- `examples` demonstrate real applications on specific datasets with a standardized structure.

## How to Use

### Step 0: Installation

KaleCancer supports Python 3.11 or 3.12. Before installing `kalecancer`, we suggest you to first [install PyTorch](https://pytorch.org/get-started/locally/) matching your hardware. PyKale and the other core dependencies are installed with `kalecancer`.

Install `kalecancer` with [uv](https://docs.astral.sh/uv/) into your own project:

```bash
uv add "kalecancer @ git+https://github.com/pykale/cancer.git"
```

Or with pip:

```bash
pip install "git+https://github.com/pykale/cancer.git"
```

Or clone the repository, which is also how to run the examples and develop. With uv, this installs the exact versions in `uv.lock`, including the development tools:

```bash
git clone https://github.com/pykale/cancer.git
cd cancer
uv sync --all-extras
```

With pip, from the clone (the `--group dev` option needs pip 25.1 or newer):

```bash
pip install -e ".[tabular]" --group dev
```

Heavy libraries are kept out of the core install and grouped into extras, so that a pathology workflow does not pull in a radiology stack. Add one with `--extra <name>` in uv, or `".[<name>]"` in pip:

| Extra | Packages | When to use |
| --- | --- | --- |
| `tabular` | tabicl | Clinical tables through a tabular foundation model |
| `imaging` | monai, nibabel, pydicom, SimpleITK | DICOM / NIfTI workflows (declared, not yet used by the library) |
| `pathology` | openslide-python, tifffile | Reading whole-slide images (declared, not yet used by the library) |
| `interpret` | shap, captum, umap-learn | Model explanation (declared, not yet used by the library) |

The development tools (pytest, ruff, mypy, pre-commit) are a dependency group rather than an extra.

For the examples, see [the quickstart guide](https://github.com/pykale/cancer/blob/main/docs/quickstart.md).

### Step 1: Tutorials and Examples

Start with the [quickstart](https://github.com/pykale/cancer/blob/main/docs/quickstart.md), which lists each example and how they differ. The synthetic example needs no data and runs on a CPU in under a minute; the HANCOCK examples read a local copy of the [HANCOCK dataset](https://hancock.research.fau.eu/) (CC BY 4.0) from `data/hancock/`.

Browse through the [**examples**](https://github.com/pykale/cancer/tree/main/examples) to see the usage of KaleCancer, from a synthetic cohort to fused multimodal models on real data. Run them from the repository root:

```bash
# Synthetic cohort: intermediate fusion with a Cox head, CPU only, downloads nothing
uv run python examples/synthetic/synthetic_survival.py

# HANCOCK: clinical table + whole-slide patch features, intermediate fusion, Cox
uv run python examples/hancock/survival_intermediate.py

# The same pipeline loaded from a YAML config
uv run python examples/hancock/from_config.py

# Five-fold cross-validation, and late-fusion classification
uv run python examples/hancock/cross_validation.py
uv run python examples/hancock/classification_late.py
```

Each example is a standalone script, and the model is a dictionary of stages per modality, so changing how modalities combine is a change to the model rather than to a pipeline. See [multimodal fusion](https://github.com/pykale/cancer/blob/main/docs/multimodal_fusion.md) for the models and fusion methods available.

Ask questions on [PyKale's GitHub Discussions tab](https://github.com/pykale/pykale/discussions) if you need help or create an [issue](https://github.com/pykale/cancer/issues) if you find something wrong.

### Step 2: Building and Contributing

Build new modules and/or projects with KaleCancer referring to the [architecture guide](https://github.com/pykale/cancer/blob/main/docs/architecture.md), e.g., on how to modify an existing pipeline or build a new one. New code belongs in the pipeline stage that names what it does, and anything specific to one dataset belongs in `examples` rather than in the library.

This is an open-source project welcoming your contributions. You can contribute in three ways:

- [Star](https://docs.github.com/en/github/getting-started-with-github/saving-repositories-with-stars) and [fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) KaleCancer to follow its latest developments, share it with your networks, and [ask questions](https://github.com/pykale/pykale/discussions) about it.
- Use KaleCancer in your project and let us know any bugs (& fixes) and feature requests/suggestions via creating an [issue](https://github.com/pykale/cancer/issues).
- Contribute via [branch, fork, and pull](https://github.com/pykale/pykale/blob/main/.github/CONTRIBUTING.md#branch-fork-and-pull) for minor fixes and new features, functions, or examples to become one of the [contributors](https://github.com/pykale/cancer/graphs/contributors).

Run the same checks as CI before opening a pull request:

```bash
uv sync --all-extras
uv run pre-commit run --all-files && uv run pytest --cov=kalecancer
```

Conventions and architectural constraints are documented in [AGENTS.md](https://github.com/pykale/cancer/blob/main/AGENTS.md), which applies to human and automated contributors alike. See PyKale's [contributing guidelines](https://github.com/pykale/pykale/blob/main/.github/CONTRIBUTING.md) for more details. The participation in this open source project is subject to PyKale's [Code of Conduct](https://github.com/pykale/pykale/blob/main/.github/CODE_OF_CONDUCT.md).

## Who We Are

### The Team

KaleCancer is developed within the [PyKale](https://github.com/pykale/pykale) project at the University of Sheffield, with contributions from many other [contributors](https://github.com/pykale/cancer/graphs/contributors).

### Citation

KaleCancer does not have a publication of its own yet. Please consider citing the PyKale [CIKM2022 paper](https://doi.org/10.1145/3511808.3557676) below if you find _KaleCancer_ useful to your research.

```lang-latex
    @inproceedings{pykale-cikm2022,
      title     = {{PyKale}: Knowledge-Aware Machine Learning from Multiple Sources in {Python}},
      author    = {Haiping Lu and Xianyuan Liu and Shuo Zhou and Robert Turner and Peizhen Bai and Raivo Koot and Mustafa Chasmai and Lawrence Schobs and Hao Xu},
      booktitle = {Proceedings of the 31st ACM International Conference on Information and Knowledge Management (CIKM)},
      doi       = {10.1145/3511808.3557676},
      year      = {2022}
    }
```

### Acknowledgements

KaleCancer is built on [PyKale](https://github.com/pykale/pykale) and inherits its [acknowledgements](https://github.com/pykale/pykale#acknowledgements). The examples use the [HANCOCK dataset](https://hancock.research.fau.eu/), a multimodal head and neck cancer cohort released under CC BY 4.0 and licensed separately from this software.
