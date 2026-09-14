# KaleCancer

> *Multimodal machine learning for oncology: whole-slide pathology, radiology, and clinical records, with time-to-event prediction.*

-----------------------------------------

<!-- Keep badges to just ONE line, i.e. only the most important badges! -->
[![Built on PyKale](https://img.shields.io/badge/built%20on-PyKale-5699C6)](https://github.com/pykale/pykale)
[![CI](https://github.com/pykale/cancer/actions/workflows/ci.yml/badge.svg)](https://github.com/pykale/cancer/actions/workflows/ci.yml)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/pykale/cancer/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)

[Getting Started](https://github.com/pykale/cancer#how-to-use) |
[Documentation](https://github.com/pykale/cancer/tree/main/docs) |
[Examples](https://github.com/pykale/cancer/tree/main/examples) |
[Contributing](https://github.com/pykale/cancer#step-2-building-and-contributing) |
[Architecture](https://github.com/pykale/cancer/blob/main/docs/architecture.md)

KaleCancer is an oncology library built on [PyKale](https://github.com/pykale/pykale), a library in the [PyTorch ecosystem](https://pytorch.org/ecosystem/), aiming to make cancer machine learning more accessible to interdisciplinary research by bridging gaps between clinical data, software, and end users. Both machine learning experts and clinical researchers can do better research with our accessible, scalable, and sustainable design, guided by green machine learning principles. KaleCancer inherits PyKale's unified *pipeline-based* API and extends it where oncology needs more: [multimodal learning](https://en.wikipedia.org/wiki/Multimodal_learning) across pathology, radiology, and clinical records, and [survival analysis](https://en.wikipedia.org/wiki/Survival_analysis) for time-to-event outcomes such as overall and disease-specific survival.

Cancer prediction is rarely single-modality. A prognosis draws on a whole-slide image, a CT or MRI scan, and a clinical record at once, and its outcome is a *time* to an event that is usually censored rather than a class label. [MONAI](https://monai.io/) covers 3D medical imaging well but provides neither multimodal fusion nor survival modelling, while general survival libraries provide neither the imaging encoders nor a shared workflow. KaleCancer supplies both in one pipeline, so that a unimodal baseline and a fused multimodal model are the same code path under a different configuration.

KaleCancer enforces the same *standardization* and *minimalism* as PyKale, via green machine learning concepts of *reducing* repetitions and redundancy, *reusing* existing resources, and *recycling* learning models across areas. Modules are built on PyKale rather than beside it, and the `survival` module is deliberately kept cancer-agnostic, importing only PyTorch and NumPy so that it can migrate into PyKale core unchanged. This boundary is enforced in CI.

#### Pipeline-based API

- `loaddata` loads data from disk or online resources as input, including cohort matching and leakage-safe splitting
- `prepdata` preprocesses data to fit machine learning modules below (transforms)
- `model.embed` embeds data in a new space to learn a new representation (attention multiple-instance learning over patch bags, tabular encoders, and multimodal fusion)
- `model.predict` turns a representation into a score: `LinearHead` and `CoxHead`, with the losses that train them
- `evaluate` evaluates the performance using some metrics, including IPCW time-dependent AUC and the integrated Brier score
- `interpret` interprets the features and outputs via post-prediction analysis mainly via visualization
- `auto` selects and constructs a workflow from a configuration alone (`AutoCancer*` classes, planned)
- `pipeline` specifies a machine learning workflow by combining several other modules: a *trainer* says where the data comes from, a *task* (`SurvivalTask`, `ClassificationTask`) says what is predicted from it, and any trainer takes any task

#### Example usage

- `examples` demonstrate real applications on specific datasets with a standardized structure.

## How to Use

### Step 0: Installation

KaleCancer supports Python 3.10, 3.11, or 3.12. Before installing `kalecancer`, we suggest you to first [install PyTorch](https://pytorch.org/get-started/locally/) matching your hardware, and then [install PyKale](https://pykale.readthedocs.io/en/latest/installation.html) following its official instructions.

Installation of `kalecancer` from source:

```bash
git clone https://github.com/pykale/cancer.git
cd cancer
pip install -e .
```

Heavy libraries are kept out of the core install and grouped into extras, so that a pathology workflow does not pull in a radiology stack:

| Extra | Packages | When to use |
| --- | --- | --- |
| `tabular` | tabicl | Clinical tables through a tabular foundation model |
| `imaging` | monai, nibabel, pydicom, SimpleITK | DICOM / NIfTI workflows |
| `pathology` | openslide-python, tifffile | Reading whole-slide images |
| `interpret` | shap, captum | Model explanation |
| `dev` | pytest, ruff, mypy, pre-commit | Development and CI |

For more details and other options, please refer to [the quickstart guide](https://github.com/pykale/cancer/blob/main/docs/quickstart.md).

### Step 1: Tutorials and Examples

Start with the [quickstart](https://github.com/pykale/cancer/blob/main/docs/quickstart.md), which walks through the same run from the command line, from Python, and from a coding agent. No account or credentials are needed: the examples stream the published [HANCOCK dataset](https://hancock.research.fau.eu/) (CC BY 4.0) over HTTP range requests, so only the patients you ask for are transferred.

Browse through the [**examples**](https://github.com/pykale/cancer/tree/main/examples) to see the usage of KaleCancer in performing survival prediction in a wide range of settings, from a single modality to fused multimodal models:

```bash
# Whole-slide pathology only: attention MIL with a Cox head
python -m examples.hancock_wsi_survival.main --cfg examples/hancock_wsi_survival/configs/hancock_primary_tumour_quick.yaml

# Clinical records only, imaging only, or both, on the official HANCOCK split
python -m examples.hancock_multimodal_survival.main --cfg examples/hancock_multimodal_survival/configs/tabular.yaml
python -m examples.hancock_multimodal_survival.main --cfg examples/hancock_multimodal_survival/configs/imaging.yaml
python -m examples.hancock_multimodal_survival.main --cfg examples/hancock_multimodal_survival/configs/multimodal.yaml

# How the modalities combine is a configuration change, not a code change
python -m examples.hancock_multimodal_survival.main --cfg examples/hancock_multimodal_survival/configs/multimodal.yaml FUSION.STAGE late
```

Each example follows the standardized structure of `main.py`, `config.py`, and `configs/*.yaml`, so an experiment is described by its configuration rather than by edited code. See [multimodal fusion](https://github.com/pykale/cancer/blob/main/docs/multimodal_fusion.md) for the fusion stages and methods available, and the [WSI survival pipeline](https://github.com/pykale/cancer/tree/main/examples/hancock_wsi_survival) for inputs, configuration, and outputs in detail.

Ask questions on [PyKale's GitHub Discussions tab](https://github.com/pykale/pykale/discussions) if you need help or create an [issue](https://github.com/pykale/cancer/issues) if you find something wrong.

### Step 2: Building and Contributing

Build new modules and/or projects with KaleCancer referring to the [architecture guide](https://github.com/pykale/cancer/blob/main/docs/architecture.md), e.g., on how to modify an existing pipeline or build a new one. New code belongs in the pipeline stage that names what it does, and anything specific to one dataset belongs in `examples` rather than in the library.

This is an open-source project welcoming your contributions. You can contribute in three ways:

- [Star](https://docs.github.com/en/github/getting-started-with-github/saving-repositories-with-stars) and [fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) KaleCancer to follow its latest developments, share it with your networks, and [ask questions](https://github.com/pykale/pykale/discussions) about it.
- Use KaleCancer in your project and let us know any bugs (& fixes) and feature requests/suggestions via creating an [issue](https://github.com/pykale/cancer/issues).
- Contribute via [branch, fork, and pull](https://github.com/pykale/pykale/blob/main/.github/CONTRIBUTING.md#branch-fork-and-pull) for minor fixes and new features, functions, or examples to become one of the [contributors](https://github.com/pykale/cancer/graphs/contributors).

Run the same checks as CI before opening a pull request:

```bash
pip install -e ".[dev]"
pre-commit run --all-files && pytest --cov=kalecancer
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
