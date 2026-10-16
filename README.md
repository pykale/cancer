<p align="center">
  <img src="https://raw.githubusercontent.com/pykale/cancer/main/docs/KaleCancer_Horizontal_PyKale_Background.png" alt="KaleCancer" width="600"/>
</p>

> *Multimodal machine learning for oncology: whole-slide pathology, radiology, and clinical records, with time-to-event prediction.*

-----------------------------------------

<!-- Keep badges to just ONE line, i.e. only the most important badges! -->
[![Built on PyKale](https://img.shields.io/badge/built%20on-PyKale-5699C6)](https://github.com/pykale/pykale)
[![CI](https://github.com/pykale/cancer/actions/workflows/ci.yml/badge.svg)](https://github.com/pykale/cancer/actions/workflows/ci.yml)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)

[Getting Started](#how-to-use) |
[Documentation](docs/) |
[Examples](examples/) |
[Contributing](#step-2-building-and-contributing) |
[Architecture](docs/architecture.md)

KaleCancer is an oncology library built on [PyKale](https://github.com/pykale/pykale), a library in the [PyTorch ecosystem](https://pytorch.org/ecosystem/), aiming to make cancer machine learning more accessible to interdisciplinary research by bridging gaps between clinical data, software, and end users. Both machine learning experts and clinical researchers can do better research with our accessible, scalable, and sustainable design, guided by green machine learning principles. KaleCancer inherits PyKale's unified *pipeline-based* API and extends it where oncology needs more: [multimodal learning](https://en.wikipedia.org/wiki/Multimodal_learning) across pathology, radiology, and clinical records, and [survival analysis](https://en.wikipedia.org/wiki/Survival_analysis) for time-to-event outcomes such as overall and disease-specific survival.

Cancer prediction is rarely single-modality. A prognosis draws on a whole-slide image, a CT or MRI scan, and a clinical record at once, and its outcome is a *time* to an event that is usually censored rather than a class label. [MONAI](https://monai.io/) covers 3D medical imaging well but provides neither multimodal fusion nor survival modelling, while general survival libraries provide neither the imaging encoders nor a shared workflow. KaleCancer supplies both in one pipeline, so that a unimodal baseline and a fused multimodal model are the same code path under a different configuration.

#### Pipeline-based API

- `loaddata` loads tables and patch features keyed by patient id, with survival and classification targets and patient-level splitting
- `prepdata` preprocesses clinical tables
- `model` encodes each modality, fuses them (unimodal, early, intermediate or late) and predicts with a survival or classification head
- `evaluate` computes survival and classification metrics with bootstrap confidence intervals, and cross-validates by patient
- `interpret` explains predictions, starting with attention weights over patches
- `pipeline` runs the workflow: one `Pipeline` for every model and target, built in Python or from YAML

See [architecture](docs/architecture.md) for more details.

## How to Use

### Step 0: Installation

KaleCancer supports Python 3.11 or 3.12.

Install `kalecancer` with [uv](https://docs.astral.sh/uv/) into your own project:

```bash
uv add "kalecancer @ git+https://github.com/pykale/cancer.git"
```

Or with pip:

```bash
pip install "git+https://github.com/pykale/cancer.git"
```

Heavy libraries are kept out of the core install and grouped into extras, so that a pathology workflow does not pull in a radiology stack. Add one with `--extra <name>` in uv, or `".[<name>]"` in pip:

| Extra | Packages | When to use |
| --- | --- | --- |
| `tabular` | tabicl | Clinical tables through a tabular foundation model |

### Step 1: Tutorials and Examples

Run the synthetic example from a clone of the repository. It generates its own cohort, trains an intermediate-fusion Cox model and runs on a CPU in under a minute:

```bash
uv run python examples/synthetic/synthetic_survival.py
```

The [quickstart](docs/quickstart.md) lists the other [examples](examples/), including fused multimodal models on real data, and how they differ.

Ask questions on [PyKale's GitHub Discussions tab](https://github.com/pykale/pykale/discussions) if you need help or create an [issue](https://github.com/pykale/cancer/issues) if you find something wrong.

### Step 2: Building and Contributing

This is an open-source project welcoming your contributions. You can contribute in three ways:

- [Star](https://docs.github.com/en/github/getting-started-with-github/saving-repositories-with-stars) and [fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) KaleCancer to follow its latest developments, share it with your networks, and [ask questions](https://github.com/pykale/pykale/discussions) about it.
- Use KaleCancer in your project and let us know any bugs (& fixes) and feature requests/suggestions via creating an [issue](https://github.com/pykale/cancer/issues).
- Contribute new features, functions, examples or fixes by following the [contribution guide](CONTRIBUTING.md) to become one of the [contributors](https://github.com/pykale/cancer/graphs/contributors).

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
