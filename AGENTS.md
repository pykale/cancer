# AGENTS.md

Instructions for coding agents working in this repository. Human contributors should
read [README.md](README.md) and [CONTRIBUTING](README.md#contributing).

## Setup

Python 3.10–3.12 is required. Install PyTorch first if a specific CUDA build is
needed, then:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"     # Windows: .venv/Scripts/pip
```

Verify the install:

```bash
.venv/bin/python -c "import kalecancer, torch, kale; print(kalecancer.__version__)"
```

## Commands

| Task | Command |
| --- | --- |
| Run tests | `.venv/bin/pytest` |
| Run tests with coverage | `.venv/bin/pytest --cov=kalecancer --cov-report=term-missing` |
| Format | `.venv/bin/ruff format .` |
| Lint | `.venv/bin/ruff check .` |
| Run the WSI pipeline on local files | `.venv/bin/kalecancer wsi-survival --features <dir> --clinical <file> --preset quick` |
| Fetch and run on HANCOCK | `.venv/bin/python examples/wsi_survival/main.py --cfg examples/wsi_survival/configs/hancock_primary_tumour_quick.yaml DATASET.SOURCE hancock DATASET.PATIENTS 50` |
| Inspect a configuration | `.venv/bin/kalecancer wsi-survival --print-config` |

Both `pre-commit run --all-files` (ruff, ruff-format, mypy) and `pytest` must pass
before a change is complete. These are the same checks CI runs.

## Data

Two sources, selected by `DATASET.SOURCE`:

- `local` reads an existing copy from `DATASET.FEATURE_ROOT` and
  `DATASET.CLINICAL_PATH`. This is the only source the `kalecancer` command
  supports, because downloading needs a fetcher that knows the dataset layout.
- `hancock` fetches the published dataset (CC BY 4.0) over HTTP range requests,
  transferring only the selected patients out of a 9.65 GB archive and caching them.
  Requires no credentials. Its fetcher lives in `examples/wsi_survival/hancock.py`
  and is passed to `run(..., fetch=...)` by that example, so use the example script
  rather than the CLI. Keep it small with `DATASET.PATIENTS N`.

Tests need neither: they build synthetic HDF5 cohorts in a temporary directory, and
must never reach the network. `tests/conftest.py` shows the pattern, and
`tests/loaddata/test_sources.py` serves archives from a local HTTP server.

A cohort of 50–60 patients leaves fewer than 10 in the test split, which is enough to
verify the pipeline runs but not to produce a meaningful metric. Report such runs as
verification, not as results.

There is no GPU in a Codespace or dev container. Use it for development, tests, and
small demonstration runs; train full cohorts on a GPU machine.

## Layout

```
kalecancer/
├── loaddata/    HDF5 patch features, clinical labels, cohort matching, splitting
├── prepdata/    transforms
├── model/       encoders (attention MIL), multimodal fusion strategies
├── pipeline/    trainers and end-to-end runners
├── survival/    Cox head, loss, metrics, baseline hazard
├── evaluate/    metrics and prediction reports
├── interpret/   attention export
├── utils/       seeding, artefact writing
├── config.py    configuration schema and defaults
└── cli.py       command-line interface
```

Tests mirror this layout: code in `kalecancer/loaddata/split.py` is tested in
`tests/loaddata/test_split.py`.

## Conventions

- Follow [PyKale](https://github.com/pykale/pykale) conventions: verb-oriented stages,
  Google-style docstrings, type hints, YACS configuration.
- Reuse PyKale APIs where one exists rather than reimplementing.
- Line length 120. Formatting is enforced by ruff-format; do not hand-format.
- Comments explain why, not what. Do not restate the code.
- New configuration belongs in `kalecancer/config.py`, never hardcoded in a module.
- Dataset paths appear only in example configs and command-line arguments.

## Constraints

- `kalecancer/survival/` must not import from any other `kalecancer` module. It is
  cancer-agnostic and destined for PyKale core. `tests/test_survival_boundary.py`
  enforces this and will fail the build.
- Splitting is patient-level. Never split on slides or patches; a patient's slides
  must never span two splits.
- Censoring-aware quantities (IPCW weights, baseline hazard) are estimated from the
  training split only.
- The event convention is `1` observed, `0` censored, normalised at load time.
- Risk scores are log partial hazards where higher means higher risk.

## Adding a component

1. Implement it in the matching `kalecancer/` subpackage.
2. Export it from that subpackage's `__init__.py`.
3. Add tests in the mirrored `tests/` path, using synthetic data.
4. Add any new setting to `kalecancer/config.py` with a comment.
5. Run the format, lint, and test commands above.
