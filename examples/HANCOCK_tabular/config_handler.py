"""Config-driven wiring for the HANCOCK tabular demo.

Everything here turns ``configs/config.yaml`` into the objects ``main.py`` runs:
the dataset, its transforms, the split and the encoder, plus the demo's Cox head,
loss and metric. ``main.py`` keeps the training loop, so the cohort, the columns
and the preprocessing can all be changed from the config without touching it.

Transforms are named in the config and looked up in :data:`TRANSFORMS` here.

Example:
    >>> config = load_config("configs/config.yaml")
    >>> cohort = build_dataset(config)
    >>> train, test = split_dataset(cohort, config)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from torch import nn

from kalecancer.loaddata.tabular import TabularDataset
from kalecancer.model.embed import TabICLEncoder
from kalecancer.survival.survival_target import SurvivalTarget

#: Config lives next to this file, so the demo runs from any working directory.
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "config.yaml"

#: Transformer names a config may use. Add an import above and an entry here to extend.
TRANSFORMS: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        SimpleImputer,
        KNNImputer,
        StandardScaler,
        MinMaxScaler,
        MaxAbsScaler,
        RobustScaler,
        QuantileTransformer,
        PowerTransformer,
        OneHotEncoder,
        OrdinalEncoder,
    )
}


# --------------------------------------------------------------------------- #
# reading the config
# --------------------------------------------------------------------------- #


def load_config(path: str | Path | None = None) -> dict:
    """Read a YAML config into a dict.

    Args:
        path (str | Path | None, optional): Config to read. Defaults to
            ``configs/config.yaml`` beside this file.

    Returns:
        dict: The parsed config.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {path} must be a YAML mapping, got {type(config).__name__}.")
    return config


def config_from_cli(argv: list[str] | None = None) -> dict:
    """Read the config named by ``--config``, or the default one."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config to run (default: {DEFAULT_CONFIG}).",
    )
    return load_config(parser.parse_args(argv).config)


def section(config: dict, name: str) -> dict:
    """Return one top-level section of the config, or an empty one if absent."""
    value = config.get(name) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping, got {type(value).__name__}.")
    return value


def _seeded(value: Any, config: dict) -> Any:
    """Resolve a ``random_state`` entry: ``None`` in the config means the global seed."""
    return config.get("seed") if value is None else value


def set_seed(config: dict) -> int | None:
    """Seed torch from the config's top-level ``seed``, and return it."""
    seed = config.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
    return seed


# --------------------------------------------------------------------------- #
# building the objects
# --------------------------------------------------------------------------- #


def build_transform(spec: Any) -> list | None:
    """Instantiate the transformers named by a role's ``transform`` entry.

    Args:
        spec: ``None``, a single ``{Name: {kwargs}}`` mapping or bare ``Name``
            string, or a list of either -- applied in the order given.

    Returns:
        list | None: Unfitted transformer instances, or ``None`` for passthrough.
    """
    if spec is None:
        return None
    steps = spec if isinstance(spec, list) else [spec]

    kwargs: dict[str, Any]
    transformers = []
    for step in steps:
        if isinstance(step, str):
            name, kwargs = step, {}
        elif isinstance(step, dict) and len(step) == 1:
            ((name, kwargs),) = step.items()
            kwargs = kwargs or {}
        else:
            raise ValueError(
                f"Each transform step must be 'Name' or {{Name: {{kwargs}}}} with exactly one "
                f"name, got {step!r}."
            )
        if name not in TRANSFORMS:
            raise ValueError(
                f"Unknown transform '{name}'. Available: {sorted(TRANSFORMS)}. To use another "
                f"scikit-learn transformer, import it in pipeline.py and add it to TRANSFORMS."
            )
        transformers.append(TRANSFORMS[name](**kwargs))
    return transformers or None


def build_target(config: dict) -> SurvivalTarget | None:
    """Build the ``SurvivalTarget``, or ``None`` if the config declares no target."""
    target = section(config, "target")
    if not target:
        return None
    return SurvivalTarget(**target)


def build_dataset(config: dict) -> TabularDataset:
    """Build the unfitted ``TabularDataset`` the config describes."""
    data = section(config, "data")
    features = section(config, "features")
    continuous = section(features, "continuous")
    categorical = section(features, "categorical")

    return TabularDataset(
        data["source"],
        identifier=data["identifier"],
        target=build_target(config),
        continuous=continuous.get("columns", []),
        continuous_transform=build_transform(continuous.get("transform")),
        categorical=categorical.get("columns", []),
        categorical_transform=build_transform(categorical.get("transform")),
    )


def split_dataset(cohort: TabularDataset, config: dict) -> tuple[TabularDataset, TabularDataset]:
    """Split a cohort into train and test, both unfitted."""
    split = section(config, "split")
    return cohort.split(
        test_size=split.get("test_size", 0.2),
        random_state=_seeded(split.get("random_state"), config),
    )


def build_encoder(config: dict) -> TabICLEncoder:
    """Build the unfitted ``TabICLEncoder`` the config describes."""
    encoder = dict(section(config, "encoder"))
    encoder["random_state"] = _seeded(encoder.get("random_state"), config)
    return TabICLEncoder(**encoder)


# --------------------------------------------------------------------------- #
# Demo only: a trainable head on top of the frozen TabICL representations.
# --------------------------------------------------------------------------- #

class CoxHead(nn.Module):
    """One log-risk score per patient. Higher score = higher hazard = shorter survival."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def build_head(in_dim: int, config: dict) -> nn.Module:
    """The linear layer that actually trains; the frozen representations feed it."""
    hidden_dim = section(config, "head").get("hidden_dim", 32)
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        CoxHead(hidden_dim),
    )


def build_optimiser(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Adam over the head's parameters, with the config's learning rate and decay."""
    head = section(config, "head")
    return torch.optim.Adam(
        model.parameters(),
        lr=head.get("learning_rate", 1e-3),
        weight_decay=head.get("weight_decay", 1e-4),
    )


def cox_ph_loss(risk, time, event):
    """Negative Cox partial log-likelihood (Breslow), averaged over observed events.

    Each patient who had the event contributes their risk minus the log-sum-exp of
    the risks of everyone still at risk at that time, so the loss only ever compares
    patients against those who outlived them -- which is exactly what the c-index
    then measures.
    """
    order = torch.argsort(time, descending=True)  # risk set = a running prefix once sorted
    risk, event = risk[order], event[order]
    log_risk_set = torch.logcumsumexp(risk, dim=0)
    return -((risk - log_risk_set) * event).sum() / event.sum().clamp(min=1)


def c_index(risk, time, event):
    """Harrell's concordance index over comparable pairs. 0.5 = coin flip, 1.0 = perfect."""
    risk, time, event = (np.asarray(a, dtype=float) for a in (risk, time, event))
    concordant = permissible = 0.0
    for i in range(len(time)):
        if not event[i]:
            continue  # a censored patient's ordering against later times is unknown
        comparable = time > time[i]
        concordant += (risk[i] > risk[comparable]).sum() + 0.5 * (risk[i] == risk[comparable]).sum()
        permissible += comparable.sum()
    return concordant / permissible
