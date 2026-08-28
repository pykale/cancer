"""Config-driven wiring for the HANCOCK tabular demo.

Turns ``configs/config.yaml`` into the objects ``pipeline.py`` runs, plus the demo's
Cox head, loss and metric. Transforms are named in the config and looked up in
:data:`TRANSFORMS`.

It deliberately does not fit anything or decide which rows a fold sees. There is no
``build_fold(config)`` helper because hiding those three lines would put the one
thing worth checking -- which rows the statistics came from -- out of sight.

Example:
    >>> config = load_config("configs/config.yaml")
    >>> cohort = build_cohort(config)
    >>> train_ids, test_ids = split_identifiers(cohort, config)
"""

from __future__ import annotations

import argparse
import difflib
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

from kalecancer.loaddata import CohortView, SurvivalTarget, TabularCohort
from kalecancer.model.embed import TabICLEmbedder

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

#: Keys each section accepts. A config typo is silent by nature -- mis-spell
#: `transform` and the preprocessing is simply not applied -- so it is checked.
SCHEMA: dict[str | None, set[str]] = {
    None: {"seed", "data", "target", "features", "split", "embedder", "head"},
    "data": {"source", "identifier"},
    "target": {"time", "event", "event_value", "unit"},
    "features": {"continuous", "categorical"},
    "split": {"mode", "file", "test_size", "random_state", "stratify"},
    "embedder": {"context_label", "trainable", "checkpoint", "n_estimators", "device", "random_state"},
    "head": {"hidden_dim", "epochs", "learning_rate", "weight_decay", "log_every"},
}

#: Keys accepted inside `features.continuous` and `features.categorical`.
ROLE_KEYS = {"columns", "transform"}


def _check_keys(mapping: dict, allowed: set[str], where: str) -> None:
    """Reject unrecognised keys, suggesting the intended one where it is obvious."""
    unknown = sorted(set(mapping) - allowed)
    if not unknown:
        return
    described = []
    for key in unknown:
        close = difflib.get_close_matches(key, sorted(allowed), n=1)
        described.append(f"'{key}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
    raise ValueError(f"Unknown key(s) in {where}: {', '.join(described)}. Allowed: {sorted(allowed)}.")


def validate_config(config: dict) -> dict:
    """Check every section's keys, and return the config unchanged."""
    _check_keys(config, SCHEMA[None], "the top level")
    for name, allowed in SCHEMA.items():
        if name is None:
            continue
        _check_keys(section(config, name), allowed, f"'{name}'")
    for role in ("continuous", "categorical"):
        _check_keys(section(section(config, "features"), role), ROLE_KEYS, f"'features.{role}'")
    return config


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
    return validate_config(config)


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
                f"Each transform step must be 'Name' or {{Name: {{kwargs}}}} with exactly one name, got {step!r}."
            )
        if name not in TRANSFORMS:
            raise ValueError(
                f"Unknown transform '{name}'. Available: {sorted(TRANSFORMS)}. To use another "
                f"scikit-learn transformer, import it in config_handler.py and add it to TRANSFORMS."
            )
        transformers.append(TRANSFORMS[name](**kwargs))
    return transformers or None


def build_target(config: dict) -> SurvivalTarget | None:
    """Build the ``SurvivalTarget``, or ``None`` if the config declares no target."""
    target = section(config, "target")
    if not target:
        return None
    return SurvivalTarget(**target)


#: ``data.source`` value meaning "fetch the published table" rather than read a path.
HANCOCK_SOURCE = "hancock"


def resolve_source(source: str):
    """Resolve ``data.source`` to a readable path.

    ``hancock`` fetches the published clinical table into the shared cache, which is
    what the other HANCOCK examples do; anything else is taken as a local path, so a
    private extract can still be pointed at directly.
    """
    if source != HANCOCK_SOURCE:
        return source
    from examples.hancock import HancockDataset

    return HancockDataset().clinical()


def build_cohort(config: dict) -> TabularCohort:
    """Build the cohort the config describes. Holds no fitted state, so reuse it."""
    data = section(config, "data")
    features = section(config, "features")
    continuous = section(features, "continuous")
    categorical = section(features, "categorical")

    return TabularCohort(
        resolve_source(data["source"]),
        identifier=data["identifier"],
        target=build_target(config),
        continuous=continuous.get("columns", []),
        continuous_transform=build_transform(continuous.get("transform")),
        categorical=categorical.get("columns", []),
        categorical_transform=build_transform(categorical.get("transform")),
    )


#: Values of ``split.stratify`` that name something other than a table column.
STRATIFY_KEYWORDS = ("event", "none")


def build_stratify(cohort: TabularCohort, config: dict) -> bool | np.ndarray:
    """Resolve ``split.stratify`` into what :meth:`Cohort.split` expects.

    Accepts ``"event"``, ``"none"``, a column name, or a list of column names combined
    into one stratum per patient.

    Returns:
        bool | np.ndarray: ``True`` to let the target decide, ``False`` for none, or
        one label per row.

    Raises:
        ValueError: If a column is missing, a keyword is ambiguous with a real column,
            or the strata are too small to split.
    """
    split = section(config, "split")
    if "stratify" not in split:
        raise ValueError(
            "split.stratify is required; there is no default. What a split is balanced on "
            "changes the numbers you report, and an unstratified split of a few hundred "
            "patients can easily skew the event rate by several points. Write one of:\n"
            "    stratify: event                  the target's event indicator\n"
            "    stratify: none                   do not stratify\n"
            "    stratify: sex                    a column from the table\n"
            "    stratify: [sex, smoking_status]  several columns combined"
        )
    spec = split["stratify"]

    if spec is None or spec is False:
        return False
    if spec is True:
        return True
    if spec in STRATIFY_KEYWORDS:
        # A survival table may have a column called "event". The list form is always
        # columns and the booleans always keywords, so both readings stay reachable.
        _reject_if_column(cohort, spec)
        return spec == "event"

    columns = [spec] if isinstance(spec, str) else list(spec)
    missing = [c for c in columns if c not in cohort.frame.columns]
    if missing:
        raise ValueError(
            f"stratify names column(s) {missing}, which are not in the table. "
            f"Available: {sorted(cohort.frame.columns)}. Use 'event' for the target's "
            f"event indicator, or 'none' to switch stratification off."
        )

    labels = _strata(cohort, columns)
    _check_strata(labels, columns)
    return labels


def _reject_if_column(cohort: TabularCohort, keyword: str) -> None:
    """Refuse a bare keyword that is also a column name in this table."""
    if keyword not in cohort.frame.columns:
        return
    unambiguous = "true" if keyword == "event" else "false"
    meaning = "the target's event indicator" if keyword == "event" else "no stratification"
    raise ValueError(
        f"This table has a column called '{keyword}', so 'stratify: {keyword}' is ambiguous. "
        f"Write 'stratify: [{keyword}]' to stratify on the column, or 'stratify: {unambiguous}' "
        f"for {meaning}."
    )


def _strata(cohort: TabularCohort, columns: list[str]) -> np.ndarray:
    """One label per patient, joining the named columns.

    A missing value becomes its own ``<missing>`` group -- an unrecorded value is a
    real subgroup, not an error.
    """
    block = cohort.frame[columns]
    # copy(): pandas hands back a read-only view under copy-on-write.
    values = block.to_numpy(dtype=object).copy()
    values[block.isna().to_numpy()] = "<missing>"
    return np.array(["|".join(str(value) for value in row) for row in values])


def _check_strata(labels: np.ndarray, columns: list[str]) -> None:
    """Refuse strata too small to appear in both halves of a split.

    scikit-learn raises too, but names neither the column nor the value.
    """
    values, counts = np.unique(labels, return_counts=True)
    if counts.min() >= 2:
        return
    smallest = values[counts.argmin()]
    raise ValueError(
        f"Stratifying on {columns} gives {len(values)} groups across {len(labels)} patients, "
        f"and the smallest ('{smallest}') has {counts.min()}. Every group needs at least two "
        f"members to appear in both halves. A continuous column is the usual cause -- bin it "
        f"first, or stratify on something coarser."
    )


def split_identifiers(cohort: TabularCohort, config: dict) -> tuple[list[str], list[str]]:
    """Split into train and test **identifiers**, as ``split.mode`` describes.

    ``published`` is the default and applies the dataset's own train/test assignment,
    which is what keeps a result comparable with other work on the cohort. ``random``
    draws a fresh stratified split instead, for a cohort that publishes none.

    Raises:
        ValueError: If ``split.mode`` is neither ``published`` nor ``random``.
    """
    split = section(config, "split")
    mode = split.get("mode", "published")

    if mode == "published":
        from examples.hancock import HancockDataset, official_split

        assignment = official_split(HancockDataset().splits(split.get("file", "dataset_split_in.json")))
        available = set(cohort.identifiers)
        return (
            sorted(available & set(assignment["training"])),
            sorted(available & set(assignment["test"])),
        )
    if mode == "random":
        return cohort.split(
            test_size=split.get("test_size", 0.2),
            random_state=_seeded(split.get("random_state"), config),
            stratify=build_stratify(cohort, config),
        )
    raise ValueError(f"unknown split.mode {mode!r}; expected 'published' or 'random'")


def build_embedder(config: dict, context: CohortView) -> TabICLEmbedder:
    """Build the ``TabICLEmbedder`` the config describes, over ``context``'s rows.

    ``context_label`` is resolved here rather than inside the embedder: which
    supervision value conditions the representation is a modelling choice this
    wiring layer makes, and the embedder only ever sees an array of labels.
    """
    spec = dict(section(config, "embedder"))
    label = spec.pop("context_label")
    spec["random_state"] = _seeded(spec.get("random_state"), config)

    # The batch already carries the supervision under its own names, so the label is
    # a key lookup rather than a reach into the cohort's target.
    batch = context.batch()
    if label not in batch.target:
        available = sorted(batch.target) or "nothing -- this cohort was built without a target"
        raise ValueError(f"embedder.context_label={label!r} is not available. This view offers {available}.")

    return TabICLEmbedder(
        context_x=batch.modalities[context.cohort.name],
        context_y=batch.target[label],
        **spec,
    )


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

    Each patient with the event contributes their risk minus the log-sum-exp of
    everyone still at risk then, so the loss only compares patients against those who
    outlived them -- which is what the c-index measures.

    Restricting this to a minibatch restricts the risk set to that batch, a weaker
    objective, so batch size is a modelling decision rather than a memory one.
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


def supervision(cohort: TabularCohort, view) -> tuple[torch.Tensor, torch.Tensor]:
    """Times and events for a view's patients, in the view's own row order.

    Pulled by identifier, so they line up with what the encoder returned even if the
    view was subset or reordered.
    """
    target = cohort.target
    # times_for/events_for are survival-specific; the Target protocol stops at
    # required_columns/bind/for_, because event language is not universal to targets.
    if not isinstance(target, SurvivalTarget):
        raise ValueError("This pipeline needs a survival target; set 'target:' in the config.")
    return (
        torch.tensor(target.times_for(view.identifiers), dtype=torch.float32),
        torch.tensor(target.events_for(view.identifiers), dtype=torch.float32),
    )
