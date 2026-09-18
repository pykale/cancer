"""Build a Pipeline from a YAML file or a dict, and dump a pipeline back to one.

The format is jsonargparse's ``class_path``/``init_args``, as used by LightningCLI. Dumping relies on one
convention: every component stores each ``__init__`` argument under the same attribute name.
"""

from __future__ import annotations

import copy
import functools
import importlib.metadata
import inspect
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from jsonargparse import ArgumentParser
from torch import nn

from kalecancer.pipeline.pipeline import Pipeline

_VERSION_KEY = "kalecancer_version"


_PLAIN = (bool, int, float, str, type(None))


_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def load_pipeline(config: str | Path | Mapping[str, Any]) -> Pipeline:
    """Build a Pipeline from a YAML path or a dict. Unknown arguments, wrong types and missing arguments raise."""
    if isinstance(config, str | Path):
        config = yaml.safe_load(Path(config).read_text())
    if not isinstance(config, Mapping):
        raise TypeError(f"a pipeline config must be a mapping, got {type(config).__name__}")
    # jsonargparse turns the nested dicts it is given into Namespaces in place
    config = copy.deepcopy(dict(config))
    written_by = config.pop(_VERSION_KEY, None)
    _check_required_arguments(config, "pipeline")
    installed = importlib.metadata.version("kalecancer")
    if written_by is not None and written_by != installed:
        warnings.warn(
            f"this config was written by kalecancer {written_by}; the installed version is {installed}",
            UserWarning,
            stacklevel=2,
        )
    parser = ArgumentParser(exit_on_error=False)
    parser.add_argument("--pipeline", type=Pipeline)
    return parser.instantiate(parser.parse_object({"pipeline": config})).pipeline


def dump_config(pipeline: Pipeline) -> dict[str, Any]:
    """Every constructor argument of the pipeline, recursively, in the form ``load_pipeline`` reads.

    Key order is meaningful (it fixes feature and concatenation order): write YAML with
    ``yaml.safe_dump(config, sort_keys=False)``.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"expected a Pipeline, got {type(pipeline).__name__}")
    config = _dump(pipeline, "pipeline")
    config[_VERSION_KEY] = importlib.metadata.version("kalecancer")
    return config


def _import(class_path: str) -> Any:
    module, _, name = class_path.rpartition(".")
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError, ValueError):
        return None


def _check_required_arguments(node: Any, path: str) -> None:
    """Raise when a config omits an argument that has no default.

    jsonargparse fills an omitted ``X | None`` argument with ``None``; for ``random_state`` that silently means
    "not seeded". Unknown classes are left for jsonargparse to report.
    """
    if isinstance(node, list):
        for k, item in enumerate(node):
            _check_required_arguments(item, f"{path}[{k}]")
        return
    if not isinstance(node, Mapping):
        return
    if "class_path" not in node:
        for key, value in node.items():
            _check_required_arguments(value, f"{path}.{key}")
        return
    target = _import(node["class_path"])
    init_args = node.get("init_args") or {}
    if isinstance(target, type):
        parameters = inspect.signature(target).parameters
        missing = [
            name
            for name, parameter in parameters.items()
            if parameter.kind not in _VARIADIC
            and parameter.default is inspect.Parameter.empty
            and name not in init_args
            # an optimizer's parameters are supplied when the pipeline builds it
            and not (name == "params" and issubclass(target, torch.optim.Optimizer))
        ]
        if missing:
            raise ValueError(f"{path}: {node['class_path']} is missing required arguments {missing}")
    for key, value in init_args.items():
        _check_required_arguments(value, f"{path}.{key}")


def _qualified(target: Any) -> str:
    """The shortest ``package.subpackage`` path that exports ``target``, the way scikit-learn and torch present their
    classes (``kalecancer.model.IntermediateFusion``, ``torch.nn.Linear``), so configs survive moves between files."""
    parts = target.__module__.split(".")
    if target.__qualname__ == target.__name__:
        for depth in range(2, len(parts)):
            package = ".".join(parts[:depth])
            if getattr(importlib.import_module(package), target.__name__, None) is target:
                return f"{package}.{target.__name__}"
    return f"{target.__module__}.{target.__qualname__}"


def _dump(value: Any, path: str) -> Any:
    if isinstance(value, _PLAIN):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list | tuple | nn.ModuleList):
        return [_dump(item, f"{path}[{k}]") for k, item in enumerate(value)]
    if isinstance(value, Mapping | nn.ModuleDict):
        return {key: _dump(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, functools.partial):
        return _dump_partial(value, path)
    return _dump_object(value, path)


def _dump_partial(value: functools.partial, path: str) -> dict[str, Any]:
    target, args = value.func, value.args
    # jsonargparse builds callables such as optimizers as partial(default_class_instantiator, cls, **kwargs)
    if args and isinstance(args[0], type) and getattr(target, "__name__", "") == "default_class_instantiator":
        target, args = args[0], args[1:]
    if args:
        raise TypeError(f"{path}: cannot dump a partial with positional arguments {args}")
    init_args = {}
    # the first parameter is supplied at call time (e.g. an optimizer's parameters)
    for name, parameter in list(inspect.signature(target).parameters.items())[1:]:
        if parameter.kind in _VARIADIC:
            continue
        if name in value.keywords:
            init_args[name] = _dump(value.keywords[name], f"{path}.{name}")
        elif isinstance(parameter.default, (*_PLAIN, tuple)):
            init_args[name] = _dump(parameter.default, f"{path}.{name}")
    return {"class_path": _qualified(target), "init_args": init_args}


def _dump_object(value: Any, path: str) -> dict[str, Any]:
    cls = type(value)
    is_torch_module = isinstance(value, nn.Module) and cls.__module__.startswith("torch.")
    init_args = {}
    for name, parameter in inspect.signature(cls.__init__).parameters.items():
        if name == "self" or parameter.kind in _VARIADIC or (is_torch_module and name in ("device", "dtype")):
            continue
        if not hasattr(value, name):
            raise TypeError(
                f"{path}: {cls.__name__} does not store its argument {name!r} as an attribute, so it cannot be dumped"
            )
        attribute = getattr(value, name)
        if is_torch_module and isinstance(parameter.default, bool) and isinstance(attribute, nn.Parameter | None):
            # torch keeps e.g. Linear(bias=True) as the bias Parameter itself
            attribute = attribute is not None
        if attribute is parameter.default and not isinstance(attribute, _PLAIN):
            continue
        init_args[name] = _dump(attribute, f"{path}.{name}")
    return {"class_path": _qualified(cls), "init_args": init_args}
