"""Writing experiment artefacts in machine-readable formats."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plain(value: Any) -> Any:
    """Convert a NumPy scalar or array to its Python equivalent."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serialisable")


def write_json(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as indented JSON.

    NumPy scalars and arrays are converted, because metrics reach this function from
    NumPy and scikit-learn far more often than from plain Python.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=_plain)
        handle.write("\n")
    return path


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str] | None = None) -> Path:
    """Write ``rows`` as CSV.

    Without ``fieldnames`` the header is the union of the keys present, in the order
    first seen, and a row missing one of them is written blank. Passing ``fieldnames``
    is a declaration that every row matches it, so an unexpected key is an error.
    """
    rows = list(rows)
    path = Path(path)
    ensure_dir(path.parent)

    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    with path.open("w", encoding="utf-8", newline="") as handle:
        header = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=header, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path
