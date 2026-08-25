"""Tests for artefact writing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from kalecancer.utils import ensure_dir, write_csv, write_json


def test_ensure_dir_creates_parents(tmp_path: Path) -> None:
    created = ensure_dir(tmp_path / "a" / "b")

    assert created.is_dir()


def test_write_json_round_trips(tmp_path: Path) -> None:
    path = write_json(tmp_path / "nested" / "metrics.json", {"c_index": 0.61})

    assert json.loads(path.read_text(encoding="utf-8")) == {"c_index": 0.61}


def test_write_csv_uses_the_first_row_as_header(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "rows.csv", [{"a": 1, "b": 2}, {"a": 3, "b": 4}])

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_write_csv_handles_no_rows(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "empty.csv", [])

    assert path.read_text(encoding="utf-8") == ""


def test_write_json_accepts_numpy_scalars(tmp_path: Path) -> None:
    payload = {"num_events": np.int64(7), "c_index": np.float32(0.5), "risk": np.arange(3)}

    written = json.loads(write_json(tmp_path / "metrics.json", payload).read_text(encoding="utf-8"))

    assert written == {"num_events": 7, "c_index": 0.5, "risk": [0, 1, 2]}


def test_write_json_still_rejects_unsupported_objects(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        write_json(tmp_path / "metrics.json", {"path": object()})


def test_write_csv_header_covers_every_key(tmp_path: Path) -> None:
    path = write_csv(tmp_path / "rows.csv", [{"a": 1}, {"a": 2, "b": 3}])

    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert rows == [{"a": "1", "b": ""}, {"a": "2", "b": "3"}]


def test_write_csv_rejects_a_key_outside_declared_fieldnames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not in fieldnames"):
        write_csv(tmp_path / "rows.csv", [{"a": 1, "b": 2}], fieldnames=["a"])
