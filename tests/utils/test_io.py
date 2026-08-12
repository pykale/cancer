"""Tests for artefact writing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

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
