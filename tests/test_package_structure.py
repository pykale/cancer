"""Smoke tests for the kalecancer package skeleton."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import kalecancer

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+[a-zA-Z0-9]*$")

SUBPACKAGES = (
    "auto",
    "loaddata",
    "prepdata",
    "model",
    "pipeline",
    "survival",
    "evaluate",
    "interpret",
    "utils",
)


def test_import_kalecancer_succeeds() -> None:
    assert kalecancer.__name__ == "kalecancer"


def test_version_is_non_empty_semver_like_string() -> None:
    version = kalecancer.__version__
    assert isinstance(version, str)
    assert version
    assert VERSION_PATTERN.match(version), f"unexpected version format: {version!r}"


@pytest.mark.parametrize("subpackage", SUBPACKAGES)
def test_subpackage_importable_via_attribute_access(subpackage: str) -> None:
    module = getattr(kalecancer, subpackage)
    assert module.__name__ == f"kalecancer.{subpackage}"


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match=r"has no attribute 'not_a_subpackage'"):
        _ = kalecancer.not_a_subpackage


def test_import_kalecancer_does_not_eagerly_import_torch_or_monai() -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), pythonpath]))

    script = """
import sys
import kalecancer

assert kalecancer.__version__

if "torch" in sys.modules:
    raise AssertionError("torch was eagerly imported")
if "monai" in sys.modules:
    raise AssertionError("monai was eagerly imported")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
