"""Every name in kalecancer.survival/.evaluate's __all__ must be importable."""

from __future__ import annotations

import importlib

import pytest

import kalecancer.evaluate
import kalecancer.survival

PACKAGES = (kalecancer.survival, kalecancer.evaluate)


@pytest.mark.parametrize("package", PACKAGES, ids=lambda pkg: pkg.__name__)
def test_all_is_defined_and_non_empty(package) -> None:
    assert hasattr(package, "__all__")
    assert len(package.__all__) > 0


@pytest.mark.parametrize("package", PACKAGES, ids=lambda pkg: pkg.__name__)
def test_all_is_sorted(package) -> None:
    assert package.__all__ == sorted(package.__all__)


@pytest.mark.parametrize("package", PACKAGES, ids=lambda pkg: pkg.__name__)
def test_every_name_in_all_is_importable_from_package_root(package) -> None:
    module = importlib.import_module(package.__name__)
    for name in package.__all__:
        assert hasattr(module, name), f"{package.__name__}.__all__ lists {name!r} but it is not importable"
