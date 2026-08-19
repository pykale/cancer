"""The contracts in :mod:`kalecancer.loaddata.protocols`.

Two things are being defended here. First, that the contract is *checked* -- a
Protocol on its own checks nothing at runtime, so ``check_target`` is the only
thing standing between a mis-shaped target and a confusing failure much later.
Second, that the protocols stay dependency-free, because ``SurvivalTarget`` is
expected to travel to PyKale core without dragging the data layer along.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from kalecancer.loaddata.protocols import Preprocessor, Target, check_target

PROTOCOLS_FILE = Path(__file__).resolve().parents[1] / "kalecancer" / "loaddata" / "protocols.py"


class GoodTarget:
    """The minimum that satisfies the contract."""

    required_columns = ("t", "e")

    def bind(self, identifiers, values) -> None:
        self.ids = list(identifiers)

    def for_(self, identifier):
        return {"t": torch.tensor(1.0)}

    def values_for(self, identifiers):
        return {"t": torch.ones(len(list(identifiers)))}


def test_a_conforming_object_passes():
    check_target(GoodTarget())


def test_survival_target_passes(make_target):
    check_target(make_target())


@pytest.mark.parametrize("dropped", ["required_columns", "bind", "for_", "values_for"])
def test_each_missing_member_is_named(dropped):
    """The message must say *what* is missing, not merely that something is."""
    namespace = {k: v for k, v in vars(GoodTarget).items() if k != dropped and not k.startswith("__")}
    partial = type("Partial", (), namespace)()

    with pytest.raises(TypeError) as excinfo:
        check_target(partial)
    assert dropped in str(excinfo.value)
    assert "Partial" in str(excinfo.value)


def test_the_message_points_at_the_protocol():
    with pytest.raises(TypeError, match="kalecancer.loaddata.protocols.Target"):
        check_target(object())


def test_protocols_are_not_runtime_checkable():
    """``runtime_checkable`` would only compare method *names*, never signatures.

    An object whose ``bind`` takes entirely the wrong arguments would pass it, so
    it reports confidence it cannot justify. ``check_target`` is the real check.
    """
    for protocol in (Target, Preprocessor):
        with pytest.raises(TypeError):
            isinstance(object(), protocol)


def test_protocols_import_nothing_from_kalecancer():
    """The leaf-of-the-import-graph property, asserted rather than hoped for.

    If this module ever imports from the package, ``survival/`` implementing
    ``Target`` acquires a dependency on the data layer, and ``SurvivalTarget``
    can no longer move to PyKale core on its own.
    """
    tree = ast.parse(PROTOCOLS_FILE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    offenders = sorted(name for name in imported if name.split(".")[0] == "kalecancer")
    assert offenders == [], f"protocols.py must import nothing from kalecancer, found {offenders}"
