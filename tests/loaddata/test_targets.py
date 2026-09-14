"""The supervision contract, and the check that enforces it.

A ``Protocol`` checks nothing at runtime, so ``check_target`` is the only thing
standing between a mis-shaped target and a confusing failure much later -- when a
cohort asks for a value it cannot produce, halfway through training.
"""

from __future__ import annotations

import pytest
import torch

from kalecancer.loaddata.multimodal_access import Preprocessor, Target, check_target


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
    with pytest.raises(TypeError, match="kalecancer.loaddata.multimodal_access.Target"):
        check_target(object())


def test_protocols_are_not_runtime_checkable():
    """``runtime_checkable`` would only compare method *names*, never signatures.

    An object whose ``bind`` takes entirely the wrong arguments would pass it, so
    it reports confidence it cannot justify. ``check_target`` is the real check.
    """
    for protocol in (Target, Preprocessor):
        with pytest.raises(TypeError):
            isinstance(object(), protocol)
