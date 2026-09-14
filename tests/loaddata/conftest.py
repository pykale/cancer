"""Fixtures for the data-access tests."""

from __future__ import annotations

import pytest

from kalecancer.loaddata import SurvivalTarget


@pytest.fixture
def make_target():
    """Factory for a ``SurvivalTarget``, the real implementation of ``Target``.

    A factory rather than a single instance: ``bind`` stores state on the target, so
    two cohorts sharing one instance would fight over it.
    """

    def build(**overrides) -> SurvivalTarget:
        kwargs = {"time": "time", "event": "status", "event_value": "dead"}
        kwargs.update(overrides)
        return SurvivalTarget(**kwargs)

    return build
