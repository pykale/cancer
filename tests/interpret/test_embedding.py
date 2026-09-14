"""Tests for low-dimensional projection of patient representations."""

from __future__ import annotations

import numpy as np
import pytest

from kalecancer.interpret.embedding import umap_embedding

umap = pytest.importorskip("umap", reason="umap-learn ships in the interpret extra")


def test_projection_keeps_one_row_per_patient() -> None:
    features = np.random.default_rng(0).normal(size=(40, 12))

    coordinates = umap_embedding(features, n_components=2, random_state=0)

    assert coordinates.shape == (40, 2)


def test_fitting_on_a_subset_still_projects_everything() -> None:
    """Fitting on the training rows is what keeps a projection honest about held-out data."""
    generator = np.random.default_rng(0)
    features, train = generator.normal(size=(40, 12)), generator.normal(size=(25, 12))

    coordinates = umap_embedding(features, random_state=0, fit_on=train)

    assert coordinates.shape == (40, 2)


def test_a_flat_array_is_refused() -> None:
    with pytest.raises(ValueError, match=r"\(N, D\)"):
        umap_embedding(np.zeros(10))
