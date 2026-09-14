"""Low-dimensional projection of patient representations.

A multimodal patient vector has too many dimensions to inspect directly, so
projecting a cohort into two dimensions is how its structure becomes visible: whether
patients sharing a characteristic cluster, and how a train/test split sits inside the
distribution it was drawn from.

Coordinates are returned rather than plots, so the caller chooses how to render them.
This matches :mod:`kalecancer.interpret.attention`, which exports weights rather than
heatmaps.
"""

from __future__ import annotations

import numpy as np


def _require_umap():
    """Import umap-learn, explaining the install if it is absent."""
    try:
        import umap
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "umap_embedding needs umap-learn, which is not installed. "
            'Install it with: pip install "kalecancer[interpret]"'
        ) from error
    return umap


def umap_embedding(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int | None = None,
    fit_on: np.ndarray | None = None,
) -> np.ndarray:
    """Project feature vectors into a low-dimensional space with UMAP.

    Args:
        features: ``(N, D)`` vectors to project.
        n_components: Dimensions to project into.
        n_neighbors: Neighbourhood size; larger values favour global structure.
        min_dist: Minimum separation between points in the projection.
        metric: Distance used in the original space.
        random_state: Seed for a reproducible layout. Setting it forces
            single-threaded execution, and the layout is still not guaranteed
            identical across umap-learn versions.
        fit_on: Optional ``(M, D)`` subset to fit the projection on before
            transforming ``features``. Pass the training rows when the projection
            illustrates a model's view of held-out patients; leave it unset when the
            projection is a description of the whole cohort.

    Returns:
        ``(N, n_components)`` coordinates.

    Raises:
        ImportError: If umap-learn is not installed.
        ValueError: If ``features`` is not two-dimensional.
    """
    umap = _require_umap()

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"features must have shape (N, D), got {features.shape}")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, max(2, len(features) - 1)),
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    if fit_on is None:
        return np.asarray(reducer.fit_transform(features))
    return np.asarray(reducer.fit(np.asarray(fit_on, dtype=np.float64)).transform(features))
