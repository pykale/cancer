"""Torch dataset over precomputed whole-slide patch features.

Rows of the cohort table are grouped so that one sample carries every slide of a
patient in a single bag, which lets the model emit one risk score per survival label.
Feature files are opened inside ``__getitem__`` so HDF5 handles are never shared
across DataLoader workers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from kalecancer.loaddata.wsi_feature_access import read_feature_bag

LABEL_KEYS = ("duration", "event")


class WSIFeatureDataset(Dataset[dict]):
    """Bags of precomputed patch features, one bag per group.

    Each sample is a dictionary:

    ==================  ======================================================
    ``features``        ``(num_patches, feature_dim)`` patch embeddings
    ``coords``          ``(num_patches, 2)`` patch coordinates
    ``slide_index``     ``(num_patches,)`` index into ``slide_ids``
    ``group_id``        Identifier the bag was grouped by
    ``slide_ids``       Slides contributing to the bag
    ``duration``        Time to event or censoring
    ``event``           ``1`` if observed, ``0`` if censored
    ==================  ======================================================

    ``features``, ``coords`` and ``slide_index`` share the patch dimension, so
    attention over ``features`` stays traceable to a coordinate and a source slide.

    Args:
        cohort: Table with ``slide_id``, ``path``, the label columns, and ``group_key``.
        group_key: Column whose rows are pooled into one bag.
        expected_dim: Feature dimension every file must provide, if it should be checked.
        max_patches: Cap on patches per bag, applied during training to bound memory.
            Leave ``None`` for evaluation and interpretation so attention covers whole
            slides.
        seed: Base seed for patch subsampling, combined with the sample index.
    """

    def __init__(
        self,
        cohort: pd.DataFrame,
        group_key: str = "patient_id",
        expected_dim: int | None = None,
        max_patches: int | None = None,
        seed: int = 0,
    ) -> None:
        if max_patches is not None and max_patches < 1:
            raise ValueError(f"max_patches must be a positive number of patches or None, got {max_patches}")
        missing = {group_key, "slide_id", "path", *LABEL_KEYS} - set(cohort.columns)
        if missing:
            raise KeyError(f"cohort is missing column(s) {sorted(missing)}")

        self.group_key = group_key
        self.expected_dim = expected_dim
        self.max_patches = max_patches
        self.seed = seed
        self.groups = [group for _, group in cohort.groupby(group_key, sort=True)]

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict:
        group = self.groups[index]

        feature_parts: list[np.ndarray] = []
        coord_parts: list[np.ndarray] = []
        slide_index_parts: list[np.ndarray] = []
        for position, (_, row) in enumerate(group.iterrows()):
            slide_features, slide_coords = read_feature_bag(row["path"], expected_dim=self.expected_dim)
            feature_parts.append(slide_features)
            coord_parts.append(slide_coords)
            slide_index_parts.append(np.full(len(slide_features), position, dtype=np.int64))

        features = np.concatenate(feature_parts)
        coords = np.concatenate(coord_parts)
        slide_index = np.concatenate(slide_index_parts)

        if self.max_patches is not None and len(features) > self.max_patches:
            generator = np.random.default_rng(self.seed + index)
            selection = np.sort(generator.choice(len(features), size=self.max_patches, replace=False))
            features, coords, slide_index = features[selection], coords[selection], slide_index[selection]

        first = group.iloc[0]
        return {
            "features": torch.from_numpy(features),
            "coords": torch.from_numpy(coords),
            "slide_index": torch.from_numpy(slide_index),
            "group_id": str(first[self.group_key]),
            "slide_ids": tuple(group["slide_id"]),
            "duration": torch.tensor(float(first["duration"]), dtype=torch.float32),
            "event": torch.tensor(float(first["event"]), dtype=torch.float32),
        }


def collate_bags(samples: list[dict]) -> dict:
    """Collate variable-length bags without padding.

    Bags stay in a list rather than a padded tensor, because padding to the largest
    bag wastes most of the tensor when sizes vary by orders of magnitude. Labels are
    stacked so a Cox risk set spans the batch.

    Raises:
        ValueError: If ``samples`` is empty.
    """
    if not samples:
        raise ValueError("cannot collate an empty batch")
    return {
        "samples": samples,
        "duration": torch.stack([sample["duration"] for sample in samples]),
        "event": torch.stack([sample["event"] for sample in samples]),
    }
