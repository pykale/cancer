"""Torch dataset over precomputed whole-slide patch bags.

One sample is one *patient*, not one slide: a patient's slides are pooled into a
single bag so the model emits exactly one risk score per survival label. Bags are
read lazily inside ``__getitem__`` so HDF5 handles are never shared across
DataLoader workers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from kalecancer.loaddata.cohort import PatientBag
from kalecancer.loaddata.wsi_feature_access import read_feature_bag


@dataclass
class BagSample:
    """One patient's pooled patch bag and survival label.

    ``features``, ``coords`` and ``slide_index`` share the patch dimension, so
    attention over ``features`` stays traceable to a coordinate and a source slide.
    """

    patient_id: str
    slide_ids: tuple[str, ...]
    features: torch.Tensor
    coords: torch.Tensor
    slide_index: torch.Tensor
    duration: torch.Tensor
    event: torch.Tensor


@dataclass
class BagBatch:
    """A mini-batch of variable-length bags.

    Bags stay in a list rather than a padded tensor: padding a batch to the largest
    bag would waste most of the tensor when sizes vary by two orders of magnitude.
    ``duration`` and ``event`` are stacked so the Cox risk set spans the batch.
    """

    samples: list[BagSample]
    duration: torch.Tensor
    event: torch.Tensor

    def __len__(self) -> int:
        return len(self.samples)


class WSIFeatureBagDataset(Dataset[BagSample]):
    """Patient-level bags of precomputed WSI patch features.

    Args:
        bags: Matched patients, e.g. from :func:`~kalecancer.loaddata.cohort.build_cohort`.
        expected_dim: Feature dimension every bag must provide, if it should be checked.
        max_patches: Cap on patches per bag. Applied during training only, to bound
            memory; leave ``None`` for evaluation and interpretation so attention
            covers the whole slide.
        seed: Base seed for patch subsampling. Combined with the sample index so
            subsampling is reproducible and differs between patients.
    """

    def __init__(
        self,
        bags: list[PatientBag],
        expected_dim: int | None = None,
        max_patches: int | None = None,
        seed: int = 0,
    ) -> None:
        self.bags = list(bags)
        self.expected_dim = expected_dim
        self.max_patches = max_patches
        self.seed = seed

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, index: int) -> BagSample:
        bag = self.bags[index]

        features_per_slide = []
        coords_per_slide = []
        slide_index_per_slide = []
        for slide_position, slide in enumerate(bag.slides):
            features, coords = read_feature_bag(slide.path, expected_dim=self.expected_dim)
            features_per_slide.append(features)
            coords_per_slide.append(coords)
            slide_index_per_slide.append(np.full(len(features), slide_position, dtype=np.int64))

        features = np.concatenate(features_per_slide)
        coords = np.concatenate(coords_per_slide)
        slide_index = np.concatenate(slide_index_per_slide)

        if self.max_patches is not None and len(features) > self.max_patches:
            generator = np.random.default_rng(self.seed + index)
            selection = np.sort(generator.choice(len(features), size=self.max_patches, replace=False))
            features, coords, slide_index = features[selection], coords[selection], slide_index[selection]

        return BagSample(
            patient_id=bag.patient_id,
            slide_ids=tuple(slide.slide_id for slide in bag.slides),
            features=torch.from_numpy(features),
            coords=torch.from_numpy(coords),
            slide_index=torch.from_numpy(slide_index),
            duration=torch.tensor(bag.duration, dtype=torch.float32),
            event=torch.tensor(bag.event, dtype=torch.float32),
        )


def collate_bags(samples: list[BagSample]) -> BagBatch:
    """Collate variable-length bags without padding."""
    return BagBatch(
        samples=samples,
        duration=torch.stack([sample.duration for sample in samples]),
        event=torch.stack([sample.event for sample in samples]),
    )
