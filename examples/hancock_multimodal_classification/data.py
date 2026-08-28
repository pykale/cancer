"""Patient records carrying a structured vector, a patch bag, or both.

Slide features are read from HDF5 once and held in memory for the whole run: the
comparison matrix trains the same patients dozens of times, and re-reading several
hundred megabytes per repeat would dominate everything else. The attention pooler is
learned, so the bags themselves must stay -- only a frozen pooler could be embedded
once up front.

The patch subsample is drawn once per patient and shared by every repeat, so the
repeats differ only in what the paper's repeats differ in: initialisation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from examples.hancock.dataset import SLIDE_PATTERN
from kalecancer.loaddata import ModalitySource
from kalecancer.loaddata.wsi_access import read_feature_bag, slide_table

logger = logging.getLogger(__name__)

TABULAR = "tabular"
IMAGING = "imaging"
LABEL = "label"


def slides_by_patient(feature_root: str | Path) -> dict[str, list[Path]]:
    """Every patient's slide feature files, keyed by patient id."""
    table = slide_table(feature_root, SLIDE_PATTERN)
    return {patient: list(rows["path"]) for patient, rows in table.groupby("patient_id")}


class BagCache(ModalitySource):
    """Patch bags held in memory, with the coordinates interpretation needs.

    A :class:`~kalecancer.loaddata.ModalitySource` that reads once and keeps
    everything resident, rather than the library's
    :class:`~kalecancer.loaddata.FeatureBagSource`, which re-reads per access. Worth
    it only because this experiment is a matrix: the same bags are iterated by three
    modality arms, two endpoints and three splits, so paying the read once and the
    memory throughout is the right trade here and would not be elsewhere.

    Args:
        slides: Feature files per patient.
        feature_dim: Width each bag must provide.
        max_patches: Cap per patient; 0 keeps the whole bag.
        seed: Seed for the subsample, fixed across repeats.
    """

    def __init__(self, slides: dict[str, list[Path]], feature_dim: int, max_patches: int = 0, seed: int = 0) -> None:
        self.feature_dim = feature_dim
        self.bags: dict[str, torch.Tensor] = {}
        self.coords: dict[str, np.ndarray] = {}
        self.slide_ids: dict[str, list[str]] = {}
        self.slide_index: dict[str, np.ndarray] = {}

        for position, (patient, paths) in enumerate(sorted(slides.items())):
            # A patient's slides are pooled into one bag, with each patch remembering
            # which slide it came from so attention stays traceable to a location.
            per_slide_features, per_slide_coords, per_slide_index = [], [], []
            for slide, path in enumerate(sorted(paths)):
                slide_features, slide_coords = read_feature_bag(path, expected_dim=feature_dim)
                per_slide_features.append(slide_features)
                per_slide_coords.append(slide_coords)
                per_slide_index.append(np.full(len(slide_features), slide))

            bag = np.concatenate(per_slide_features)
            coordinates = np.concatenate(per_slide_coords)
            index = np.concatenate(per_slide_index)

            if max_patches and len(bag) > max_patches:
                generator = np.random.default_rng(seed + position)
                keep = np.sort(generator.choice(len(bag), size=max_patches, replace=False))
                bag, coordinates, index = bag[keep], coordinates[keep], index[keep]

            # float16 halves the resident set; the encoder casts back on use.
            self.bags[patient] = torch.from_numpy(bag).half()
            self.coords[patient] = coordinates
            self.slide_index[patient] = index
            self.slide_ids[patient] = [Path(path).stem for path in sorted(paths)]

        megabytes = sum(bag.numel() * bag.element_size() for bag in self.bags.values()) / 1e6
        logger.info("cached %d patient bags, %.0f MB", len(self.bags), megabytes)

    def get(self, identifier: str, index: int) -> torch.Tensor | None:
        bag = self.bags.get(identifier)
        # Cached as float16 to halve the resident set; cast back on use.
        return None if bag is None else bag.float()

    def placeholder(self) -> torch.Tensor:
        return torch.zeros(1, self.feature_dim)

    def provenance(self, identifier: str, index: int) -> dict:
        """Where this patient's patches came from, for a batch's ``metadata``.

        Not model input: it is what lets an attention weight be placed back on a
        slide, which :func:`~kalecancer.interpret.batch_records` reads.
        """
        if identifier not in self.bags:
            return {}
        return {
            "slide_ids": self.slide_ids[identifier],
            "slide_index": torch.from_numpy(self.slide_index[identifier]),
            "coords": torch.from_numpy(self.coords[identifier]),
        }
