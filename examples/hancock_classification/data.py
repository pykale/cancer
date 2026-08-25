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
from torch.utils.data import Dataset

from kalecancer.loaddata.sample import PatientBatch
from kalecancer.loaddata.wsi_feature_access import read_feature_bag, slide_table

logger = logging.getLogger(__name__)

TABULAR = "tabular"
IMAGING = "imaging"
LABEL = "label"


def slides_by_patient(feature_root: str | Path) -> dict[str, list[Path]]:
    """Every patient's slide feature files, keyed by patient id."""
    table = slide_table(feature_root)
    return {patient: list(rows["path"]) for patient, rows in table.groupby("patient_id")}


class BagCache:
    """Patch bags held in memory, with the coordinates interpretation needs.

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
            features, coords, index = [], [], []
            for slide, path in enumerate(sorted(paths)):
                bag, coordinates = read_feature_bag(path, expected_dim=feature_dim)
                features.append(bag)
                coords.append(coordinates)
                index.append(np.full(len(bag), slide))

            bag = np.concatenate(features)
            coordinates = np.concatenate(coords)
            index = np.concatenate(index)

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

    def sample_for(self, patient: str) -> dict:
        """The record :func:`~kalecancer.interpret.attention_records` expects."""
        return {
            "group_id": patient,
            "slide_ids": self.slide_ids[patient],
            "slide_index": torch.from_numpy(self.slide_index[patient]),
            "coords": torch.from_numpy(self.coords[patient]),
        }


class PatientRecords(Dataset):
    """One record per patient, carrying whichever modalities are in play.

    Args:
        patient_ids: Patients in this split.
        modalities: Modality names to emit.
        tabular: Structured vector per patient.
        bags: Cached patch bags, or ``None`` when imaging is not used.
        labels: Binary label per patient.
        feature_dim: Width of a patch feature, for the absent-imaging placeholder.
    """

    def __init__(
        self,
        patient_ids: list[str],
        modalities: list[str],
        tabular: dict[str, torch.Tensor],
        bags: BagCache | None,
        labels: dict[str, int],
        feature_dim: int = 1024,
    ) -> None:
        self.patient_ids = list(patient_ids)
        self.modalities = list(modalities)
        self.tabular = tabular
        self.bags = bags
        self.labels = labels
        self.feature_dim = feature_dim
        self.tabular_dim = len(next(iter(tabular.values()))) if tabular else 0

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict:
        patient = self.patient_ids[index]
        modalities: dict[str, torch.Tensor] = {}
        present: dict[str, torch.Tensor] = {}

        if TABULAR in self.modalities:
            vector = self.tabular.get(patient)
            modalities[TABULAR] = vector if vector is not None else torch.zeros(self.tabular_dim)
            present[TABULAR] = torch.tensor(vector is not None)

        if IMAGING in self.modalities:
            bag = self.bags.bags.get(patient) if self.bags else None
            # A placeholder keeps the batch rectangular; ``present`` is what the model
            # reads, so the zeros are never treated as evidence.
            modalities[IMAGING] = bag.float() if bag is not None else torch.zeros(1, self.feature_dim)
            present[IMAGING] = torch.tensor(bag is not None)

        return {
            "patient_id": patient,
            "modalities": modalities,
            "present": present,
            "target": {LABEL: torch.tensor(float(self.labels[patient]))},
        }


def collate(samples: list[dict]) -> PatientBatch:
    """Collate records, leaving ragged patch bags as a list.

    Padding every bag to the batch maximum would waste most of the tensor when bag
    sizes differ by an order of magnitude, and the attention pooler consumes a list
    directly.

    Raises:
        ValueError: If ``samples`` is empty.
    """
    if not samples:
        raise ValueError("cannot collate an empty batch")

    names = list(samples[0]["modalities"])
    modalities = {}
    for name in names:
        values = [sample["modalities"][name] for sample in samples]
        modalities[name] = values if values[0].dim() > 1 else torch.stack(values)

    return PatientBatch(
        patient_id=[sample["patient_id"] for sample in samples],
        modalities=modalities,
        present={name: torch.stack([sample["present"][name] for sample in samples]) for name in names},
        target={LABEL: torch.stack([sample["target"][LABEL] for sample in samples])},
    )
