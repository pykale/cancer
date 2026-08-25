"""The public HANCOCK head and neck cancer dataset.

Where the HANCOCK archives live and how they are laid out. The survival endpoint is
described in the experiment configuration; fetching, caching and member selection
are inherited from
:class:`~kalecancer.loaddata.dataset_access.ArchiveDataset`.

HANCOCK is published under CC BY 4.0 as ZIP archives that support HTTP range
requests, so the pipeline reads only the patients it needs.

    Dorner et al., A multimodal dataset for precision oncology in head and neck
    cancer, Nature Communications (2025). https://hancock.research.fau.eu/
"""

from __future__ import annotations

import re
from pathlib import Path

from kalecancer.loaddata.dataset_access import ArchiveDataset, DatasetAccessError, RemoteArchive

#: Slide stems such as ``PrimaryTumor_HE_036`` or ``PrimaryTumor_HE_036_a`` (a second
#: slide for the same patient). The ``patient_id`` group keeps its zero padding so it
#: matches the clinical records.
SLIDE_PATTERN = re.compile(r"^.+?_(?P<patient_id>\d+)(?:_[a-z])?$")

#: Feature directories inside the encodings archive, by anatomical region.
REGIONS = {"primary": "WSI_PrimaryTumor", "lymph_node": "WSI_LymphNode"}

CLINICAL_FILENAME = "clinical_data.json"

#: The published train/test assignments, one file per study question.
SPLIT_FILENAME = "dataset_split_in.json"


class HancockDataset(ArchiveDataset):
    """The HANCOCK archives, fetched into a local cache on first use."""

    name = "hancock"
    archives = {
        "uni_encodings": RemoteArchive(
            url="https://hancock.research.fau.eu/public/assets/WSI_UNI_encodings.zip",
            description="pre-extracted UNI patch features for primary tumour and lymph node slides",
        ),
        "structured": RemoteArchive(
            url="https://data.fau.de/public/24/87/322108724/StructuredData.zip",
            description="clinical, pathological and blood records",
        ),
        "splits": RemoteArchive(
            url="https://data.fau.de/public/24/87/322108724/DataSplits_DataDictionaries.zip",
            description="official train/test assignments and data dictionaries",
        ),
    }

    def features(self, region: str = "primary", patients: int = 0) -> Path:
        """Fetch patch features and return the directory to use as the feature root.

        The returned directory is the cache for this region, so it contains every
        patient fetched into it so far, not only the ones this call selected.
        ``patients`` therefore bounds the transfer rather than the cohort; pass a
        cache directory of its own to pin a run to a fixed set of patients.

        Args:
            region: Anatomical region, see :data:`REGIONS`.
            patients: Number of patients to fetch; 0 fetches the whole region.

        Raises:
            DatasetAccessError: If the region is unknown.
        """
        if region not in REGIONS:
            raise DatasetAccessError(f"unknown region {region!r}; available: {sorted(REGIONS)}")

        prefix = REGIONS[region]
        self.fetch_matching("uni_encodings", prefix=prefix, suffix=".h5", limit=patients, group_pattern=SLIDE_PATTERN)
        return self.root / prefix

    def clinical(self) -> Path:
        """Fetch the clinical records and return the path to the JSON file."""
        return self.fetch_named("structured", CLINICAL_FILENAME)

    def splits(self, filename: str = SPLIT_FILENAME) -> Path:
        """Fetch an official train/test assignment file.

        Using the published split rather than a fresh random one keeps results
        comparable with other work on this cohort.
        """
        return self.fetch_named("splits", filename)

    def paths(self, region: str = "primary", patients: int = 0) -> tuple[Path, Path]:
        """Fetch both parts, ready for :func:`~kalecancer.loaddata.cohort.build_cohort`."""
        return self.features(region=region, patients=patients), self.clinical()


def fetch_for(cfg) -> tuple[Path, Path]:
    """Fetch the region and patient count named in ``cfg``."""
    dataset = HancockDataset(cache_dir=cfg.DATASET.CACHE_DIR or None)
    return dataset.paths(region=cfg.DATASET.REGION, patients=cfg.DATASET.PATIENTS)
