"""Fetching the public HANCOCK head and neck cancer dataset.

HANCOCK is published under CC BY 4.0 and distributed as ZIP archives that support
HTTP range requests, so the pipeline reads the patients it needs directly from the
source rather than relying on a copy.

    Dorner et al., A multimodal dataset for precision oncology in head and neck
    cancer, Nature Communications (2025). https://hancock.research.fau.eu/
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from kalecancer.loaddata.sources import extract_members, open_remote_zip
from kalecancer.loaddata.wsi_feature_access import DEFAULT_SLIDE_PATTERN

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "kalecancer"


@dataclass(frozen=True)
class Archive:
    """A downloadable dataset archive."""

    url: str
    description: str


ARCHIVES: dict[str, Archive] = {
    "uni_encodings": Archive(
        url="https://hancock.research.fau.eu/public/assets/WSI_UNI_encodings.zip",
        description="pre-extracted UNI patch features for primary tumour and lymph node slides",
    ),
    "structured": Archive(
        url="https://data.fau.de/public/24/87/322108724/StructuredData.zip",
        description="clinical, pathological and blood records",
    ),
}

#: Feature directories inside the encodings archive, by anatomical region.
REGIONS = {"primary": "WSI_PrimaryTumor", "lymph_node": "WSI_LymphNode"}

CLINICAL_FILENAME = "clinical_data.json"


class HancockError(RuntimeError):
    """Raised when the HANCOCK dataset cannot be prepared."""


def _select_patients(members: list[str], limit: int, pattern: re.Pattern[str]) -> list[str]:
    """Choose members for the first ``limit`` patient identifiers.

    Selection is by sorted patient id so a given limit always yields the same cohort.
    All slides of a selected patient are kept together.
    """
    by_patient: dict[str, list[str]] = {}
    for member in members:
        match = pattern.match(Path(member).stem)
        if match:
            by_patient.setdefault(match.group("patient_id"), []).append(member)

    chosen = sorted(by_patient)[:limit]
    return [member for patient in chosen for member in sorted(by_patient[patient])]


def fetch_features(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    region: str = "primary",
    patients: int = 0,
    slide_pattern: re.Pattern[str] = DEFAULT_SLIDE_PATTERN,
) -> Path:
    """Fetch patch features into a local cache and return their root directory.

    Args:
        cache_dir: Directory holding the cached dataset.
        region: Anatomical region, see :data:`REGIONS`.
        patients: Number of patients to fetch; 0 fetches every patient in the region.
        slide_pattern: Pattern used to read patient ids from member names.

    Returns:
        The directory to pass as the feature root.

    Raises:
        HancockError: If the region is unknown or the archive holds no matching files.
    """
    if region not in REGIONS:
        raise HancockError(f"unknown region {region!r}; available: {sorted(REGIONS)}")

    root = Path(cache_dir) / "hancock"
    prefix = REGIONS[region]

    archive, handle = open_remote_zip(ARCHIVES["uni_encodings"].url)
    with archive:
        members = [name for name in archive.namelist() if name.startswith(prefix) and name.endswith(".h5")]
        if not members:
            raise HancockError(f"no .h5 files for region {region!r} in {ARCHIVES['uni_encodings'].url}")

        selected = _select_patients(members, patients, slide_pattern) if patients else sorted(members)
        logger.info(
            "HANCOCK %s: %d of %d slides (index read using %.1f MB)",
            region,
            len(selected),
            len(members),
            handle.bytes_fetched / 1e6,
        )
        extract_members(archive, selected, root)

    logger.info("fetched %.1f MB of the %.2f GB archive", handle.bytes_fetched / 1e6, handle.size / 1e9)
    return root / prefix


def fetch_clinical(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> Path:
    """Fetch the clinical records and return the path to ``clinical_data.json``.

    Raises:
        HancockError: If the archive does not contain the clinical records.
    """
    root = Path(cache_dir) / "hancock"

    archive, _ = open_remote_zip(ARCHIVES["structured"].url)
    with archive:
        members = [name for name in archive.namelist() if Path(name).name == CLINICAL_FILENAME]
        if not members:
            raise HancockError(f"{CLINICAL_FILENAME} not found in {ARCHIVES['structured'].url}")
        return extract_members(archive, members[:1], root)[0]


def fetch_dataset(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    region: str = "primary",
    patients: int = 0,
) -> tuple[Path, Path]:
    """Fetch features and clinical records together.

    Returns:
        ``(feature_root, clinical_path)`` for use with
        :func:`~kalecancer.loaddata.cohort.build_cohort`.
    """
    return fetch_features(cache_dir, region=region, patients=patients), fetch_clinical(cache_dir)


def resolve_dataset(cfg) -> tuple[Path, Path]:
    """Resolve the configured data source to local paths.

    ``DATASET.SOURCE`` selects between an existing local copy and fetching from
    HANCOCK. Both return local paths, so the rest of the pipeline is unaffected.

    Raises:
        HancockError: If the source is unknown.
    """
    source = cfg.DATASET.SOURCE
    if source == "local":
        return Path(cfg.DATASET.FEATURE_ROOT), Path(cfg.DATASET.CLINICAL_PATH)
    if source == "hancock":
        return fetch_dataset(
            cache_dir=cfg.DATASET.CACHE_DIR or DEFAULT_CACHE_DIR,
            region=cfg.DATASET.REGION,
            patients=cfg.DATASET.PATIENTS,
        )
    raise HancockError(f"unknown DATASET.SOURCE {source!r}; expected 'local' or 'hancock'")
