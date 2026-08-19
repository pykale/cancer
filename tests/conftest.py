"""Synthetic fixtures so tests never need the private WSI cohort."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

FEATURE_DIM = 8


def write_bag(path: Path, num_patches: int = 12, feature_dim: int = FEATURE_DIM, seed: int = 0) -> Path:
    """Write a valid HDF5 patch bag with coordinates on a 2048-pixel grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("features", data=generator.normal(size=(num_patches, feature_dim)).astype(np.float32))
        handle.create_dataset(
            "coords", data=(np.arange(num_patches * 2).reshape(num_patches, 2) * 2048).astype(np.int32)
        )
    return path


@pytest.fixture
def feature_root(tmp_path: Path) -> Path:
    """Two subsite folders; patient 003 has a second slide, mirroring the real cohort."""
    root = tmp_path / "features"
    write_bag(root / "Larynx" / "h5_files" / "PrimaryTumor_HE_001.h5", num_patches=10, seed=1)
    write_bag(root / "Larynx" / "h5_files" / "PrimaryTumor_HE_002.h5", num_patches=15, seed=2)
    write_bag(root / "OralCavity" / "h5_files" / "PrimaryTumor_HE_003.h5", num_patches=7, seed=3)
    write_bag(root / "OralCavity" / "h5_files" / "PrimaryTumor_HE_003_a.h5", num_patches=5, seed=4)
    return root


@pytest.fixture
def clinical_path(tmp_path: Path) -> Path:
    """Clinical records covering the feature cohort plus one patient without slides."""
    records = [
        {"patient_id": "001", "days_to_last_information": 500, "survival_status": "deceased"},
        {"patient_id": "002", "days_to_last_information": 1200, "survival_status": "living"},
        {"patient_id": "003", "days_to_last_information": 900, "survival_status": "deceased"},
        {"patient_id": "004", "days_to_last_information": 300, "survival_status": "living"},
    ]
    path = tmp_path / "clinical_data.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


@pytest.fixture
def cohort(feature_root: Path, clinical_path: Path):
    from kalecancer.loaddata import build_cohort

    return build_cohort(feature_root, clinical_path, expected_dim=FEATURE_DIM)
