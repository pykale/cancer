"""Tests for HANCOCK dataset fetching, against a stubbed archive."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from kalecancer.loaddata import hancock
from kalecancer.loaddata.hancock import HancockError, fetch_dataset, resolve_dataset

PATIENTS = ["001", "002", "003", "004"]


def build_archives(tmp_path: Path) -> tuple[Path, Path]:
    """A stand-in for the two published archives, with the same internal layout."""
    encodings = tmp_path / "encodings.zip"
    with zipfile.ZipFile(encodings, "w") as archive:
        for index, patient in enumerate(PATIENTS):
            subsite = "Larynx" if index % 2 else "OralCavity"
            for region in ("WSI_PrimaryTumor", "WSI_LymphNode"):
                prefix = "PrimaryTumor" if region == "WSI_PrimaryTumor" else "LymphNode"
                archive.writestr(
                    f"{region}/{region}_{subsite}/h5_files/{prefix}_HE_{patient}.h5",
                    np.zeros(64, dtype=np.uint8).tobytes(),
                )
        # A second slide for one patient, as the real archive contains.
        archive.writestr(
            "WSI_PrimaryTumor/WSI_PrimaryTumor_Larynx/h5_files/PrimaryTumor_HE_002_a.h5",
            np.zeros(64, dtype=np.uint8).tobytes(),
        )

    structured = tmp_path / "structured.zip"
    with zipfile.ZipFile(structured, "w") as archive:
        archive.writestr(
            "StructuredData/clinical_data.json",
            json.dumps(
                [{"patient_id": p, "days_to_last_information": 100, "survival_status": "living"} for p in PATIENTS]
            ),
        )
        archive.writestr("StructuredData/blood_data.json", "[]")
    return encodings, structured


@pytest.fixture
def local_archives(tmp_path: Path, monkeypatch):
    """Point the archive registry at local ZIPs opened without HTTP."""
    encodings, structured = build_archives(tmp_path)

    def fake_open(url: str, timeout: float = 120.0):
        path = encodings if "encodings" in url else structured
        archive = zipfile.ZipFile(path)
        handle = type("Handle", (), {"bytes_fetched": 0, "size": path.stat().st_size})()
        return archive, handle

    monkeypatch.setattr(hancock, "open_remote_zip", fake_open)
    monkeypatch.setitem(hancock.ARCHIVES, "uni_encodings", hancock.Archive("http://stub/encodings.zip", ""))
    monkeypatch.setitem(hancock.ARCHIVES, "structured", hancock.Archive("http://stub/structured.zip", ""))
    return tmp_path / "cache"


def test_fetches_features_and_clinical_records(local_archives: Path) -> None:
    feature_root, clinical_path = fetch_dataset(cache_dir=local_archives)

    assert feature_root.is_dir()
    assert clinical_path.name == "clinical_data.json"
    assert len(json.loads(clinical_path.read_text())) == len(PATIENTS)


def test_fetches_only_the_requested_region(local_archives: Path) -> None:
    feature_root, _ = fetch_dataset(cache_dir=local_archives, region="primary")

    assert feature_root.name == "WSI_PrimaryTumor"
    assert not (local_archives / "hancock" / "WSI_LymphNode").exists()


def test_lymph_node_region_is_available(local_archives: Path) -> None:
    feature_root, _ = fetch_dataset(cache_dir=local_archives, region="lymph_node")

    assert feature_root.name == "WSI_LymphNode"


def test_patient_limit_restricts_the_cohort(local_archives: Path) -> None:
    feature_root, _ = fetch_dataset(cache_dir=local_archives, patients=2)

    fetched = sorted(p.stem for p in feature_root.rglob("*.h5"))
    assert fetched == ["PrimaryTumor_HE_001", "PrimaryTumor_HE_002", "PrimaryTumor_HE_002_a"]


def test_patient_selection_is_deterministic(tmp_path: Path, local_archives: Path) -> None:
    first, _ = fetch_dataset(cache_dir=local_archives, patients=2)
    names = sorted(p.name for p in first.rglob("*.h5"))

    second, _ = fetch_dataset(cache_dir=tmp_path / "other", patients=2)

    assert sorted(p.name for p in second.rglob("*.h5")) == names


def test_a_patient_keeps_all_of_their_slides(local_archives: Path) -> None:
    """Patient 002 has two slides; a limit must not split them."""
    feature_root, _ = fetch_dataset(cache_dir=local_archives, patients=2)

    assert len(list(feature_root.rglob("PrimaryTumor_HE_002*.h5"))) == 2


def test_zero_patients_fetches_the_whole_region(local_archives: Path) -> None:
    feature_root, _ = fetch_dataset(cache_dir=local_archives, patients=0)

    assert len(list(feature_root.rglob("*.h5"))) == len(PATIENTS) + 1


def test_unknown_region_is_rejected(local_archives: Path) -> None:
    with pytest.raises(HancockError, match="unknown region"):
        fetch_dataset(cache_dir=local_archives, region="brain")


def test_local_source_returns_the_configured_paths() -> None:
    from kalecancer.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.DATASET.SOURCE = "local"
    cfg.DATASET.FEATURE_ROOT = "features"
    cfg.DATASET.CLINICAL_PATH = "clinical.json"

    assert resolve_dataset(cfg) == (Path("features"), Path("clinical.json"))


def test_hancock_source_fetches(local_archives: Path) -> None:
    from kalecancer.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.DATASET.SOURCE = "hancock"
    cfg.DATASET.CACHE_DIR = str(local_archives)
    cfg.DATASET.PATIENTS = 1

    feature_root, clinical_path = resolve_dataset(cfg)

    assert feature_root.exists() and clinical_path.exists()


def test_unknown_source_is_rejected() -> None:
    from kalecancer.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.DATASET.SOURCE = "s3"

    with pytest.raises(HancockError, match="unknown DATASET.SOURCE"):
        resolve_dataset(cfg)
