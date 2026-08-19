"""Tests for resolving the configured data source."""

from __future__ import annotations

from pathlib import Path

import pytest

from kalecancer.config import get_cfg_defaults
from kalecancer.loaddata.dataset_source import SOURCES, DataSourceError, resolve_dataset
from kalecancer.loaddata.synthetic import write_synthetic_cohort


def make_cfg(tmp_path: Path, source: str, **dataset):
    cfg = get_cfg_defaults()
    cfg.DATASET.SOURCE = source
    cfg.DATASET.CACHE_DIR = str(tmp_path)
    cfg.MODEL.INPUT_DIM = 8
    for key, value in dataset.items():
        setattr(cfg.DATASET, key, value)
    return cfg


def test_local_source_passes_the_configured_paths_through(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, "local", FEATURE_ROOT="features", CLINICAL_PATH="clinical.json")

    assert resolve_dataset(cfg) == (Path("features"), Path("clinical.json"))


def test_synthetic_source_writes_a_usable_cohort(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, "synthetic", PATIENTS=12)

    feature_root, clinical_path = resolve_dataset(cfg)

    assert list(feature_root.glob("*.h5"))
    assert clinical_path.exists()


def test_unknown_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataSourceError, match="unknown DATASET.SOURCE"):
        resolve_dataset(make_cfg(tmp_path, "elsewhere"))


def test_documented_sources_are_the_supported_ones() -> None:
    assert SOURCES == ("local", "hancock", "synthetic")


def test_generated_cohort_matches_end_to_end(tmp_path: Path) -> None:
    """The generated files must satisfy the same loaders the real data goes through."""
    from kalecancer.evaluate import cohort_summary
    from kalecancer.loaddata import build_cohort

    feature_root, clinical_path = write_synthetic_cohort(tmp_path, num_patients=20, feature_dim=8, seed=0)
    cohort = build_cohort(feature_root, clinical_path, expected_dim=8)

    summary = cohort_summary(cohort)
    assert summary["num_matched_groups"] == 20
    assert 0 < summary["num_events"] < 20


def test_every_generated_patient_has_a_usable_label(tmp_path: Path) -> None:
    """Durations must be positive, or the loader would silently drop patients."""
    from kalecancer.loaddata import build_cohort

    feature_root, clinical_path = write_synthetic_cohort(tmp_path, num_patients=40, feature_dim=8, seed=3)
    cohort = build_cohort(feature_root, clinical_path, expected_dim=8)

    assert cohort.attrs["clinical_exclusions"] == {}
    assert (cohort["duration"] > 0).all()


def test_multi_slide_patients_are_generated(tmp_path: Path) -> None:
    from kalecancer.evaluate import cohort_summary
    from kalecancer.loaddata import build_cohort

    feature_root, clinical_path = write_synthetic_cohort(
        tmp_path, num_patients=16, feature_dim=8, multi_slide_every=4, seed=0
    )
    cohort = build_cohort(feature_root, clinical_path, expected_dim=8)

    assert cohort_summary(cohort)["groups_with_multiple_slides"]


def test_generation_is_reproducible(tmp_path: Path) -> None:
    first = write_synthetic_cohort(tmp_path / "a", num_patients=8, feature_dim=8, seed=1)[1]
    second = write_synthetic_cohort(tmp_path / "b", num_patients=8, feature_dim=8, seed=1)[1]

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_feature_dimension_is_honoured(tmp_path: Path) -> None:
    from kalecancer.loaddata import inspect_feature_bag

    feature_root, _ = write_synthetic_cohort(tmp_path, num_patients=4, feature_dim=32, seed=0)

    assert inspect_feature_bag(next(feature_root.glob("*.h5")))[1] == 32
