"""Tests for rasterising attention into a spatial heatmap.

Every test for :func:`attention_heatmap` and :func:`save_heatmap` runs without openslide installed -- that is the
whole point of keeping rasterisation (pure numpy) separate from the slide overlay (needs openslide). The two
overlay tests fake the ``openslide`` module via ``monkeypatch`` so they exercise the real logic (the ImportError
message, the shape check) regardless of whether the ``pathology`` extra happens to be installed.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalecancer.interpret.heatmap import attention_heatmap, overlay_on_slide, save_heatmap


def _instances(coords: list[tuple[int, int]], weights: list[float], patient_id: str | None = "p0") -> pd.DataFrame:
    frame = pd.DataFrame(coords, columns=["x", "y"]).assign(attention=weights)
    if patient_id is not None:
        frame = frame.assign(patient_id=patient_id)
    return frame


def test_single_patch_lands_in_the_expected_cell() -> None:
    instances = _instances([(64, 96)], [0.42])

    heatmap = attention_heatmap(instances, patch_size=32, downsample=32, canvas_shape=(256, 256), normalise=False)

    assert heatmap.shape == (8, 8)
    nonnan = np.argwhere(~np.isnan(heatmap))
    assert nonnan.tolist() == [[3, 2]]  # row = y // downsample, col = x // downsample
    assert heatmap[3, 2] == pytest.approx(0.42)


@pytest.mark.parametrize(
    ("aggregate", "expected"),
    [("mean", 0.5), ("sum", 1.0), ("max", 0.7)],
)
def test_overlapping_patches_aggregate_as_requested(aggregate: str, expected: float) -> None:
    instances = _instances([(0, 0), (0, 0)], [0.3, 0.7])

    heatmap = attention_heatmap(instances, patch_size=32, downsample=32, aggregate=aggregate, normalise=False)

    assert heatmap[0, 0] == pytest.approx(expected)


def test_regions_with_no_patch_are_nan_not_zero() -> None:
    instances = _instances([(0, 0)], [0.9])

    heatmap = attention_heatmap(instances, patch_size=32, downsample=32, canvas_shape=(128, 128), normalise=False)

    assert np.isnan(heatmap).sum() == heatmap.size - 1
    assert heatmap[0, 0] == pytest.approx(0.9)


def test_normalise_maps_the_max_to_one_and_ignores_nan() -> None:
    instances = _instances([(0, 0), (256, 0)], [2.0, 8.0])

    heatmap = attention_heatmap(instances, patch_size=256, downsample=256, canvas_shape=(256, 512), normalise=True)

    assert np.nanmax(heatmap) == pytest.approx(1.0)
    assert np.nanmin(heatmap) == pytest.approx(0.0)
    assert not np.isnan(heatmap).any()  # this canvas is fully covered by the two patches


def test_canvas_shape_given_vs_inferred_produce_correspondingly_sized_arrays() -> None:
    instances = _instances([(0, 0), (256, 0)], [1.0, 2.0])

    inferred = attention_heatmap(instances, patch_size=256, downsample=256, canvas_shape=None)
    explicit = attention_heatmap(instances, patch_size=256, downsample=256, canvas_shape=(2048, 2048))

    assert inferred.shape == (1, 2)  # bounding box: one patch-height tall, two wide
    assert explicit.shape == (8, 8)  # 2048 / 256


def test_coordinate_convention_k_patches_apart_land_k_cells_apart() -> None:
    patch_size = 256
    instances = _instances([(0, 0), (patch_size * 1, 0), (patch_size * 3, 0)], [1.0, 2.0, 3.0])

    heatmap = attention_heatmap(instances, patch_size=patch_size, downsample=patch_size, normalise=False)

    assert heatmap.shape == (1, 4)
    assert heatmap[0, 0] == pytest.approx(1.0)
    assert heatmap[0, 1] == pytest.approx(2.0)
    assert np.isnan(heatmap[0, 2])  # no patch 2 cells from the first -- k=2 is skipped
    assert heatmap[0, 3] == pytest.approx(3.0)  # k=3 cells from the first patch


def test_a_frame_with_two_patients_is_refused() -> None:
    instances = pd.concat([_instances([(0, 0)], [1.0], "p0"), _instances([(256, 0)], [2.0], "p1")])

    with pytest.raises(ValueError, match="p0"):
        attention_heatmap(instances, patch_size=256, downsample=256)


def test_save_heatmap_writes_a_readable_png_of_the_right_dimensions(tmp_path: Path) -> None:
    heatmap = np.array([[0.0, 0.5, np.nan], [1.0, np.nan, 0.25]])
    out_path = tmp_path / "heatmap.png"

    save_heatmap(heatmap, out_path)

    data = out_path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, bit_depth, colour_type = struct.unpack(">IIBB", data[16:26])
    assert (width, height) == (heatmap.shape[1], heatmap.shape[0])
    assert bit_depth == 8
    assert colour_type == 2  # RGB

    pil = pytest.importorskip("PIL.Image")
    image = pil.open(out_path)
    assert image.size == (heatmap.shape[1], heatmap.shape[0])


def test_save_heatmap_works_without_matplotlib(tmp_path: Path, monkeypatch) -> None:
    """The built-in fallback colormap must not need matplotlib at all."""
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    heatmap = np.array([[0.0, 1.0]])

    save_heatmap(heatmap, tmp_path / "fallback.png")

    assert (tmp_path / "fallback.png").exists()


def test_overlay_without_openslide_raises_import_error_naming_the_extra(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openslide", None)

    with pytest.raises(ImportError, match="pathology"):
        overlay_on_slide(np.zeros((2, 2)), "fake_slide.svs")


def test_overlay_shape_mismatch_raises_value_error_naming_both_shapes(monkeypatch) -> None:
    pil_image = pytest.importorskip("PIL.Image")

    class _FakeSlide:
        dimensions = (256, 256)  # (width, height); downsample=32 -> thumbnail (8, 8)
        level_dimensions = [(256, 256)]

        def get_best_level_for_downsample(self, downsample: int) -> int:
            return 0

        def read_region(self, location, level, size):
            return pil_image.new("RGB", size, color=(10, 20, 30))

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

    class _FakeOpenslideModule:
        @staticmethod
        def OpenSlide(path):
            return _FakeSlide()

    monkeypatch.setitem(sys.modules, "openslide", _FakeOpenslideModule())

    mismatched_heatmap = np.full((5, 5), np.nan)  # thumbnail will be (8, 8), not (5, 5)

    with pytest.raises(ValueError) as exc_info:
        overlay_on_slide(mismatched_heatmap, "fake_slide.svs", downsample=32)

    assert "(8, 8)" in str(exc_info.value)
    assert "(5, 5)" in str(exc_info.value)
