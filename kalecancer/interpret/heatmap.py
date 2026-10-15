"""Turning per-patch attention into a spatial heatmap over a slide.

:mod:`~kalecancer.interpret.attention` exports attention weights paired with the
patch coordinates they came from; this module renders them. Two layers, kept
deliberately separate:

(a) **Rasterisation** (:func:`attention_heatmap`, :func:`save_heatmap`) -- pure numpy
    (plus the standard library for PNG encoding). No slide file, no pathology
    dependency. This is the whole substance of the module, and it is what every test
    below exercises without anything from the ``pathology`` extra installed.
(b) **Overlay** (:func:`overlay_on_slide`) -- a thin wrapper needing an actual slide
    file read through ``openslide``, imported lazily so that importing this module,
    or any of :mod:`kalecancer.interpret`, never requires it.

**Coordinate convention.** ``coords`` are level-0 (full-resolution) pixel coordinates
of each patch's top-left corner, exactly as
:mod:`kalecancer.loaddata.wsi_access` stores and :mod:`kalecancer.interpret.attention`
reads them: ``coords[:, 0]`` is x, ``coords[:, 1]`` is y. ``downsample`` is how many
level-0 pixels map to one output cell, so a level-0 coordinate lands in the grid at
``coordinate // downsample``. Getting the level wrong here -- rasterising at a
downsample that does not match what the coordinates are actually expressed in --
produces a heatmap that still looks plausible but is spatially wrong, which is worse
than an error; hence this is stated explicitly rather than left to be inferred.

**What the ``.h5`` loader does NOT record.** ``kalecancer/loaddata/wsi_access.py``
validates only that ``features`` is ``(N, D)`` and ``coords`` is ``(N, >=2)``, row
counts matching; it stores no patch size, no pyramid level, and no units anywhere in
that schema or its docstring. The HANCOCK examples' own README states its imaging
pipeline matches "CLAM... at 256x256" patches, and CLAM's own convention is to record
coordinates as level-0 pixels of each patch's top-left corner -- which is what the
level-0 convention above assumes -- but neither the schema nor the loader enforces or
checks this anywhere; it is supplied by whoever ran the extraction. That is why
``patch_size`` below has no default: this module cannot discover it, so it must be an
explicit, deliberate argument every time, not a guess.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

_AGGREGATES = ("mean", "max", "sum")

#: Published viridis anchor colours (t, r, g, b), used only when matplotlib is not
#: installed. Five points, linearly interpolated between.
_VIRIDIS_ANCHORS = np.array(
    [
        [0.00, 68, 1, 84],
        [0.25, 59, 82, 139],
        [0.50, 33, 145, 140],
        [0.75, 94, 201, 98],
        [1.00, 253, 231, 37],
    ],
    dtype=np.float64,
)


def attention_heatmap(
    coords: np.ndarray,
    weights: np.ndarray,
    *,
    patch_size: int,
    downsample: int = 32,
    canvas_shape: tuple[int, int] | None = None,
    aggregate: str = "mean",
    normalise: bool = True,
) -> np.ndarray:
    """Rasterise patch attention into a 2-D heatmap. Pure numpy.

    Each patch occupies the block of output cells its ``patch_size x patch_size``
    level-0 footprint overlaps once divided by ``downsample`` -- one cell exactly
    when ``downsample == patch_size`` and the patch is grid-aligned, more than one
    when ``downsample < patch_size``, part of one when ``downsample > patch_size``.

    Args:
        coords: ``(N, 2)`` level-0 top-left pixel coordinates, ``(x, y)`` per patch
            -- see the module docstring for what "level-0" assumes about the data.
        weights: ``(N,)`` attention weight per patch, row-aligned with ``coords``.
        patch_size: Patch width/height in level-0 pixels. No default: the ``.h5``
            loader this package reads does not record it (see the module docstring),
            so it must come from whoever extracted the patches.
        downsample: Level-0 pixels per output cell.
        canvas_shape: ``(H, W)`` in level-0 pixels. ``None`` infers the canvas from
            the coordinates' own extent plus one patch width/height -- which crops
            the output to the tissue bounding box the patches cover, NOT the full
            slide (a slide is typically mostly background outside that box). Pass
            the slide's real level-0 ``(height, width)`` explicitly when the result
            needs to line up with the untouched slide, e.g. before
            :func:`overlay_on_slide`.
        aggregate: How overlapping patches combine within one cell -- ``"mean"``
            (tracked via a separate count array, not by re-dividing a running sum),
            ``"max"``, or ``"sum"``.
        normalise: Scale the result to ``[0, 1]`` over the non-``NaN`` cells only,
            leaving ``NaN`` cells untouched. A perfectly uniform heatmap (every
            contributing patch sharing one weight) normalises to all-zero: there is
            no contrast to stretch, so 0 is used rather than an arbitrary non-``[0,
            1]`` constant.

    Returns:
        ``(H, W)`` float64 array. A cell with no patch covering it is ``NaN`` --
        never ``0`` -- because zero attention and no tissue are different things a
        viewer must be able to tell apart.

    Raises:
        ValueError: If ``coords``/``weights`` are the wrong shape or disagree in
            length, if ``patch_size``/``downsample`` is not positive, or if
            ``aggregate`` is not one of ``"mean"``, ``"max"``, ``"sum"``.
    """
    coords = np.asarray(coords)
    weights = np.asarray(weights, dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (N, 2), got {coords.shape}")
    if weights.ndim != 1 or weights.shape[0] != coords.shape[0]:
        raise ValueError(f"weights must have shape (N,) matching coords's N, got {weights.shape} vs {coords.shape}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if downsample <= 0:
        raise ValueError(f"downsample must be positive, got {downsample}")
    if aggregate not in _AGGREGATES:
        raise ValueError(f"aggregate must be one of {_AGGREGATES}, got {aggregate!r}")

    x = coords[:, 0].astype(np.int64)
    y = coords[:, 1].astype(np.int64)

    if canvas_shape is None:
        # Crop to the patches' own bounding box rather than assuming the slide
        # origin is (0, 0) -- see the Args docstring on canvas_shape.
        origin_x = int(x.min()) if x.size else 0
        origin_y = int(y.min()) if y.size else 0
        extent_x = (int(x.max()) + patch_size - origin_x) if x.size else patch_size
        extent_y = (int(y.max()) + patch_size - origin_y) if y.size else patch_size
    else:
        origin_x = origin_y = 0
        extent_y, extent_x = canvas_shape

    height = -(-extent_y // downsample)  # ceil division, stdlib-only
    width = -(-extent_x // downsample)

    gx0 = np.clip((x - origin_x) // downsample, 0, width - 1)
    gy0 = np.clip((y - origin_y) // downsample, 0, height - 1)
    gx1 = np.clip((x - origin_x + patch_size - 1) // downsample, 0, width - 1)
    gy1 = np.clip((y - origin_y + patch_size - 1) // downsample, 0, height - 1)

    sum_grid = np.zeros((height, width), dtype=np.float64)
    count_grid = np.zeros((height, width), dtype=np.int64)
    max_grid = np.full((height, width), -np.inf, dtype=np.float64)
    touched = np.zeros((height, width), dtype=bool)

    # A patch's footprint can span several cells (or several patches can share one),
    # so this is a scatter over variable-size blocks -- not expressible as a single
    # vectorised op without assuming one patch maps to exactly one cell. Rasterising
    # one slide's few thousand patches this way costs milliseconds, not a hot path.
    for i in range(weights.shape[0]):
        block = (slice(gy0[i], gy1[i] + 1), slice(gx0[i], gx1[i] + 1))
        sum_grid[block] += weights[i]
        count_grid[block] += 1
        np.maximum(max_grid[block], weights[i], out=max_grid[block])
        touched[block] = True

    output = np.full((height, width), np.nan, dtype=np.float64)
    if aggregate == "sum":
        output[touched] = sum_grid[touched]
    elif aggregate == "mean":
        output[touched] = sum_grid[touched] / count_grid[touched]
    else:
        output[touched] = max_grid[touched]

    if normalise:
        valid = ~np.isnan(output)
        if valid.any():
            lo, hi = output[valid].min(), output[valid].max()
            output[valid] = (output[valid] - lo) / (hi - lo) if hi > lo else 0.0

    return output


def _normalise_for_colour(heatmap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(normalised_values, valid_mask)``, scaling non-NaN cells to ``[0, 1]``."""
    valid = ~np.isnan(heatmap)
    normalised = np.zeros_like(heatmap, dtype=np.float64)
    if valid.any():
        lo, hi = heatmap[valid].min(), heatmap[valid].max()
        normalised[valid] = (heatmap[valid] - lo) / (hi - lo) if hi > lo else 0.0
    return normalised, valid


def _matplotlib_colourmap(values: np.ndarray, colormap: str) -> np.ndarray | None:
    """``(H, W, 3)`` uint8, or ``None`` if matplotlib is not installed.

    Uses ``matplotlib.colormaps``, the registry matplotlib has offered since 3.5
    (2021); matplotlib is not a declared dependency at all, so there is no older
    floor this needs to keep working against.
    """
    try:
        import matplotlib
    except ImportError:
        return None

    cmap = matplotlib.colormaps[colormap]
    rgba = cmap(values)
    return (rgba[..., :3] * 255).astype(np.uint8)


def _fallback_colourmap(values: np.ndarray, colormap: str) -> np.ndarray:
    """A small built-in colormap, used only without matplotlib installed.

    Approximates ``"viridis"`` by piecewise-linear interpolation between five
    published anchor colours; any other name falls back further, to a plain
    grayscale ramp -- a deliberately modest substitute, not a matplotlib clone.
    """
    flat = values.ravel()
    if colormap == "viridis":
        positions = _VIRIDIS_ANCHORS[:, 0]
        channels = [np.interp(flat, positions, _VIRIDIS_ANCHORS[:, c]) for c in (1, 2, 3)]
        rgb = np.stack(channels, axis=-1)
    else:
        grey = flat * 255.0
        rgb = np.stack([grey, grey, grey], axis=-1)
    return rgb.reshape(*values.shape, 3).astype(np.uint8)


def _colourise(heatmap: np.ndarray, colormap: str, nan_colour: tuple[int, int, int]) -> np.ndarray:
    normalised, valid = _normalise_for_colour(heatmap)
    rgb = _matplotlib_colourmap(normalised, colormap)
    if rgb is None:
        rgb = _fallback_colourmap(normalised, colormap)
    rgb[~valid] = nan_colour
    return rgb


def _write_png(path: Path, rgb: np.ndarray) -> None:
    """Write an ``(H, W, 3)`` uint8 array as a PNG, using only the standard library.

    PNG's container is simple enough to assemble directly -- ``struct`` for the
    chunk framing, ``zlib`` for the compression the format itself requires -- which
    keeps this free of any new dependency, hard or optional, for something as small
    as writing one raster image.
    """
    height, width, _ = rgb.shape
    scanlines = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    compressed = zlib.compress(scanlines, level=6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit, colour type 2 = RGB
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    path.write_bytes(png)


def save_heatmap(
    heatmap: np.ndarray,
    path: str | Path,
    *,
    colormap: str = "viridis",
    nan_colour: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Write the heatmap to a PNG.

    Uses matplotlib's colormap if it is installed (imported lazily, so it is never
    required), otherwise a small built-in fallback (see :func:`_fallback_colourmap`)
    -- this module does not add a hard matplotlib dependency for one colour lookup.
    The PNG itself is written with the standard library only (see
    :func:`_write_png`), so no imaging library is needed even for that.

    Args:
        heatmap: ``(H, W)`` array, as :func:`attention_heatmap` returns; ``NaN``
            cells are painted ``nan_colour``.
        path: Output file.
        colormap: Name looked up via matplotlib when available. Without matplotlib,
            only ``"viridis"`` has a dedicated approximation; any other name falls
            back to grayscale.
        nan_colour: RGB colour (0-255 per channel) for cells with no patch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _colourise(np.asarray(heatmap, dtype=np.float64), colormap, nan_colour)
    _write_png(path, rgb)


def _require_openslide():
    """Import openslide, explaining the install if it is absent."""
    try:
        import openslide
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "overlay_on_slide needs openslide-python, which is not installed. "
            'Install it with: pip install "kalecancer[pathology]"'
        ) from error
    return openslide


def overlay_on_slide(
    heatmap: np.ndarray,
    slide_path: str | Path,
    *,
    alpha: float = 0.5,
    downsample: int = 32,
) -> np.ndarray:
    """Composite the heatmap over a slide thumbnail. Requires openslide.

    Reads the slide at whichever pyramid level best matches ``downsample``, then
    resizes to exactly ``ceil(level-0 dimensions / downsample)`` -- the same formula
    :func:`attention_heatmap` uses for an explicit ``canvas_shape`` -- so the two
    agree exactly when the heatmap was built with ``canvas_shape=(slide_height,
    slide_width)`` at this same ``downsample``. A heatmap built with
    ``canvas_shape=None`` (cropped to the tissue bounding box) will not match a full
    slide's thumbnail and is refused below, rather than silently composited at the
    wrong place.

    Args:
        heatmap: ``(H, W)`` array, as :func:`attention_heatmap` returns.
        slide_path: Path to the whole-slide image file, read via openslide.
        alpha: Heatmap opacity over the thumbnail, ``0`` (invisible) to ``1`` (opaque).
            ``NaN`` cells are never blended in: the thumbnail shows through untouched.
        downsample: Level-0 pixels per thumbnail pixel; must match the value the
            heatmap was rasterised at.

    Returns:
        ``(H, W, 3)`` uint8 composite.

    Raises:
        ImportError: If ``openslide-python`` is not installed.
        ValueError: If the slide thumbnail's shape does not match ``heatmap``'s.
    """
    openslide = _require_openslide()

    with openslide.OpenSlide(str(slide_path)) as slide:
        level0_width, level0_height = slide.dimensions
        thumb_width = -(-level0_width // downsample)
        thumb_height = -(-level0_height // downsample)

        level = slide.get_best_level_for_downsample(downsample)
        region = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB")
        thumbnail = np.asarray(region.resize((thumb_width, thumb_height)))

    if thumbnail.shape[:2] != heatmap.shape:
        raise ValueError(
            f"slide thumbnail shape {thumbnail.shape[:2]} does not match heatmap shape {heatmap.shape}; "
            "overlay_on_slide needs a heatmap rasterised with canvas_shape=(slide_height, slide_width) "
            "at this same downsample -- a heatmap cropped to the tissue bounding box "
            "(canvas_shape=None) will not align with a full slide's thumbnail"
        )

    normalised, valid = _normalise_for_colour(np.asarray(heatmap, dtype=np.float64))
    rgb = _matplotlib_colourmap(normalised, "viridis")
    if rgb is None:
        rgb = _fallback_colourmap(normalised, "viridis")

    composite = thumbnail.astype(np.float64)
    composite[valid] = (1.0 - alpha) * composite[valid] + alpha * rgb[valid]
    return composite.astype(np.uint8)
