"""Generic access to whole-slide patch features stored as HDF5 bags.

Each file holds one slide encoded by a pathology foundation model (e.g. UNI via
CLAM):

``features``
    ``(num_patches, feature_dim)`` patch embeddings.
``coords``
    ``(num_patches, 2)`` patch coordinates in the source slide, row-aligned with
    ``features`` so attention scores stay traceable to slide locations.

Nothing here knows a dataset. How a slide filename encodes the subject it belongs to
differs per collection, so :func:`slide_table` takes that pattern as an argument
rather than assuming one; the pattern for a given dataset belongs with that dataset.

There is no ``Dataset`` class here either. A bag is one modality among any number, so
it is served by :class:`~kalecancer.loaddata.multimodal_access.FeatureBagSource` inside a
:class:`~kalecancer.loaddata.multimodal_access.MultimodalDataset` -- which is what makes
imaging + imaging, or imaging + tabular, the same object with a different dictionary.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

FEATURES_KEY = "features"
COORDS_KEY = "coords"


class InvalidFeatureFileError(ValueError):
    """Raised when an HDF5 feature file cannot be used as a patch bag."""


class SlideIdentifierError(ValueError):
    """Raised when a patient identifier cannot be parsed from a slide name."""


def parse_patient_id(slide_id: str, pattern: re.Pattern[str]) -> str:
    """Extract the patient identifier from a slide identifier.

    Args:
        slide_id: Slide name without extension, e.g. ``"region_stain_036_a"``.
        pattern: Regular expression exposing a ``patient_id`` named group.

    Returns:
        The patient identifier exactly as written in the filename.

    Raises:
        SlideIdentifierError: If the slide name does not match ``pattern``.
    """
    match = pattern.match(slide_id)
    if match is None:
        raise SlideIdentifierError(
            f"cannot parse a patient id from slide {slide_id!r} using pattern {pattern.pattern!r}"
        )
    return match.group("patient_id")


def slide_table(
    root: str | Path,
    pattern: re.Pattern[str],
    extension: str = ".h5",
) -> pd.DataFrame:
    """Recursively discover slide feature files below ``root``.

    Directory depth is not assumed: any nesting of subsite or batch folders works.

    Args:
        root: Directory to search.
        pattern: Regular expression exposing a ``patient_id`` named group.
        extension: Feature file extension.

    Returns:
        A ``patient_id``/``slide_id``/``path`` table sorted by identifier. Files whose
        identifier could not be parsed are listed in ``attrs["unparsed"]``.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"feature root does not exist or is not a directory: {root}")

    rows, unparsed = [], []
    for path in sorted(root.rglob(f"*{extension}")):
        try:
            patient_id = parse_patient_id(path.stem, pattern)
        except SlideIdentifierError as error:
            unparsed.append({"path": str(path), "reason": str(error)})
            continue
        rows.append({"patient_id": patient_id, "slide_id": path.stem, "path": path})

    table = pd.DataFrame(rows, columns=["patient_id", "slide_id", "path"])
    table = table.sort_values(["patient_id", "slide_id"], ignore_index=True)
    table.attrs["unparsed"] = unparsed
    return table


def _validated_shape(handle: h5py.File, path: Path, expected_dim: int | None) -> tuple[int, int]:
    """Check the bag contract and return ``(num_patches, feature_dim)``."""
    missing = {FEATURES_KEY, COORDS_KEY} - set(handle.keys())
    if missing:
        raise InvalidFeatureFileError(
            f"{path}: missing required key(s) {sorted(missing)}; found {sorted(handle.keys())}"
        )

    features_shape = handle[FEATURES_KEY].shape
    coords_shape = handle[COORDS_KEY].shape

    if len(features_shape) != 2:
        raise InvalidFeatureFileError(
            f"{path}: expected 2D '{FEATURES_KEY}' (num_patches, feature_dim), got shape {features_shape}"
        )
    if features_shape[0] == 0:
        raise InvalidFeatureFileError(f"{path}: '{FEATURES_KEY}' is empty")
    if len(coords_shape) != 2 or coords_shape[1] < 2:
        raise InvalidFeatureFileError(f"{path}: expected 2D '{COORDS_KEY}' (num_patches, 2), got shape {coords_shape}")
    if features_shape[0] != coords_shape[0]:
        raise InvalidFeatureFileError(
            f"{path}: '{FEATURES_KEY}' has {features_shape[0]} rows but '{COORDS_KEY}' has "
            f"{coords_shape[0]}; patch alignment is required"
        )
    if expected_dim is not None and features_shape[1] != expected_dim:
        raise InvalidFeatureFileError(f"{path}: expected feature dimension {expected_dim}, got {features_shape[1]}")

    return int(features_shape[0]), int(features_shape[1])


def read_feature_bag(
    path: str | Path,
    expected_dim: int | None = None,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one patch bag, validating the contract before returning it.

    Args:
        path: HDF5 feature file.
        expected_dim: Feature dimension the caller requires, if it should be checked.
        indices: Optional patch subset. Sorted internally so ``features`` and
            ``coords`` stay row-aligned.

    Returns:
        ``(features, coords)`` as ``float32`` and ``int64`` arrays.

    Raises:
        InvalidFeatureFileError: If a key is missing, the bag is empty, the arrays
            disagree in length, the feature dimension is unexpected, or ``indices``
            falls outside the bag.
    """
    path = Path(path)
    try:
        with h5py.File(path, "r") as handle:
            num_patches, _ = _validated_shape(handle, path, expected_dim)
            if indices is None:
                features = handle[FEATURES_KEY][:]
                coords = handle[COORDS_KEY][:]
            else:
                # Sorted and deduplicated because HDF5 fancy indexing requires it, and
                # bounds-checked here so an out-of-range patch is named with its file.
                selection = np.unique(np.asarray(indices, dtype=np.int64))
                if selection.size and (selection[0] < 0 or selection[-1] >= num_patches):
                    raise InvalidFeatureFileError(
                        f"{path}: patch indices must lie in [0, {num_patches}), got [{selection[0]}, {selection[-1]}]"
                    )
                features = handle[FEATURES_KEY][selection]
                coords = handle[COORDS_KEY][selection]
    except OSError as error:
        raise InvalidFeatureFileError(f"{path}: cannot be read as HDF5 ({error})") from error

    return features.astype(np.float32, copy=False), coords.astype(np.int64, copy=False)


def inspect_feature_bag(path: str | Path, expected_dim: int | None = None) -> tuple[int, int]:
    """Return ``(num_patches, feature_dim)`` without reading the patch data.

    Raises:
        InvalidFeatureFileError: Under the same conditions as :func:`read_feature_bag`.
    """
    path = Path(path)
    try:
        with h5py.File(path, "r") as handle:
            return _validated_shape(handle, path, expected_dim)
    except OSError as error:
        raise InvalidFeatureFileError(f"{path}: cannot be read as HDF5 ({error})") from error
