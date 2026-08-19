"""Generating a synthetic WSI cohort on disk.

Writes the same file layout the real pipeline reads - HDF5 patch bags plus a
clinical JSON - so the whole workflow can run without network access or a licensed
dataset. Useful in CI and offline environments, and for reproducing an issue without
sharing patient data.

Labels come from :func:`~kalecancer.survival.synthetic.make_synthetic_survival`, a
linear Cox model, so the risk is genuinely learnable from the patches rather than
noise: each patient's patches are drawn around that patient's embedding, which
attention pooling can recover.

Metrics measured on this cohort describe the pipeline, not clinical performance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py
import numpy as np

from kalecancer.survival.synthetic import make_synthetic_survival

logger = logging.getLogger(__name__)

PATCH_STRIDE = 2048


def write_synthetic_cohort(
    destination: str | Path,
    num_patients: int = 64,
    feature_dim: int = 1024,
    patches_per_slide: tuple[int, int] = (40, 160),
    multi_slide_every: int = 8,
    patch_noise: float = 0.5,
    seed: int = 0,
) -> tuple[Path, Path]:
    """Write a synthetic cohort and return its feature root and clinical file.

    Bag sizes vary per slide and every ``multi_slide_every``-th patient gets a second
    slide, so the generated cohort exercises variable-length bags and patient-level
    grouping rather than a uniform best case.

    Args:
        destination: Directory to write into. Existing files are overwritten.
        num_patients: Number of patients to generate.
        feature_dim: Patch feature dimension; must match ``MODEL.INPUT_DIM``.
        patches_per_slide: Inclusive ``(minimum, maximum)`` patches per slide.
        multi_slide_every: Give every n-th patient a second slide. 0 disables this.
        patch_noise: Standard deviation of the noise added around a patient's
            embedding to form its patches.
        seed: Seed making the cohort reproducible.

    Returns:
        ``(feature_root, clinical_path)``, ready to pass to
        :func:`~kalecancer.loaddata.cohort.build_cohort`.
    """
    destination = Path(destination)
    feature_root = destination / "features"
    feature_root.mkdir(parents=True, exist_ok=True)

    cohort = make_synthetic_survival(n_samples=num_patients, n_features=feature_dim, seed=seed)
    embeddings = cohort.embeddings.numpy()
    generator = np.random.default_rng(seed)

    records = []
    for index in range(num_patients):
        patient_id = f"{index:04d}"
        slides = 2 if multi_slide_every and index % multi_slide_every == 0 else 1
        for slide in range(slides):
            num_patches = int(generator.integers(patches_per_slide[0], patches_per_slide[1] + 1))
            patches = embeddings[index] + generator.normal(scale=patch_noise, size=(num_patches, feature_dim))
            coords = generator.integers(0, 512, size=(num_patches, 2)) * PATCH_STRIDE

            suffix = "" if slide == 0 else f"_{chr(ord('a') + slide - 1)}"
            path = feature_root / f"PrimaryTumor_HE_{patient_id}{suffix}.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("features", data=patches.astype(np.float32))
                handle.create_dataset("coords", data=coords.astype(np.int32))

        records.append(
            {
                "patient_id": patient_id,
                # Follow-up is recorded in whole days, and the loader rejects a
                # non-positive duration, so the shortest survivable time is one day.
                "days_to_last_information": max(1, round(float(cohort.times[index]))),
                "survival_status": "deceased" if bool(cohort.events[index]) else "living",
            }
        )

    clinical_path = destination / "clinical_data.json"
    clinical_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    logger.info(
        "synthetic cohort: %d patients, %d events, %d slides in %s",
        num_patients,
        int(cohort.events.sum()),
        len(list(feature_root.glob("*.h5"))),
        destination,
    )
    return feature_root, clinical_path
