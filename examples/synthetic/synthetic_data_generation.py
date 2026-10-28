import numpy as np
import pandas as pd
import h5py
from pathlib import Path

PATCH_DIM = 32

def make_cohort(
        folder: str | Path,
        n_patients: int = 100,
        random_state: int = 42,
        missingness: float = 0.1,
        task: str = 'classification'
    ):
    """Write one h5 file of patch features per patient into ``folder``, plus ``clinical.csv`` with the target.

    ``task='survival'`` writes ``time`` and ``event`` columns, with ``stage`` as a clinical feature.
    ``task='classification'`` writes a ``stage`` column of ``'early'``/``'late'`` labels to predict, driven mostly by the slides.
    """
    if task not in ('survival', 'classification'):
        raise ValueError(f"task must be 'survival' or 'classification', got {task!r}")
    folder = Path(folder)
    folder.mkdir(parents = True, exist_ok=True)
    rng = np.random.default_rng(random_state)

    # tabular data
    ids = pd.Index([f"patient_{k:03d}" for k in range(n_patients)], name="patient_id")
    clinical = pd.DataFrame(
        {
            "age": rng.normal(60, 10, n_patients),
            "stage": rng.integers(1, 5, n_patients),
            "gender": rng.choice(['F', 'M'], size=n_patients)
        },
        index=ids)

    # patch features (i.e. what output from UNI could look like)
    aggressive_fraction = rng.uniform(0, 0.5, n_patients)
    for patient, fraction in zip(ids, aggressive_fraction, strict=True):
        n_patches = int(rng.integers(20, 200))
        features = rng.normal(size=(n_patches, PATCH_DIM)).astype(np.float32)
        features[: int(fraction * n_patches), 0] += 3.0  # aggressive patches stand out in one feature
        patches_folder = folder / 'patches'
        patches_folder.mkdir(exist_ok=True)
        with h5py.File(patches_folder / f"{patient}.h5", "w") as f:
            f["features"] = features
            f["coords"] = rng.integers(0, 50_000, size=(n_patches, 2))

    # target
    if task == 'survival':
        risk = 0.09 * (clinical["age"] - 60) + 1.2 * (clinical["stage"] - 2.5) + 12.0 * aggressive_fraction
        event_time = rng.exponential(2000 * np.exp(-risk))
        censoring_time = rng.uniform(200, 3000, n_patients)
        outcome = pd.DataFrame(
            {"time": np.minimum(event_time, censoring_time), "event": event_time <= censoring_time}, index=ids
        )
    else:
        # stage follows the slide (more aggressive patches -> later stage), with a small age effect
        score = aggressive_fraction + 0.005 * (clinical["age"] - 60) + rng.normal(0, 0.08, n_patients)
        outcome = pd.DataFrame({"stage": np.where(score > 0.25, 'late', 'early')}, index=ids)
        clinical = clinical.drop(columns="stage")

    # missingess for tabular data
    clinical = clinical.mask(
        rng.choice([True, False], size= clinical.shape, p = [missingness, 1.0-missingness])
    )

    # saving tabular + target
    clinical = clinical.join(outcome)
    clinical.to_csv(folder / 'clinical.csv')
