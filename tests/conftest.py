"""Shared fixtures for the ``kalecancer.loaddata`` tests.

Everything here is built on a small **synthetic** cohort rather than the real
HANCOCK table. That is deliberate: the fixture is fast, deterministic, checked in
alongside the tests, and -- most importantly -- it can carry the edge cases real
data happens not to have on the day you look at it:

* zero-padded string identifiers (``"001"``), which pandas will silently turn
  into integers given half a chance;
* missing values in both a continuous and a categorical column;
* a rare category confined to the last two rows, so a train/test split can be
  arranged to put it *only* in the held-out half;
* a mixture of numeric and string dtypes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalecancer.loaddata.tabular import TabularDataset
from kalecancer.survival.survival_target import SurvivalTarget

N_PATIENTS = 80

CONTINUOUS = ["age", "biomarker"]
CATEGORICAL = ["sex", "stage", "smoking"]

TIME_COLUMN = "os_days"
EVENT_COLUMN = "vital_status"
EVENT_VALUE = "deceased"

#: Rows carrying the rare ``stage="IV"`` category, as positional indices.
RARE_STAGE_ROWS = [N_PATIENTS - 2, N_PATIENTS - 1]

#: Rows with a missing ``age``, as positional indices.
MISSING_AGE_ROWS = [3, 17, 44]


def build_frame(n: int = N_PATIENTS, seed: int = 0) -> pd.DataFrame:
    """Construct the synthetic cohort. Deterministic for a given ``seed``."""
    rng = np.random.default_rng(seed)

    age = rng.normal(62.0, 11.0, n).round(1)
    age[MISSING_AGE_ROWS] = np.nan

    smoking = rng.choice(["never", "former", "current"], n).astype(object)
    smoking[[2, 9]] = None

    stage = rng.choice(["I", "II", "III"], n).astype(object)
    for row in RARE_STAGE_ROWS:
        stage[row] = "IV"

    vital = np.where(rng.random(n) < 0.35, EVENT_VALUE, "living").astype(object)
    # Pin both classes so the target's degenerate-event-rate guard never fires
    # on the fixture itself, whatever the seed.
    vital[0], vital[1] = EVENT_VALUE, "living"

    return pd.DataFrame(
        {
            "patient_id": [f"{i:03d}" for i in range(1, n + 1)],
            "age": age,
            "biomarker": rng.gamma(2.0, 1.5, n).round(3),
            "sex": rng.choice(["female", "male"], n),
            "stage": stage,
            "smoking": smoking,
            TIME_COLUMN: rng.integers(30, 4000, n),
            EVENT_COLUMN: vital,
        }
    )


def write_table(frame: pd.DataFrame, directory: Path, fmt: str = "csv") -> Path:
    """Write ``frame`` to ``directory`` in ``fmt`` and return the path."""
    path = directory / f"cohort.{fmt}"
    if fmt == "csv":
        frame.to_csv(path, index=False)
    elif fmt == "tsv":
        frame.to_csv(path, sep="\t", index=False)
    elif fmt == "json":
        frame.to_json(path, orient="records")
    else:  # pragma: no cover - guards the fixture itself
        raise ValueError(f"unsupported fixture format {fmt!r}")
    return path


@pytest.fixture
def frame() -> pd.DataFrame:
    """A fresh copy of the synthetic cohort, so a test can mutate it freely."""
    return build_frame()


@pytest.fixture
def table_path(frame: pd.DataFrame, tmp_path: Path) -> Path:
    """The synthetic cohort written to a CSV."""
    return write_table(frame, tmp_path)


@pytest.fixture(params=["csv", "tsv", "json"])
def any_format(request: pytest.FixtureRequest) -> str:
    """Each supported table format in turn.

    Every one of these reads with a stdlib-only pandas, so nothing here skips. A
    format that needs an optional engine does not belong in the reader at all --
    it would advertise support the default install cannot deliver, and its tests
    would skip in CI, leaving the reader unexercised.
    """
    return request.param


@pytest.fixture
def make_target():
    """Factory for a ``SurvivalTarget`` over the fixture's columns.

    A factory rather than a single instance: ``bind`` stores state on the target,
    so two datasets sharing one instance would fight over it.
    """

    def build(**overrides) -> SurvivalTarget:
        kwargs = {"time": TIME_COLUMN, "event": EVENT_COLUMN, "event_value": EVENT_VALUE}
        kwargs.update(overrides)
        return SurvivalTarget(**kwargs)

    return build


@pytest.fixture
def make_dataset(table_path: Path, make_target):
    """Factory for a fully-declared ``TabularDataset`` over the synthetic cohort.

    Defaults to the realistic setup -- impute-then-scale on the continuous
    columns, impute-then-one-hot on the categorical ones -- with every argument
    overridable. Import the transformers here rather than at module scope so a
    test that overrides them is not shadowed by an unused import.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    def build(**overrides) -> TabularDataset:
        kwargs = {
            "source": table_path,
            "identifier": "patient_id",
            "target": make_target(),
            "continuous": CONTINUOUS,
            "continuous_transform": [SimpleImputer(strategy="median"), StandardScaler()],
            "categorical": CATEGORICAL,
            "categorical_transform": [
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ],
        }
        kwargs.update(overrides)
        return TabularDataset(**kwargs)

    return build


@pytest.fixture
def cohort(make_dataset) -> TabularDataset:
    """A fully-declared, unfitted cohort."""
    return make_dataset()
