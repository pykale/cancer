"""Resolving the configured data source to local paths.

``DATASET.SOURCE`` selects where the cohort comes from; every source returns local
paths, so the rest of the pipeline is unaffected by the choice.

========== =====================================================================
local      Use ``DATASET.FEATURE_ROOT`` and ``DATASET.CLINICAL_PATH`` as given
hancock    Fetch the published HANCOCK archives into ``DATASET.CACHE_DIR``
synthetic  Generate a cohort locally; runs without network access
========== =====================================================================
"""

from __future__ import annotations

from pathlib import Path

from kalecancer.loaddata.hancock import DEFAULT_CACHE_DIR, HancockError, fetch_dataset
from kalecancer.loaddata.synthetic import write_synthetic_cohort

SOURCES = ("local", "hancock", "synthetic")

#: Patients generated when ``DATASET.PATIENTS`` is left at 0 for the synthetic source,
#: where "everything" has no meaning.
DEFAULT_SYNTHETIC_PATIENTS = 64


class DataSourceError(RuntimeError):
    """Raised when the configured data source cannot be resolved."""


def resolve_dataset(cfg) -> tuple[Path, Path]:
    """Resolve ``cfg.DATASET.SOURCE`` to a feature root and clinical file.

    Args:
        cfg: Pipeline configuration, see :func:`kalecancer.config.get_cfg_defaults`.

    Returns:
        ``(feature_root, clinical_path)``.

    Raises:
        DataSourceError: If the source is unknown, or a remote source is unreachable.
    """
    source = cfg.DATASET.SOURCE
    cache_dir = Path(cfg.DATASET.CACHE_DIR or DEFAULT_CACHE_DIR)

    if source == "local":
        return Path(cfg.DATASET.FEATURE_ROOT), Path(cfg.DATASET.CLINICAL_PATH)

    if source == "synthetic":
        return write_synthetic_cohort(
            cache_dir / "synthetic",
            num_patients=cfg.DATASET.PATIENTS or DEFAULT_SYNTHETIC_PATIENTS,
            feature_dim=cfg.MODEL.INPUT_DIM,
            seed=cfg.SOLVER.SEED,
        )

    if source == "hancock":
        try:
            return fetch_dataset(cache_dir=cache_dir, region=cfg.DATASET.REGION, patients=cfg.DATASET.PATIENTS)
        except HancockError as error:
            raise DataSourceError(
                f"{error}\n"
                "The HANCOCK archives are hosted by FAU Erlangen and are often unreachable from "
                "CI runners and sandboxes with restricted outbound network access.\n"
                "Options: allow 'hancock.research.fau.eu' and 'data.fau.de' through the network "
                "policy, point at an existing copy with DATASET.SOURCE 'local', or run offline "
                "with DATASET.SOURCE 'synthetic'."
            ) from error

    raise DataSourceError(f"unknown DATASET.SOURCE {source!r}; expected one of {SOURCES}")
