"""Configuration for the HANCOCK WSI survival example.

Extends the package defaults with this experiment's data source. Override anything
with ``--cfg`` or trailing ``KEY VALUE`` pairs; no path belongs in the library.

The default source is the published archive rather than a local directory, so the
example runs on any machine without a copy of the data. To read files already on
disk instead, use ``configs/local_primary_tumour.yaml`` or set ``DATASET.SOURCE``
to ``local`` with ``DATASET.FEATURE_ROOT`` and ``DATASET.CLINICAL_PATH``.
"""

from yacs.config import CfgNode

from kalecancer.config import get_cfg_defaults as _package_defaults


def get_cfg_defaults() -> CfgNode:
    """Return the package defaults with this experiment's data source applied."""
    cfg = _package_defaults()
    cfg.DATASET.SOURCE = "hancock"
    cfg.DATASET.SPLIT_FILE = "dataset_split_in.json"
    # HANCOCK's own column names and event vocabulary. The library carries no
    # default for these: applying one dataset's spelling to another silently marks
    # every patient censored.
    cfg.SURVIVAL.TIME_FIELD = "days_to_last_information"
    cfg.SURVIVAL.STATUS_FIELD = "survival_status"
    cfg.SURVIVAL.EVENT_VALUES = ["deceased"]
    return cfg
