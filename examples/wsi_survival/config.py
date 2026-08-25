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
    return cfg
