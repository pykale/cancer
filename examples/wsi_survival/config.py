"""Configuration for the HANCOCK WSI survival example.

Extends the package defaults with the dataset paths for this experiment. Override
them with ``--cfg`` or trailing ``KEY VALUE`` pairs; no path belongs in the library.
"""

from yacs.config import CfgNode

from kalecancer.config import get_cfg_defaults as _package_defaults


def get_cfg_defaults() -> CfgNode:
    """Return the package defaults with this experiment's dataset paths applied."""
    cfg = _package_defaults()
    cfg.DATASET.FEATURE_ROOT = "E:/WSI_UNI_encodings/WSI_PrimaryTumor"
    cfg.DATASET.CLINICAL_PATH = "E:/StructuredData/StructuredData/clinical_data.json"
    return cfg
