"""Configuration for the HANCOCK multimodal survival example.

Extends the package defaults with what this experiment adds: which modalities to
use, how the clinical table is encoded, and where HANCOCK lives. Selecting the
modalities is what turns one code path into three runs.
"""

from yacs.config import CfgNode

from kalecancer.config import get_cfg_defaults as _package_defaults


def get_cfg_defaults() -> CfgNode:
    """Return the package defaults extended for this experiment."""
    cfg = _package_defaults()

    # Which modalities take part. One name gives a unimodal baseline; two are fused.
    cfg.DATASET.MODALITIES = ["clinical", "imaging"]
    # Modalities a patient must have to enter the cohort, beyond those used. Set to
    # ["imaging"] for a unimodal run to score it on the same patients as the
    # multimodal run, which is what makes the two directly comparable.
    cfg.DATASET.REQUIRE_MODALITIES = []
    # Use the published train/test assignment rather than a fresh random split.
    cfg.DATASET.OFFICIAL_SPLIT = True
    cfg.DATASET.SOURCE = "hancock"

    # The clinical table: which columns are used, and how each role is encoded.
    cfg.TABULAR = CfgNode()
    cfg.TABULAR.CONTINUOUS = ["age_at_initial_diagnosis", "year_of_initial_diagnosis"]
    cfg.TABULAR.CATEGORICAL = ["sex", "smoking_status", "primarily_metastasis"]
    # TabICL conditions its representation on labelled context rows, taken from the
    # training split only. The checkpoint downloads from Hugging Face on first use.
    cfg.TABULAR.TRAINABLE = False
    cfg.TABULAR.CHECKPOINT = ""
    cfg.TABULAR.N_ESTIMATORS = 1
    # Hidden widths of the projection applied before fusion. Empty is a single linear
    # layer; the default gives the small MLP a frozen representation benefits from.
    cfg.TABULAR.PROJECTION_HIDDEN = [128]

    cfg.SURVIVAL.EVAL_TIMES = [365.0, 730.0, 1095.0]
    cfg.OUTPUT.OUT_DIR = "outputs/hancock_multimodal"
    return cfg
