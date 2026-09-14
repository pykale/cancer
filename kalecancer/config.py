"""Configuration schema for KaleCancer pipelines.

Defaults are dataset-agnostic; paths are supplied by an experiment configuration,
command-line flags, or the Python API.
"""

from __future__ import annotations

from yacs.config import CfgNode

_C = CfgNode()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
_C.DATASET = CfgNode()
# Where the data comes from. "local" uses FEATURE_ROOT and CLINICAL_PATH; any other
# value is resolved by the fetcher the experiment supplies, caching into CACHE_DIR.
_C.DATASET.SOURCE = "local"
# Directory of pre-extracted patch features, searched recursively for .h5 files.
_C.DATASET.FEATURE_ROOT = ""
# JSON file of clinical records, one object per patient.
_C.DATASET.CLINICAL_PATH = ""
# Region to fetch, for datasets that publish more than one.
_C.DATASET.REGION = "primary"
# Patients to fetch from a remote source; 0 fetches the whole cohort.
_C.DATASET.PATIENTS = 0
# Cache for fetched data. Empty uses ~/.cache/kalecancer.
_C.DATASET.CACHE_DIR = ""
# Which published train/test assignment to apply, for a dataset that offers several.
# Empty here because the filename is the dataset's; an experiment configuration names it.
_C.DATASET.SPLIT_FILE = ""
# Read every file's header at start-up to reject invalid bags before training.
_C.DATASET.VALIDATE_FEATURES = True
# Cap on patches per bag during training, bounding memory for large slides.
# Evaluation and interpretation always use the full bag. 0 disables the cap.
_C.DATASET.MAX_PATCHES = 2048
_C.DATASET.NUM_WORKERS = 4
# How the test set is chosen. Three modes, in the order they are preferred:
#
#   "published"  the dataset's own train/test assignment, so a number is comparable
#                with other work on the cohort. The default, and what every example
#                uses; needs a dataset that publishes one.
#   "cv"         NUM_FOLDS-fold cross-validation over the whole cohort, for when a
#                single held-out split is too small to be informative.
#   "random"     a fresh stratified split using the ratios below.
#
# Validation is always carved out of the training half, never out of the test set.
_C.DATASET.SPLIT_MODE = "published"
# Share of the cohort held out for validation, and for testing when SPLIT_MODE is
# "random". Validation is taken from the training half in every mode.
_C.DATASET.VAL_RATIO = 0.15
_C.DATASET.TEST_RATIO = 0.15
# Number of folds when SPLIT_MODE is "cv".
_C.DATASET.NUM_FOLDS = 5
# Cohort column whose samples must never be split across sets.
_C.DATASET.GROUP_KEY = "patient_id"
# Cohort columns whose distribution is preserved across splits. Samples sharing a
# GROUP_KEY value stay together regardless of this setting.
_C.DATASET.STRATIFY_KEYS = ["event"]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
_C.MODEL = CfgNode()
# Patch feature width; 1024 for UNI embeddings.
_C.MODEL.INPUT_DIM = 1024
_C.MODEL.HIDDEN_DIM = 256
_C.MODEL.ATTENTION_DIM = 128
_C.MODEL.DROPOUT = 0.25
_C.MODEL.GATED = True

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.SEED = 2026
_C.SOLVER.BASE_LR = 1e-4
_C.SOLVER.WEIGHT_DECAY = 1e-5
_C.SOLVER.MAX_EPOCHS = 30
# The Cox risk set spans a batch, so batches must be large enough to contain events.
_C.SOLVER.BATCH_SIZE = 16
# Epochs without validation C-index improvement before stopping. 0 disables it.
_C.SOLVER.EARLY_STOP = 10
_C.SOLVER.OPTIMIZER = "AdamW"
# Accelerator devices passed to the trainer.
_C.SOLVER.DEVICES = "auto"

# ---------------------------------------------------------------------------
# Fusion
#
# Applies to multimodal pipelines; see kalecancer.model.embed.MultimodalFusion.
# STAGE and METHOD are independent: either can change on its own.
# ---------------------------------------------------------------------------
_C.FUSION = CfgNode()
# Where modalities meet: "intermediate" (features), "late" (predictions) or
# "hybrid" (both). "early" names raw-input fusion, which this pipeline cannot do
# because it starts from extracted features.
_C.FUSION.STAGE = "intermediate"
# How vectors are combined: "concat", "poe" or "lowrank". Used by "intermediate"
# and "hybrid".
_C.FUSION.METHOD = "concat"
# Width every modality is projected to, and of the fused representation.
_C.FUSION.FUSION_DIM = 256
# Rank of the factorisation, for METHOD "lowrank" only.
_C.FUSION.RANK = 4
# Hybrid only: a task head on each modality embedding.
_C.FUSION.AUXILIARY_HEADS = True
_C.FUSION.AUXILIARY_WEIGHT = 0.3
# Hybrid only: blend per-modality predictions into the output as well as the features.
_C.FUSION.COMBINE_PREDICTIONS = False
# Chance of dropping each present modality during training.
_C.FUSION.MODALITY_DROPOUT = 0.0

# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------
_C.SURVIVAL = CfgNode()
# Label for the endpoint, used when reporting the cohort.
_C.SURVIVAL.ENDPOINT = "OS"
# Which clinical columns define it, and which values count as an observed event.
# Empty here because these name the source dataset's columns; an experiment
# configuration sets them, and leaving a dataset's names as the default would apply
# them silently to a cohort that spells its columns differently.
_C.SURVIVAL.TIME_FIELD = ""
_C.SURVIVAL.STATUS_FIELD = ""
_C.SURVIVAL.EVENT_VALUES = []
# Status values that cannot be classified for this endpoint; excluded rather than
# assumed censored.
_C.SURVIVAL.UNKNOWN_VALUES = []
# Horizons for time-dependent metrics, in the unit of the survival time.
_C.SURVIVAL.EVAL_TIMES = [365.0, 1095.0, 1825.0]

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
_C.OUTPUT = CfgNode()
_C.OUTPUT.OUT_DIR = "outputs/wsi_survival"
# Top attended patches summarised per patient.
_C.OUTPUT.TOP_K = 20


def get_cfg_defaults() -> CfgNode:
    """Return a copy of the default configuration."""
    return _C.clone()
