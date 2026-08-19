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
# Where the data comes from: "local" uses FEATURE_ROOT and CLINICAL_PATH, "hancock"
# fetches from the published HANCOCK archives into CACHE_DIR.
_C.DATASET.SOURCE = "local"
# Directory of pre-extracted patch features, searched recursively for .h5 files.
_C.DATASET.FEATURE_ROOT = ""
# JSON file of clinical records, one object per patient.
_C.DATASET.CLINICAL_PATH = ""
# Anatomical region to fetch: "primary" or "lymph_node".
_C.DATASET.REGION = "primary"
# Patients to fetch from a remote source; 0 fetches the whole cohort.
_C.DATASET.PATIENTS = 0
# Cache for fetched data. Empty uses ~/.cache/kalecancer.
_C.DATASET.CACHE_DIR = ""
# Read every file's header at start-up to reject invalid bags before training.
_C.DATASET.VALIDATE_FEATURES = True
# Cap on patches per bag during training, bounding memory for large slides.
# Evaluation and interpretation always use the full bag. 0 disables the cap.
_C.DATASET.MAX_PATCHES = 2048
_C.DATASET.NUM_WORKERS = 4
# Share of the cohort held out for validation and testing; training takes the rest.
_C.DATASET.VAL_RATIO = 0.15
_C.DATASET.TEST_RATIO = 0.15
# Number of cross-validation folds. 0 uses the ratios above instead.
_C.DATASET.NUM_FOLDS = 0
# Cohort columns whose distribution is preserved across splits. Samples sharing a
# patient are always kept in one split regardless of this setting.
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
# Applies to multimodal pipelines; see kalecancer.model.multimodal.
# ---------------------------------------------------------------------------
_C.FUSION = CfgNode()
# What gets combined: "early" (features), "late" (decisions), or "hybrid" (both).
_C.FUSION.STRATEGY = "hybrid"
# Feature-fusion operator, used by "early" and "hybrid": "concat", "poe", or "lowrank".
_C.FUSION.METHOD = "concat"
_C.FUSION.FUSED_DIM = 64
# Rank of the factorisation, for METHOD "lowrank" only.
_C.FUSION.RANK = 4
# Hybrid only: a survival head on each modality latent.
_C.FUSION.AUXILIARY_HEADS = True
_C.FUSION.AUXILIARY_WEIGHT = 0.3
# Hybrid only: blend per-modality risks into the prediction as well as the features.
_C.FUSION.COMBINE_RISKS = False
# Chance of dropping each present modality during training.
_C.FUSION.MODALITY_DROPOUT = 0.0

# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------
_C.SURVIVAL = CfgNode()
# Endpoint key; see kalecancer.loaddata.clinical_access.ENDPOINTS.
_C.SURVIVAL.ENDPOINT = "OS"
# Tie handling in the Cox partial likelihood: "efron" or "breslow".
_C.SURVIVAL.TIES = "efron"
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
