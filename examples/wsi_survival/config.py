"""Default configuration for the WSI primary-tumour survival example.

Values here are defaults for the HANCOCK head-and-neck cohort. Override them on the
command line or with ``--cfg <file>.yaml``; no path belongs in the library itself.
"""

from yacs.config import CfgNode

_C = CfgNode()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
_C.DATASET = CfgNode()
# Directory of pre-extracted UNI patch features, searched recursively for .h5 files.
_C.DATASET.FEATURE_ROOT = "E:/WSI_UNI_encodings/WSI_PrimaryTumor"
# JSON file of clinical records, one object per patient.
_C.DATASET.CLINICAL_PATH = "E:/StructuredData/StructuredData/clinical_data.json"
# Read every file's header at start-up to reject invalid bags before training.
_C.DATASET.VALIDATE_FEATURES = True
# Cap on patches per bag during training, bounding memory for large slides.
# Evaluation and interpretation always use the full bag. 0 disables the cap.
_C.DATASET.MAX_PATCHES = 2048
_C.DATASET.NUM_WORKERS = 4
_C.DATASET.TRAIN_RATIO = 0.7
_C.DATASET.VAL_RATIO = 0.15
_C.DATASET.TEST_RATIO = 0.15
# Number of patient-level cross-validation folds. 0 uses the ratios above instead.
_C.DATASET.NUM_FOLDS = 0

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
_C.MODEL = CfgNode()
# UNI patch embeddings are 1024-dimensional.
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

# ---------------------------------------------------------------------------
# Fusion
#
# Unused by this unimodal example; these keys define the contract for the
# multimodal pipeline so a researcher can switch strategy and fusion method from
# configuration alone. See kalecancer.model.multimodal.
# ---------------------------------------------------------------------------
_C.FUSION = CfgNode()
# Where the modalities meet: "early", "late", or "hybrid".
_C.FUSION.STRATEGY = "hybrid"
# Latent fusion operator used by the hybrid strategy: "concat", "poe", or "lowrank".
# "poe" is the route to prefer when modalities may be missing at inference.
_C.FUSION.METHOD = "concat"
# Width of the fused representation passed to the survival head.
_C.FUSION.FUSED_DIM = 64
# Rank of the factorisation, for METHOD "lowrank" only.
_C.FUSION.RANK = 4
# Attach a survival head to each modality latent, keeping every encoder supervised.
_C.FUSION.AUXILIARY_HEADS = True
# Weight on the mean per-modality auxiliary loss. 0 trains the fused head alone.
_C.FUSION.AUXILIARY_WEIGHT = 0.3
# Chance of dropping each present modality during training, for robustness to
# missing modalities at inference.
_C.FUSION.MODALITY_DROPOUT = 0.0

# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------
_C.SURVIVAL = CfgNode()
# Endpoint key, see kalecancer.loaddata.clinical_access.ENDPOINTS.
_C.SURVIVAL.ENDPOINT = "OS"
# Tie handling in the Cox partial likelihood: "efron" or "breslow".
_C.SURVIVAL.TIES = "efron"
# Horizons for time-dependent metrics, in the unit of the survival time (days here):
# 1, 3 and 5 years.
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
