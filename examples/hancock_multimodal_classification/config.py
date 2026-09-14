"""Configuration for the HANCOCK outcome-classification example.

Extends the package defaults with what this experiment adds: which endpoint to
predict, which published split to use, and how the structured table is encoded.
Selecting the endpoint, the split and the modalities is what turns one code path into
the whole comparison matrix.

The column roles below follow Dörrich et al. (2025), Methods, "Multimodal patient
vectors". Their vector also carries 40 ICD bag-of-words columns and 4 TMA cell
densities, which come from archives outside ``StructuredData.zip`` and are out of
scope here.
"""

from yacs.config import CfgNode

from kalecancer.config import get_cfg_defaults as _package_defaults

#: Endpoints the example can predict, and the published splits it can use.
TARGETS = ("survival_status", "recurrence")
SPLIT_FILES = ("dataset_split_in.json", "dataset_split_out.json", "dataset_split_Oropharynx.json")


def get_cfg_defaults() -> CfgNode:
    """Return the package defaults extended for this experiment."""
    cfg = _package_defaults()

    cfg.DATASET.SOURCE = "hancock"
    # Which modalities take part. One name gives a unimodal baseline; two are fused.
    cfg.DATASET.MODALITIES = ["tabular", "imaging"]
    # Every arm is scored on the patients that have slide features, so the three
    # modality settings are comparable. Without this the tabular arm would be scored
    # on a larger cohort than the imaging arm and the AUCs would not be commensurable.
    cfg.DATASET.REQUIRE_MODALITIES = ["imaging"]
    cfg.DATASET.SPLIT_FILE = SPLIT_FILES[0]
    cfg.DATASET.NUM_WORKERS = 0
    cfg.DATASET.MAX_PATCHES = 512

    # ---------------------------------------------------------------- endpoint
    cfg.CLASSIFY = CfgNode()
    cfg.CLASSIFY.TARGET = TARGETS[0]
    # Repeats differing only in seed, matching the paper's five iterations per split.
    cfg.CLASSIFY.REPEATS = 5
    # Deaths that cannot be attributed to the tumour carry no usable label for a
    # disease-specific endpoint, so they are dropped rather than called negative.
    cfg.CLASSIFY.EXCLUDE_STATUS = ["deceased not tumor specific"]
    cfg.CLASSIFY.RECURRENCE_HORIZON_DAYS = 1095
    # Weight on the positive class, standing in for the paper's SMOTE. 0 derives it
    # from the training split's class balance; a negative value disables it.
    cfg.CLASSIFY.POS_WEIGHT = 0.0

    # ------------------------------------------------------------ structured data
    cfg.TABULAR = CfgNode()
    cfg.TABULAR.BINARY = [
        "sex",
        "primarily_metastasis",
        "lymphovascular_invasion_L",
        "vascular_invasion_V",
        "perineural_invasion_Pn",
        "perinodal_invasion",
        "carcinoma_in_situ",
    ]
    cfg.TABULAR.NOMINAL = [
        "smoking_status",
        "primary_tumor_site",
        "histologic_type",
        "hpv_association_p16",
        "grading",
        "resection_status",
        "resection_status_carcinoma_in_situ",
    ]
    cfg.TABULAR.DISCRETE = [
        "age_at_initial_diagnosis",
        "number_of_positive_lymph_nodes",
        "infiltration_depth_in_mm",
    ]
    cfg.TABULAR.ORDINAL = ["pT_stage", "pN_stage"]
    cfg.TABULAR.USE_BLOOD = True
    cfg.TABULAR.PROJECTION_HIDDEN = [128]

    # ---------------------------------------------------------------- baseline
    cfg.BASELINE = CfgNode()
    cfg.BASELINE.ENABLED = True
    cfg.BASELINE.N_ESTIMATORS = 500
    # Class weighting rather than the paper's SMOTE, which would add a dependency for
    # a baseline. Both counter imbalance; they are not equivalent.
    cfg.BASELINE.CLASS_WEIGHT = "balanced"

    # Fig 2B-D. Needs umap-learn, which ships in the `interpret` extra.
    cfg.OUTPUT.UMAP = True
    cfg.OUTPUT.TOP_K = 20
    cfg.OUTPUT.OUT_DIR = "outputs/hancock_multimodal_classification"
    return cfg
