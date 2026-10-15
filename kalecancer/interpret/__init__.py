"""Post-prediction interpretation of what a model attended to."""

from kalecancer.interpret.attention import (
    attention_records,
    bag_attention,
    batch_records,
    export_attention,
    top_k_patches,
)
from kalecancer.interpret.attribution import feature_shapley, permutation_importance
from kalecancer.interpret.embedding import umap_embedding
from kalecancer.interpret.heatmap import attention_heatmap, overlay_on_slide, save_heatmap
from kalecancer.interpret.modality import leave_one_out_ablation, modality_shapley
from kalecancer.interpret.report import PatientReport, patient_report, save_reports

__all__ = [
    "PatientReport",
    "attention_heatmap",
    "attention_records",
    "bag_attention",
    "batch_records",
    "export_attention",
    "feature_shapley",
    "leave_one_out_ablation",
    "modality_shapley",
    "overlay_on_slide",
    "patient_report",
    "permutation_importance",
    "save_heatmap",
    "save_reports",
    "top_k_patches",
    "umap_embedding",
]
