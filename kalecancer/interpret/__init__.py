"""Interpret: attention export, heatmaps, and feature and modality attribution, from fitted pipelines."""

from kalecancer.interpret.attention import attention
from kalecancer.interpret.attribution import feature_shapley, permutation_importance
from kalecancer.interpret.heatmap import attention_heatmap, overlay_on_slide, save_heatmap
from kalecancer.interpret.modality import leave_one_out_ablation, modality_shapley

__all__ = [
    "attention",
    "attention_heatmap",
    "feature_shapley",
    "leave_one_out_ablation",
    "modality_shapley",
    "overlay_on_slide",
    "permutation_importance",
    "save_heatmap",
]
