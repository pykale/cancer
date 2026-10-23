"""Interpret: attention export from fitted pipelines, heatmaps rendered from it, and feature attribution."""

from kalecancer.interpret.attention import attention
from kalecancer.interpret.attribution import feature_shapley, permutation_importance
from kalecancer.interpret.heatmap import attention_heatmap, overlay_on_slide, save_heatmap

__all__ = [
    "attention",
    "attention_heatmap",
    "feature_shapley",
    "overlay_on_slide",
    "permutation_importance",
    "save_heatmap",
]
