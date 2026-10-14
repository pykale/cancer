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
from kalecancer.interpret.modality import leave_one_out_ablation, modality_shapley

__all__ = [
    "attention_records",
    "bag_attention",
    "batch_records",
    "export_attention",
    "feature_shapley",
    "leave_one_out_ablation",
    "modality_shapley",
    "permutation_importance",
    "top_k_patches",
    "umap_embedding",
]
