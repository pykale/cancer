"""Post-prediction interpretation: SHAP, Grad-CAM, and modality contribution."""

from kalecancer.interpret.attention import (
    attention_records,
    collect_attention,
    export_attention,
    multimodal_attention,
    top_k_patches,
)
from kalecancer.interpret.embedding import umap_embedding

__all__ = [
    "attention_records",
    "collect_attention",
    "export_attention",
    "multimodal_attention",
    "top_k_patches",
    "umap_embedding",
]
