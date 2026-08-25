"""Post-prediction interpretation: SHAP, Grad-CAM, and modality contribution."""

from kalecancer.interpret.attention import (
    attention_records,
    collect_attention,
    export_attention,
    top_k_patches,
)

__all__ = ["attention_records", "collect_attention", "export_attention", "top_k_patches"]
