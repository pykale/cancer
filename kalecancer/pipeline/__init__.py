"""End-to-end trainers combining the pipeline stages for a specific task."""

from kalecancer.pipeline.multimodal_classifier import MultimodalClassificationTrainer
from kalecancer.pipeline.multimodal_trainer import MultimodalSurvivalTrainer
from kalecancer.pipeline.wsi_survival_trainer import WSISurvivalTrainer

__all__ = ["MultimodalClassificationTrainer", "MultimodalSurvivalTrainer", "WSISurvivalTrainer"]
