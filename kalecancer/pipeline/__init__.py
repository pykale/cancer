"""Pipeline: train a model end to end with Lightning, predict, and build pipelines from configs."""

from kalecancer.pipeline.config import dump_config, load_pipeline
from kalecancer.pipeline.pipeline import EarlyStopping, Pipeline

__all__ = ["EarlyStopping", "Pipeline", "dump_config", "load_pipeline"]
