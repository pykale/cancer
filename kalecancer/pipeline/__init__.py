"""Training, as two independent choices.

:class:`CohortTrainer` is the only trainer. What it trains on is a set of modality
embedders; what it trains *for* is a :class:`PredictionTask`. Neither constrains the
other, so the same class covers every experiment here::

    CohortTrainer({"wsi": BagEncoder(AttentionMIL(...))}, task=SurvivalTask())
    CohortTrainer({"clinical": MLPEmbedder(...)}, task=SurvivalTask())
    CohortTrainer(both_of_those, task=ClassificationTask(pos_weight=3.0))

A task decides exactly four things -- the head, the loss, whether a batch carries a
gradient, and the epoch metric -- which is why adding an endpoint means adding a
task rather than another trainer.

Orchestration is deliberately absent: how a cohort is assembled, split and reported
on belongs to an experiment, so it lives in ``examples/`` rather than here.
"""

from kalecancer.pipeline.task import ClassificationTask, PredictionTask, SurvivalTask
from kalecancer.pipeline.trainer import CohortTrainer

__all__ = ["ClassificationTask", "CohortTrainer", "PredictionTask", "SurvivalTask"]
