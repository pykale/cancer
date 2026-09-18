"""Model: encoders, fusion methods, prediction heads, and the fusion models that wire them together."""

from kalecancer.model.encoders import ABMIL, MLP
from kalecancer.model.fusion import Concat, MajorityVote, MaskedMean, MeanLogits
from kalecancer.model.heads import ClassificationHead, CoxHead
from kalecancer.model.incontext import InContextModule
from kalecancer.model.models import EarlyFusion, IntermediateFusion, LateFusion, ModelOutput, StageList, Unimodal
from kalecancer.model.tabicl import TabICLEncoder

__all__ = [
    "ABMIL",
    "MLP",
    "ClassificationHead",
    "Concat",
    "CoxHead",
    "EarlyFusion",
    "InContextModule",
    "IntermediateFusion",
    "LateFusion",
    "MajorityVote",
    "MaskedMean",
    "MeanLogits",
    "ModelOutput",
    "StageList",
    "TabICLEncoder",
    "Unimodal",
]
