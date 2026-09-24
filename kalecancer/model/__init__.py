"""Model: encoders, fusion methods, prediction heads, and the fusion models that wire them together."""

from typing import TYPE_CHECKING

from kalecancer.model.encoders import ABMIL, MLP
from kalecancer.model.fusion import Concat, MajorityVote, MaskedMean, MeanLogits
from kalecancer.model.heads import ClassificationHead, CoxHead
from kalecancer.model.incontext import InContextModule
from kalecancer.model.models import EarlyFusion, IntermediateFusion, LateFusion, ModelOutput, StageList, Unimodal

if TYPE_CHECKING:
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


def __getattr__(name: str):
    # tabicl is an optional extra
    # import it on first use so the rest of the package works without it
    if name == "TabICLEncoder":
        try:
            from kalecancer.model.tabicl import TabICLEncoder
        except ImportError as error:
            raise ImportError(
                "TabICLEncoder needs the optional tabicl package: pip install 'kalecancer[tabular]'"
            ) from error
        return TabICLEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
