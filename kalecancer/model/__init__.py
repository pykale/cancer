"""Model: encoders, fusion methods, prediction heads, and the fusion models that wire them together."""

from kalecancer.model.baseline import BreslowBaseline
from kalecancer.model.encode.base_encoders import AttentionEncoder, FittedEncoder
from kalecancer.model.encode.encoder_list import EncoderList
from kalecancer.model.encode.encoders import ABMIL, MLP
from kalecancer.model.fusion import Concat, FusionContext, FusionMethod, MajorityVote, MaskedMean
from kalecancer.model.heads import ClassificationHead, CoxHead
from kalecancer.model.models import EarlyFusion, IntermediateFusion, LateFusion, ModelOutput, Unimodal

__all__ = [
    "ABMIL",
    "MLP",
    "AttentionEncoder",
    "BreslowBaseline",
    "ClassificationHead",
    "Concat",
    "EncoderList",
    "CoxHead",
    "EarlyFusion",
    "FusionContext",
    "FusionMethod",
    "FittedEncoder",
    "IntermediateFusion",
    "LateFusion",
    "MajorityVote",
    "MaskedMean",
    "ModelOutput",
    "Unimodal",
]


def __getattr__(name: str):
    # tabicl is an optional extra
    # import it on first use so the rest of the package works without it
    if name == "TabICLEncoder":
        try:
            from kalecancer.model.encode.tabicl import TabICLEncoder
        except ImportError as error:
            raise ImportError(
                "TabICLEncoder needs the optional tabicl package: pip install 'kalecancer[tabular]'"
            ) from error
        return TabICLEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
