"""Model: encoders, fusion methods, endpoints, typed outputs, and the fusion models that wire them together."""

from kalecancer.model.baseline import BreslowBaseline
from kalecancer.model.encode.base_encoders import AttentionEncoder, FittedEncoder
from kalecancer.model.encode.encoder_list import EncoderList
from kalecancer.model.encode.encoders import ABMIL, MLP
from kalecancer.model.endpoints import BaseEndpoint, ClassificationEndpoint, CoxEndpoint
from kalecancer.model.fusion import (
    Concat,
    FusionMethod,
    MajorityVote,
    MaskedMean,
    OutputFusion,
    TensorFusion,
)
from kalecancer.model.heads import LinearClassifier
from kalecancer.model.models import (
    BaseModel,
    EarlyFusion,
    IntermediateFusion,
    LateFusion,
    SingleEndpointModel,
    Unimodal,
)
from kalecancer.model.outputs import (
    BaseOutput,
    ClassProbabilities,
    ClassScores,
    CoxOutput,
    RiskScore,
    SurvivalFunction,
    VoteScores,
)

__all__ = [
    "ABMIL",
    "MLP",
    "AttentionEncoder",
    "BaseEndpoint",
    "BaseModel",
    "BaseOutput",
    "BreslowBaseline",
    "ClassProbabilities",
    "ClassScores",
    "ClassificationEndpoint",
    "Concat",
    "CoxEndpoint",
    "CoxOutput",
    "EncoderList",
    "EarlyFusion",
    "FusionMethod",
    "FittedEncoder",
    "IntermediateFusion",
    "LateFusion",
    "LinearClassifier",
    "MajorityVote",
    "MaskedMean",
    "OutputFusion",
    "RiskScore",
    "SingleEndpointModel",
    "SurvivalFunction",
    "TensorFusion",
    "Unimodal",
    "VoteScores",
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
