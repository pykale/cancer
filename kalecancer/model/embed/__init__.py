"""Modality embedders: one modality in, one vector per sample out."""

from kalecancer.model.embed.protocols import Embedder
from kalecancer.model.embed.tabicl import TabICLEmbedder

__all__ = ["Embedder", "TabICLEmbedder"]
