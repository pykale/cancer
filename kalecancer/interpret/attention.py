"""Attention export: per-instance attention weights joined with the modality's description of each instance."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from torch import Tensor

from kalecancer.loaddata.dataset import MultimodalDataset
from kalecancer.loaddata.modalities import BagModality
from kalecancer.model.encode.base_encoders import AttentionEncoder

if TYPE_CHECKING:
    from kalecancer.model.encode.encoder_list import EncoderList
    from kalecancer.pipeline.pipeline import Pipeline


def attention(pipeline: Pipeline, data: MultimodalDataset, modality: str, branch: str | None = None) -> pd.DataFrame:
    """Per-instance attention weights of the modality's ``AttentionEncoder``, joined with its ``instances``.

    The encoders before it must keep each bag's instances one to one with the loaded rows.
    """
    if modality not in data.modalities:
        raise ValueError(f"the dataset has no modality {modality!r}")
    source = data.modalities[modality]
    if not isinstance(source, BagModality):
        raise TypeError(f"Attention export needs a BagModality. '{modality}' is a {type(source).__name__}")

    def step(encoders: EncoderList, x: Any, ids: list[str]) -> list[Tensor]:
        hits = [k for k, encoder in enumerate(encoders) if isinstance(encoder, AttentionEncoder)]
        if len(hits) != 1:
            raise ValueError(f"{modality!r} needs exactly one AttentionEncoder, found {len(hits)}")
        attending: AttentionEncoder = encoders[hits[0]]
        # run only the encoders before the attending one, checking that each keeps every bag's instances
        if hits[0] > 0:
            for k, encoder, out in encoders.steps(x, ids):
                if [len(bag) for bag in out] != [len(bag) for bag in x]:
                    raise ValueError(
                        f"encoders[{modality!r}][{k}] {type(encoder).__name__} changes the number of instances in a "
                        "bag, so attention weights cannot be matched to its instances; sample instances only in "
                        "training mode, or select them in the modality"
                    )
                x = out
                if k == hits[0] - 1:
                    break
        return [w.float().cpu() for w in attending.attention(x)]

    frames = []
    for ids, weights in pipeline.run_encoders(data, modality, step, branch):
        for pid, w in zip(ids, weights, strict=True):
            rows = source.instances(pid)
            if w.ndim != 1:
                raise NotImplementedError(f"{pid}: attention export reads (N,) weights, got {tuple(w.shape)}")
            if len(rows) != w.shape[-1]:
                raise ValueError(f"{pid}: {w.shape[-1]} attention weights for {len(rows)} instances")
            frames.append(rows.assign(attention=w.numpy()).assign(patient_id=pid))
    table = pd.concat(frames, ignore_index=True)
    return table[["patient_id", *[c for c in table.columns if c not in ("patient_id", "attention")], "attention"]]
