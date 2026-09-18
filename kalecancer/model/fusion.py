"""Fusion methods for early and intermediate fusion, and combiners for late fusion."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kalecancer.loaddata.targets import TargetInfo

with warnings.catch_warnings():
    # torchsurv applies torch.jit.script when imported, which recent torch deprecates; users cannot act on it
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.* is deprecated", category=FutureWarning)


def _stack_defined(values: Mapping[str, Tensor], defined: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
    stacked = torch.stack([values[name] for name in values])
    mask = torch.stack([defined[name] for name in values])
    return stacked, mask


def _masked_mean(stacked: Tensor, mask: Tensor) -> Tensor:
    # where() rather than multiplication: absent rows hold NaN, and NaN * 0 is NaN
    filled = torch.where(mask.unsqueeze(-1), stacked, torch.zeros((), dtype=stacked.dtype, device=stacked.device))
    count = mask.sum(dim=0).clamp(min=1).unsqueeze(-1).to(stacked.dtype)
    return filled.sum(dim=0) / count


class Concat(nn.Module):
    """Concatenate modality vectors. Every patient must have every modality."""

    handles_missing = False

    def output_dim(self, widths: Mapping[str, int]) -> int:
        return sum(widths.values())

    def forward(self, z: Mapping[str, Tensor], present: Mapping[str, Tensor]) -> Tensor:
        for name, mask in present.items():
            if not bool(mask.all()):
                raise ValueError(
                    f"Concat: {int((~mask).sum())} patients lack {name!r}; require it in required_modalities "
                    "or use a fusion method that handles missing modalities"
                )
        return torch.cat([z[name] for name in z], dim=-1)


class MaskedMean(nn.Module):
    """Mean of equal-width modality vectors over the modalities each patient has."""

    handles_missing = True

    def output_dim(self, widths: Mapping[str, int]) -> int:
        if len(set(widths.values())) != 1:
            raise ValueError(f"MaskedMean needs equal widths, got {dict(widths)}")
        return next(iter(widths.values()))

    def forward(self, z: Mapping[str, Tensor], present: Mapping[str, Tensor]) -> Tensor:
        return _masked_mean(*_stack_defined(z, present))


class MeanLogits(nn.Module):
    """Late fusion: mean of the branch head outputs (logits or log-hazards) over the branches each patient has."""

    handles_missing = True
    input_space = "output"

    def check_branches(self, heads: Mapping[str, nn.Module], complete: bool) -> None:
        kinds = {getattr(head, "target_kind", None) for head in heads.values()}
        if len(kinds) != 1:
            raise TypeError(f"MeanLogits needs branch heads of one kind, got {kinds}")
        if not complete and kinds == {"time_to_event"}:
            raise ValueError(
                "MeanLogits over Cox heads needs every patient to have every branch: each branch's log-hazard "
                "has an arbitrary offset, so averaging different subsets of branches would reorder patients"
            )

    def forward(self, outputs: Mapping[str, Tensor], defined: Mapping[str, Tensor]) -> Tensor:
        return _masked_mean(*_stack_defined(outputs, defined))


class MajorityVote(nn.Module):
    """Late fusion: each branch votes for its most probable class.

    Scores are ``(votes_c + 0.5 * mean_probability_c) / (n_branches + 0.5)``: they sum to 1, their argmax is the
    majority class, and ties between classes with equal votes are broken by mean probability. The probability
    term can never outweigh a whole vote.
    """

    handles_missing = True
    input_space = "prediction"

    def __init__(self, tie_break: Literal["mean_probability", "error"]):
        super().__init__()
        if tie_break not in ("mean_probability", "error"):
            raise ValueError(f"tie_break must be 'mean_probability' or 'error', got {tie_break!r}")
        self.tie_break = tie_break

    def check_branches(self, heads: Mapping[str, nn.Module], complete: bool) -> None:
        kinds = {getattr(head, "target_kind", None) for head in heads.values()}
        if kinds != {"classification"}:
            raise TypeError(f"MajorityVote needs classification heads, got {kinds}")

    def columns(self, info: TargetInfo) -> list[str]:
        # assert is temporary fix to keep mypy quiet
        # real fix requires rethinking TargetInfo
        assert info.classes is not None, "classification targets always carry classes"

        return [f"vote_score[{c}]" for c in info.classes]

    def forward(self, probabilities: Mapping[str, Tensor], defined: Mapping[str, Tensor]) -> Tensor:
        stacked, mask = _stack_defined(probabilities, defined)
        n_classes = stacked.shape[-1]
        choice = torch.nan_to_num(stacked, nan=-1.0).argmax(dim=-1)
        votes = (F.one_hot(choice, n_classes).to(stacked.dtype) * mask.unsqueeze(-1)).sum(dim=0)
        n_branches = mask.sum(dim=0).to(stacked.dtype).unsqueeze(-1)
        if self.tie_break == "error":
            top = votes.max(dim=-1, keepdim=True).values
            tied = ((votes == top).sum(dim=-1) > 1) & (n_branches.squeeze(-1) > 0)
            if bool(tied.any()):
                raise ValueError(f"MajorityVote: {int(tied.sum())} patients have tied votes and tie_break='error'")
        return (votes + 0.5 * _masked_mean(stacked, mask)) / (n_branches + 0.5)
