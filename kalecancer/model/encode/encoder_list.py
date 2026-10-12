from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from kalecancer.loaddata.targets import BaseTarget
from kalecancer.model.encode.base_encoders import FittedEncoder


class EncoderList(nn.ModuleList):
    def __init__(self, encoders: Sequence[nn.Module], name: str):
        if not isinstance(encoders, Sequence):
            raise TypeError(
                f"'encoders' for {name!r} must be a list of encoders (nn.Module), got {type(encoders).__name__}"
            )
        encoders = list(encoders)
        for k, encoder in enumerate(encoders):
            if not isinstance(encoder, nn.Module):
                raise TypeError(f"encoders[{name!r}][{k}] must be an nn.Module, got {type(encoder).__name__}")
            if k > 0 and isinstance(encoder, FittedEncoder):
                raise ValueError(
                    f"encoders[{name!r}][{k}] {type(encoder).__name__} is a FittedEncoder; "
                    "only the first encoder may be fitted"
                )
        super().__init__(encoders)
        self.name = name

    @property
    def needs_fit(self) -> bool:
        """Whether the first encoder is a ``FittedEncoder``. Models load the training rows for ``fit`` only then,
        since loading every training bag is costly."""
        return bool(self) and isinstance(self[0], FittedEncoder)

    def fit(self, x: Any, target: BaseTarget, ids: list[str]) -> None:
        """Fit the first encoder, if it is a ``FittedEncoder``.

        Args:
            x: The loaded, transformed input of the training patients ``ids``, as a batch of them would be.
            target: The training dataset's target.
            ids: The training patients who have the modality, in the order of ``x``.
        """
        if self.needs_fit:
            self[0].fit(x, target, ids)

    def steps(self, x: Any, ids: list[str]) -> Iterator[tuple[int, nn.Module, Any]]:
        """Run the encoders in order, yielding ``(index, encoder, output)`` after each one.

        Every caller runs the encoders through here, so a ``FittedEncoder`` always receives ``ids`` and an encoder's
        error always gets a note naming the encoder and its input. Stopping the iteration early skips the encoders
        after that point, which is how attention export reaches the input of the attending encoder.
        """
        for k, encoder in enumerate(self):
            try:
                out = encoder(x, ids) if isinstance(encoder, FittedEncoder) else encoder(x)
            except Exception as error:
                error.add_note(f"in encoders[{self.name!r}][{k}] {type(encoder).__name__}, called on {_describe(x)}")
                raise
            yield k, encoder, out
            x = out

    def forward(self, x: Any, ids: list[str]) -> Tensor:
        """Run the encoders in order, then check the embedding: a floating-point ``(len(ids), d)`` tensor, finite for
        every patient."""
        inputs = x
        for _, _, out in self.steps(inputs, ids):
            x = out
        if not isinstance(x, Tensor):
            raise TypeError(f"encoders[{self.name!r}] must output an (n, d) tensor, got {_describe(x)}")
        if x.ndim != 2:
            raise ValueError(f"encoders[{self.name!r}] must output an (n, d) tensor, got shape {tuple(x.shape)}")
        if x.shape[0] != len(ids):
            raise ValueError(
                f"encoders[{self.name!r}] must output one row per patient: {len(ids)} patients, {x.shape[0]} rows"
            )
        if not x.is_floating_point():
            raise TypeError(f"encoders[{self.name!r}] must output a floating-point tensor, got {x.dtype}")
        finite = torch.isfinite(x).all(dim=1)
        if not finite.all():
            bad = [pid for pid, ok in zip(ids, finite.tolist(), strict=True) if not ok]
            raise ValueError(
                f"encoders[{self.name!r}] output NaN or infinite values for {len(bad)} patient(s) (e.g. {bad[:5]}). "
                + _non_finite_cause(inputs, ids)
            )
        return x


def _describe(x: Any) -> str:
    """Describe an encoder's input for the note added to that encoder's errors."""
    if isinstance(x, Tensor):
        return f"a tensor of shape {tuple(x.shape)}"
    if isinstance(x, list) and all(isinstance(item, Tensor) for item in x):
        shapes = ", ".join(str(tuple(item.shape)) for item in x[:3]) + (", ..." if len(x) > 3 else "")
        return f"a list of {len(x)} tensor{'' if len(x) == 1 else 's'} shaped {shapes}"
    return f"a {type(x).__name__}"


def _non_finite_cause(inputs: Any, ids: list[str]) -> str:
    """Say whether a non-finite embedding comes from the data or from the encoders.

    Looks at every patient in the batch, not only those with a non-finite embedding: an encoder that mixes patients,
    such as BatchNorm in training mode, spreads one patient's NaN to all of them.
    """
    if isinstance(inputs, Tensor):
        finite = torch.isfinite(inputs).reshape(len(inputs), -1).all(dim=1).tolist()
    elif isinstance(inputs, list | tuple) and all(isinstance(item, Tensor) for item in inputs):
        finite = [bool(torch.isfinite(item).all()) for item in inputs]
    else:
        return "Their input could not be inspected."
    if bad := [pid for pid, ok in zip(ids, finite, strict=True) if not ok]:
        return (
            f"The input already has NaN or infinite values for {len(bad)} patient(s) (e.g. {bad[:5]}): impute them "
            "or use an encoder that handles them."
        )
    return "The input was finite, so the encoders produced these values."
