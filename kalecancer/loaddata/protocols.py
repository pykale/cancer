"""Contracts the data layer requires of its collaborators.

This module imports nothing from ``kalecancer``, so ``survival/`` can implement
:class:`Target` and ``prepdata/`` can implement :class:`Preprocessor` without
depending on the data layer.

Protocols rather than ABCs so implementations need no import at all -- which is
what keeps ``SurvivalTarget`` liftable into PyKale core. Not ``@runtime_checkable``:
that only compares method names, never signatures. :func:`check_target` is the real
check.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from numpy.typing import ArrayLike
from torch import Tensor


class Target(Protocol):
    """What a cohort requires of a supervision target.

    Bound once, at index-loading time, and keyed by identifier rather than row
    position, so a cohort can be subset and recombined without realignment.

    Anything derived from a *training fold* is not a target and must not live here.
    Discrete-time bin edges are the case to watch: they are quantiles of one fold's
    event times, so they belong in ``prepdata`` under the same fold-local discipline
    as a scaler.
    """

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Columns this target needs, so a cohort can check them before binding."""
        ...

    def bind(self, identifiers: Sequence[str], values: Mapping[str, ArrayLike]) -> None:
        """Extract and validate supervision values, keyed by identifier.

        Takes arrays rather than a ``DataFrame`` to keep pandas out of the contract.
        Validate here and raise; do not defer a mis-specified target to training.

        Args:
            identifiers (Sequence[str]): Sample identifiers, in row order.
            values (Mapping[str, ArrayLike]): One array per entry in
                :attr:`required_columns`, aligned with ``identifiers``.
        """
        ...

    def for_(self, identifier: str) -> dict[str, Tensor]:
        """This sample's values for ``PatientSample.target``.

        Named keys, never a positional pack: a ``tensor([time, event])`` built
        backwards runs perfectly and predicts survival inverted.
        """
        ...


class Preprocessor(Protocol):
    """Fitted state belonging to exactly one cross-validation fold.

    Deliberately thin -- a cohort defines and consumes its own preprocessor type.
    What every preprocessor owes the rest of the system is provenance, so a fold's
    held-out rows can be proven untouched.
    """

    @property
    def fitted_on(self) -> frozenset[str]:
        """Identifiers this was fitted on.

        Identifiers rather than positions, so the check survives subsetting and
        multimodal composition.
        """
        ...

    @property
    def feature_names(self) -> dict[str, list[str]]:
        """Post-encoding feature names, keyed by modality.

        Keyed because a composite preprocessor serves several modalities at once.
        A modality whose features have no meaningful names returns an empty list.
        """
        ...

    def describe(self) -> str:
        """Human-readable summary of what this fold applied."""
        ...


def check_target(target: object) -> None:
    """Verify an object satisfies :class:`Target`, or raise naming what is missing.

    Raises:
        TypeError: If any part of the contract is absent.
    """
    missing = [name for name in ("required_columns", "bind", "for_") if not hasattr(target, name)]
    if missing:
        raise TypeError(
            f"{type(target).__name__} is not a valid Target: missing {missing}. "
            f"A target must declare required_columns, and implement "
            f"bind(identifiers, values) and for_(identifier). See "
            f"kalecancer.loaddata.protocols.Target."
        )
