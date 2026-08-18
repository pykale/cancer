"""Contracts the data layer requires of its collaborators.

This module imports nothing from ``kalecancer``. That is the whole point: it is a
leaf in the import graph, so ``survival/`` can implement :class:`Target` and
``prepdata/`` can implement :class:`Preprocessor` without either of them importing
the data layer, and without the data layer importing them. The dependency arrow
points in one direction and there is nothing for it to point at here.

Both are :class:`~typing.Protocol` rather than abstract base classes, deliberately.
An ABC would work and would give better ``isinstance`` behaviour, but it would put
an import edge on every implementation -- and ``SurvivalTarget`` is expected to move
into PyKale core, where an import of ``kalecancer.loaddata`` would have to move with
it. Structural typing costs nothing and keeps that door open.

Neither is ``@runtime_checkable``. That decorator only verifies method *names*
exist, not their signatures, so an object with a ``bind`` taking entirely the wrong
arguments passes the check. That is false confidence, which is worse than no check.
The contract is enforced where it is used, loudly, with a message naming what is
missing -- see :func:`check_target`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from numpy.typing import ArrayLike
from torch import Tensor


class Target(Protocol):
    """What a cohort requires of a supervision target.

    A target is bound **once**, at index-loading time, and is keyed by identifier
    rather than by row position. That is what lets a cohort be subset, split and
    recombined without the target ever needing realignment.

    The corollary is that anything derived from a *training fold* is not a target
    and must not live here. Discrete-time survival bin edges are the case to watch:
    they are quantiles of one fold's uncensored event times, so they are fitted
    state belonging in ``prepdata`` under the same fold-local discipline as a
    scaler. Putting them on a target leaks the held-out fold's time distribution,
    silently, because it never looks like preprocessing.
    """

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Columns this target needs from the source table.

        Declared rather than discovered so a cohort can check them against the
        table it just read and fail at construction, naming the missing column,
        instead of failing at training time or -- worse -- not at all.
        """
        ...

    def bind(self, identifiers: Sequence[str], values: Mapping[str, ArrayLike]) -> None:
        """Extract and validate supervision values, keyed by identifier.

        Takes arrays rather than a ``DataFrame``: pandas has no business in a
        contract that ``survival/`` implements and that a non-tabular cohort may
        one day have to satisfy.

        Implementations should validate here and raise, rather than deferring a
        mis-specified target to training time.

        Args:
            identifiers (Sequence[str]): Sample identifiers, in row order.
            values (Mapping[str, ArrayLike]): One array per entry in
                :attr:`required_columns`, aligned with ``identifiers``.
        """
        ...

    def for_(self, identifier: str) -> dict[str, Tensor]:
        """This sample's supervision values, for ``PatientSample.target``.

        Returns:
            dict[str, Tensor]: Named tensors, e.g. ``{"time": ..., "event": ...}``.
            Named, never a positional pack: a ``tensor([time, event])`` that gets
            built backwards runs perfectly and predicts survival inverted.
        """
        ...


class Preprocessor(Protocol):
    """Fitted state belonging to exactly one cross-validation fold.

    Deliberately thin. A cohort defines and consumes its own preprocessor type --
    a tabular cohort cannot use a slide preprocessor, and there is no value in
    pretending a shared ``transform`` signature could serve both. What every
    preprocessor *does* owe the rest of the system is provenance, so that
    :class:`~kalecancer.loaddata.module.CohortDataModule` can prove a fold's
    held-out rows were never fitted on.
    """

    @property
    def fitted_on(self) -> frozenset[str]:
        """Identifiers this was fitted on.

        Identifiers rather than positional indices, so the check survives
        subsetting, recombination and multimodal composition.
        """
        ...

    @property
    def feature_names(self) -> dict[str, list[str]]:
        """Post-encoding feature names, **keyed by modality**.

        Keyed rather than flat because a composite preprocessor serves several
        modalities at once, and a single flat list has nowhere to say which name
        belongs to which. A preprocessor serving one modality returns a one-key
        dict; a modality whose features have no meaningful names (a slide bag)
        returns an empty list for its key, or omits itself entirely.

        Names belong here rather than on a cohort because they are a property of a
        *fitted* transform: a one-hot encoder fitted on one fold can emit a column
        the next fold's does not. A cohort knows only which columns were declared.
        """
        ...

    def describe(self) -> str:
        """Human-readable summary of what this fold actually applied."""
        ...


def check_target(target: object) -> None:
    """Verify an object satisfies :class:`Target`, or raise naming what is missing.

    Called by :class:`~kalecancer.loaddata.base.Cohort` at construction. This is
    the enforcement that ``@runtime_checkable`` would only pretend to provide.

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
