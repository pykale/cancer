"""Leakage-safe splitting, as splitter objects.

Splitters follow scikit-learn's shape: an object configured once, then asked to
``split`` a table as many times as needed, yielding index arrays into it. That is
what makes a split a value a caller can pass around, log, or swap for another
without the code that consumes it changing::

    splitter = CrossValidation(n_splits=5, group_by="patient_id", stratify_by=["event"])
    for split in splitter.split(cohort):
        train, val, test = (cohort.iloc[split[name]] for name in ("train", "val", "test"))

Two differences from scikit-learn, both deliberate:

* A split is a **named mapping**, not a ``(train, test)`` pair. Model selection needs
  a validation set that is not the test set, and naming the three removes any chance
  of a caller taking them in the wrong order.
* Columns are named rather than passed as parallel ``y`` and ``groups`` arrays, so a
  splitter can be built from configuration and applied to any table carrying those
  columns.

Rows sharing a ``group_by`` value always land in the same split, which is what
prevents leakage when one subject contributes several rows -- several slides, several
visits. Stratification keys are combined into one categorical label, so one, several
or no keys behave identically.

Nothing here knows a dataset. :class:`Predefined` applies an assignment somebody else
produced; where that assignment comes from -- a published split file, an
internal/external cohort, a site or scanner partition -- belongs to the dataset that
defines it, not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit

#: Positional indices into the table that was split, keyed by split name. Positional
#: rather than labels so a caller selects rows with ``frame.iloc[...]`` whatever the
#: table's index happens to be.
Split = dict[str, np.ndarray]

#: The names every splitter here produces, in the order they are meant to be used.
SPLIT_NAMES = ("train", "val", "test")


class SplitError(ValueError):
    """Raised when a requested split cannot be produced."""


def composite_labels(frame: pd.DataFrame, stratify_by: Sequence[str] = (), min_count: int = 2) -> np.ndarray:
    """Combine stratification columns into a single categorical target.

    One label per row from any number of columns, so a splitter needs no special case
    for zero, one or several keys. Combinations too rare to appear on both sides of a
    split are folded into the largest class rather than failing the split: a cohort
    with one patient of a rare stage is ordinary, and refusing to split it is not
    useful.

    Args:
        frame: Table to label.
        stratify_by: Columns whose combination should be preserved. Empty gives a
            constant label, i.e. no stratification.
        min_count: Smallest class size kept as its own label.

    Returns:
        ``(n_rows,)`` integer labels.

    Raises:
        SplitError: If a named column is not in the table.
    """
    missing = [key for key in stratify_by if key not in frame.columns]
    if missing:
        raise SplitError(f"stratify_by names {missing}, which are not columns of the table: {list(frame.columns)}")
    if not stratify_by:
        return np.zeros(len(frame), dtype=np.int64)

    combined = frame[list(stratify_by)].astype(str).agg("|".join, axis=1)
    labels, _ = pd.factorize(combined)
    counts = pd.Series(labels).value_counts()
    return np.where(np.isin(labels, counts[counts >= min_count].index), labels, counts.idxmax())


def train_test_split(
    frame: pd.DataFrame,
    test_size: float = 0.15,
    group_by: str | None = None,
    stratify_by: Sequence[str] = (),
    random_state: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """Indices for one grouped, stratified two-way split.

    The function form, for when a caller wants a single division and not a splitter
    to iterate -- carving validation out of a training half, most often. Named as
    scikit-learn's is because it does the same job, over a table with named columns
    rather than parallel arrays.

    Args:
        frame: Table describing the samples.
        test_size: Share held out.
        group_by: Column whose rows must not be divided.
        stratify_by: Columns whose distribution both sides should preserve.
        random_state: Seed, making the split reproducible.

    Returns:
        ``(kept, held_out)`` positional indices into ``frame``.

    Raises:
        SplitError: If the table is too small to divide.
    """
    if not 0 < test_size < 1:
        raise SplitError(f"test_size must lie in (0, 1), got {test_size}")
    splitter = HoldOut(
        test_size=test_size, val_size=0.0, group_by=group_by, stratify_by=stratify_by, random_state=random_state
    )
    split = next(splitter.split(frame))
    return split["train"], split["test"]


class CohortSplitter(ABC):
    """Base class for splitters over a table of samples.

    Args:
        val_size: Share of the training half held out for validation. 0 leaves no
            validation set, for a caller that does no model selection.
        group_by: Column whose rows must never be divided across splits. ``None``
            treats every row as independent.
        stratify_by: Columns whose distribution each split should preserve.
        random_state: Seed, making a split reproducible. Named as scikit-learn does.
    """

    def __init__(
        self,
        val_size: float = 0.15,
        group_by: str | None = None,
        stratify_by: Sequence[str] = (),
        random_state: int = 2026,
    ) -> None:
        if not 0 <= val_size < 1:
            raise SplitError(f"val_size must lie in [0, 1), got {val_size}")
        self.val_size = val_size
        self.group_by = group_by
        self.stratify_by = tuple(stratify_by)
        self.random_state = random_state

    @abstractmethod
    def split(self, frame: pd.DataFrame) -> Iterator[Split]:
        """Yield one split of ``frame``, or several for a cross-validating splitter.

        Args:
            frame: Table describing the samples, carrying whichever columns this
                splitter was configured with.

        Yields:
            Positional indices keyed ``"train"``, ``"val"`` and ``"test"``.

        Raises:
            SplitError: If the table cannot support the requested split.
        """

    def __repr__(self) -> str:
        settings = ", ".join(f"{key}={value!r}" for key, value in vars(self).items())
        return f"{type(self).__name__}({settings})"

    # ------------------------------------------------------------------ #
    # shared machinery
    # ------------------------------------------------------------------ #

    def _require_columns(self, frame: pd.DataFrame, minimum: int) -> None:
        if self.group_by is not None and self.group_by not in frame.columns:
            raise SplitError(f"group_by {self.group_by!r} is not a column of the table: {list(frame.columns)}")
        units = frame[self.group_by].nunique() if self.group_by else len(frame)
        if units < minimum:
            unit = f"unique {self.group_by}" if self.group_by else "rows"
            raise SplitError(f"need at least {minimum} {unit} to split, got {units}")

    def _hold_out(self, frame: pd.DataFrame, indices: np.ndarray, size: float, seed: int) -> tuple[np.ndarray, ...]:
        """Hold out approximately ``size`` of ``indices``, keeping groups intact.

        The split is decided over independent units -- one per group where grouping
        applies, otherwise one per row -- then expanded back to rows, so a group is
        never divided. Units are drawn by a stratified shuffle rather than assembled
        from whole folds, which lets the share be met directly instead of rounded to
        a fold boundary.

        Raises:
            SplitError: If fewer than two independent units are available.
        """
        subset = frame.iloc[indices]
        units = subset.drop_duplicates(self.group_by) if self.group_by else subset
        count = len(units)
        if count < 2:
            unit = f"unique {self.group_by}" if self.group_by else "rows"
            raise SplitError(f"need at least 2 {unit} to hold part out, got {count}")

        # Both sides keep at least one unit, whatever the share asks for.
        held_count = min(max(round(size * count), 1), count - 1)
        labels = composite_labels(units, self.stratify_by, min_count=2)
        # Stratification needs room for every class on both sides. Where there is not
        # enough, it is dropped rather than allowed to fail: the split degrades to a
        # plain shuffle instead of raising on a small table.
        if min(held_count, count - held_count) < len(np.unique(labels)):
            labels = np.zeros(count, dtype=np.int64)

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=held_count, random_state=seed)
        _, held_units = next(splitter.split(np.zeros(count), labels))

        if self.group_by:
            held_groups = set(units.iloc[held_units][self.group_by])
            held = subset[self.group_by].isin(held_groups).to_numpy()
        else:
            held = np.zeros(count, dtype=bool)
            held[held_units] = True
        return indices[~held], indices[held]

    def _with_validation(self, frame: pd.DataFrame, fit: np.ndarray, test: np.ndarray, seed: int) -> Split:
        """Carve validation out of the training half, never out of the test set."""
        if self.val_size == 0:
            return {"train": fit, "val": np.array([], dtype=int), "test": test}
        train, val = self._hold_out(frame, fit, self.val_size, seed)
        return {"train": train, "val": val, "test": test}


class HoldOut(CohortSplitter):
    """One stratified, grouped train/validation/test split.

    The counterpart of scikit-learn's ``train_test_split``, extended with a third set
    and with grouping.

    Args:
        test_size: Share of the cohort held out for testing.
        val_size: Share of the remaining training half held out for validation.
        group_by: Column whose rows must never be divided across splits.
        stratify_by: Columns whose distribution each split should preserve.
        random_state: Seed, making the split reproducible.
    """

    def __init__(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        group_by: str | None = None,
        stratify_by: Sequence[str] = (),
        random_state: int = 2026,
    ) -> None:
        super().__init__(val_size=val_size, group_by=group_by, stratify_by=stratify_by, random_state=random_state)
        if not 0 < test_size < 1:
            raise SplitError(f"test_size must lie in (0, 1), got {test_size}")
        # Checked together as well as apart: each can be valid alone while the pair
        # asks for more than the cohort holds, which would otherwise fail later and
        # further away.
        if val_size + test_size >= 1:
            raise SplitError(f"val_size + test_size must lie in (0, 1), got {val_size + test_size}")
        self.test_size = test_size

    def split(self, frame: pd.DataFrame) -> Iterator[Split]:
        self._require_columns(frame, minimum=3)
        indices = np.arange(len(frame))
        fit, test = self._hold_out(frame, indices, self.test_size, self.random_state)
        yield self._with_validation(frame, fit, test, self.random_state)


class CrossValidation(CohortSplitter):
    """Each of ``n_splits`` folds held out for testing in turn.

    The counterpart of scikit-learn's ``StratifiedGroupKFold``, which it uses, with a
    validation set carved from each fold's training rows so the three sets stay
    disjoint within every fold.

    Args:
        n_splits: Number of folds.
        val_size: Share of each fold's non-test rows held out for validation.
        group_by: Column whose rows must never be divided across splits.
        stratify_by: Columns whose distribution each fold should preserve.
        random_state: Seed, making the folds reproducible.
    """

    def __init__(
        self,
        n_splits: int = 5,
        val_size: float = 0.15,
        group_by: str | None = None,
        stratify_by: Sequence[str] = (),
        random_state: int = 2026,
    ) -> None:
        super().__init__(val_size=val_size, group_by=group_by, stratify_by=stratify_by, random_state=random_state)
        if n_splits < 2:
            raise SplitError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits

    def split(self, frame: pd.DataFrame) -> Iterator[Split]:
        self._require_columns(frame, minimum=self.n_splits)

        labels = composite_labels(frame, self.stratify_by, min_count=self.n_splits)
        groups = frame[self.group_by].to_numpy() if self.group_by else None
        folds = (StratifiedGroupKFold if self.group_by else StratifiedKFold)(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        for fold, (fit, test) in enumerate(folds.split(frame, labels, groups)):
            yield self._with_validation(frame, fit, test, self.random_state + fold)


class Predefined(CohortSplitter):
    """An assignment somebody else decided, applied to a table.

    The counterpart of scikit-learn's ``PredefinedSplit``. Use it wherever the
    partition is a property of the data rather than something to draw: a published
    train/test assignment, internal against external cohorts, one site held out from
    another. This class only *applies* such an assignment -- reading it, and knowing
    what its keys mean, belongs to the dataset that defines it.

    Only the validation set is drawn here, and only from the training half, so the
    given test set is reproduced exactly. Identifiers absent from the table are
    ignored, which lets an assignment covering a whole cohort apply to the subset
    that survived a modality join.

    Args:
        assignment: Identifiers keyed by split name, as the dataset publishes them.
        id_column: Column holding the identifier the assignment refers to.
        train_key: Key in ``assignment`` holding the training identifiers.
        test_key: Key in ``assignment`` holding the test identifiers.
        val_size: Share of the training half held out for validation.
        group_by: Column whose rows must not be divided across train and validation.
            The given test set is already grouped by ``id_column``, but a table with
            several rows per subject would otherwise let one subject's rows span both
            halves of the training data.
        stratify_by: Columns whose distribution the validation split preserves.
        random_state: Seed, making the validation split reproducible.
    """

    def __init__(
        self,
        assignment: Mapping[str, Sequence[str]],
        id_column: str = "patient_id",
        train_key: str = "training",
        test_key: str = "test",
        val_size: float = 0.15,
        group_by: str | None = None,
        stratify_by: Sequence[str] = (),
        random_state: int = 2026,
    ) -> None:
        super().__init__(val_size=val_size, group_by=group_by, stratify_by=stratify_by, random_state=random_state)
        self.assignment = assignment
        self.id_column = id_column
        self.train_key = train_key
        self.test_key = test_key

    def split(self, frame: pd.DataFrame) -> Iterator[Split]:
        if self.id_column not in frame.columns:
            raise SplitError(f"id_column {self.id_column!r} is not a column of the table: {list(frame.columns)}")

        identifiers = frame[self.id_column].astype(str).to_numpy()
        positions = {
            key: np.flatnonzero(np.isin(identifiers, [str(value) for value in self.assignment.get(key, ())]))
            for key in (self.train_key, self.test_key)
        }
        for key, found in positions.items():
            if found.size == 0:
                raise SplitError(
                    f"the {key!r} assignment matched no row of the table; "
                    f"the usual cause is {self.id_column} losing its zero padding"
                )

        yield self._with_validation(frame, positions[self.train_key], positions[self.test_key], self.random_state)
