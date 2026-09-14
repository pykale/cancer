"""Generic access to covariates held in a flat table.

Three objects with three lifetimes, which is what keeps a fold's fitted statistics
out of shared state:

* a :class:`Cohort` is an index, built once and read by every fold, never mutated;
* a :class:`~kalecancer.loaddata.multimodal_access.Preprocessor` is fitted state belonging
  to exactly one fold;
* a :class:`CohortView` pairs a row subset with one preprocessor, and is the
  ``torch.utils.data.Dataset`` a loader iterates.

A cohort is never fitted. :meth:`Cohort.fit_preprocessor` returns a separate artifact
scoped to the samples it was given, and :meth:`Cohort.view` pairs a subset with one.
That is what makes it impossible for a fold to see statistics from outside it: there
is nowhere on the shared object for them to live.

Loading is two phases, which lets one base class serve a 400-row clinical table and a
cohort of gigapixel slides: :meth:`Cohort._load_index` runs once and populates
``identifiers`` only; :meth:`Cohort.payload` reads one sample on demand.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone as sk_clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from kalecancer.loaddata.multimodal_access import (
    PatientBatch,
    PatientSample,
    Preprocessor,
    Target,
    check_target,
    collate_samples,
)
from kalecancer.prepdata.tabular import TabularPreprocessor

Identifiers = Sequence[str]

_READERS = {
    ".csv": lambda p, dtype: pd.read_csv(p, dtype=dtype),
    ".tsv": lambda p, dtype: pd.read_csv(p, sep="	", dtype=dtype),
    ".json": lambda p, dtype: pd.read_json(p, dtype=dtype),
}


class NotFittedError(RuntimeError):
    """Raised when transformed values are requested before transforms are fitted."""


class LeakageError(RuntimeError):
    """Raised when a preprocessor's fitted rows overlap rows it is applied to as held out."""


class Cohort(ABC):
    """An identifier-keyed index over samples. Built once, read by every fold.

    ``self.identifiers`` is the single ordering authority. A composite must reach into
    its components by identifier, never by position, or two components can disagree
    about which sample row ``5`` is -- which trains perfectly and means nothing.

    Args:
        path (str | Path | None, optional): Source for the index. ``None`` for
            composites, and for subclasses given an already-loaded object.
        name (str, optional): Key for this cohort's features in
            ``PatientSample.modalities``. Defaults to ``"features"``.
        target (Target | None, optional): Supervision. ``None`` for a pure feature
            provider, such as a slide cohort carrying no labels of its own.

    Raises:
        TypeError: If ``target`` does not satisfy the ``Target`` contract.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        name: str = "features",
        target: Target | None = None,
    ):
        if target is not None:
            check_target(target)
        self.path: Path | None = Path(path) if path is not None else None
        self.name = name
        self.target = target
        self.identifiers: list[str] = []
        self._row_of: dict[str, int] = {}

        if self._has_index_source():
            self._load_index()
            self._reindex()

    # ------------------------------------------------------------------ #
    # subclass contract -- three methods
    # ------------------------------------------------------------------ #

    def _has_index_source(self) -> bool:
        """Whether there is anything for :meth:`_load_index` to read at construction.

        Override wherever ``self.path`` is not the whole story -- a subclass given a
        loaded table, or a composite taking its index from its components.
        """
        return self.path is not None

    @abstractmethod
    def _load_index(self) -> None:
        """Read the index and populate ``self.identifiers``.

        Called once, at construction. Must not read payload -- for a slide cohort this
        reads a tile manifest and opens no images.
        """

    @abstractmethod
    def fit_preprocessor(self, identifiers: Identifiers) -> Preprocessor | None:
        """Fit transforms on the named samples only, and return them as a new artifact.

        The cohort is not modified, so a fold can never see statistics from outside it.

        Args:
            identifiers (Identifiers): Normally one fold's training samples.

        Returns:
            Preprocessor | None: The fitted artifact, or ``None`` when this cohort has
            nothing to fit. ``None`` is a legitimate answer, not a failure.
        """

    @abstractmethod
    def payload(self, identifier: str, prep: Preprocessor | None) -> dict[str, Tensor]:
        """Return this cohort's contribution to ``PatientSample.modalities``.

        Lazy by contract, and runs inside a DataLoader worker. **May be stochastic** --
        a slide cohort resamples tiles every epoch -- so nothing may assume two calls
        return the same tensor.

        Returns:
            dict[str, Tensor]: Usually ``{self.name: tensor}``. Shapes are
            modality-specific: ``(d,)`` for a tabular row, ``(n_tiles, d)`` for a bag.
        """

    # ------------------------------------------------------------------ #
    # optional hooks
    # ------------------------------------------------------------------ #

    def payload_bulk(self, identifiers: Identifiers, prep: Preprocessor | None) -> dict[str, Tensor] | None:
        """Transform every sample in one call, for modalities cheap enough to allow it.

        ``None`` by default, and a view falls back to per-sample :meth:`payload`.
        Overriding it is how a cohort opts into caching. ``TabularCohort`` does, because
        per-row scikit-learn calls would dominate the loop. **A slide cohort must not**:
        its payload is stochastic, so a cached block would freeze one tile draw for a
        whole run.

        Implementing this asserts three things. Rows come back in ``identifiers`` order
        -- a view reads the block positionally, so any other order pairs patients with
        the wrong features. :meth:`payload` is deterministic, since the two paths must
        agree. An empty ``identifiers`` returns empty tensors rather than raising.
        """
        return None

    def present(self, identifier: str) -> dict[str, Tensor]:
        """Whether each of this cohort's modalities is available for one sample.

        Unconditionally ``True`` here, since a single-modality cohort holds data for
        every identifier it indexed. Composites override it: a patient may have a
        clinical record but no usable slide.
        """
        return {self.name: torch.tensor(True)}

    # ------------------------------------------------------------------ #
    # shared
    # ------------------------------------------------------------------ #

    def view(self, identifiers: Identifiers, preprocessor: Preprocessor | None) -> CohortView:
        """Pair a subset of samples with a fitted preprocessor to make a torch Dataset.

        ``preprocessor`` is required even when ``None``: passing it explicitly keeps a
        fold's provenance at the call site rather than buried in object state.
        """
        return CohortView(self, identifiers, preprocessor)

    def split(
        self,
        test_size: float = 0.2,
        random_state: int | None = None,
        *,
        stratify: bool | np.ndarray,
    ) -> tuple[list[str], list[str]]:
        """Split into two sets of **identifiers**, balanced on what you name.

        Identifiers rather than cohorts, so this stays composable; and identifiers
        rather than positions, so a split taken from one cohort cannot be silently
        applied to another whose rows are in a different order.

        To use scikit-learn's splitters, split the returned list and index back into
        it -- ``[train_ids[i] for i in fold]`` -- the same shape as their ``groups``.

        Args:
            test_size (float, optional): Proportion held out. Defaults to 0.2.
            random_state (int | None, optional): Seed.
            stratify (bool | np.ndarray): **Required, no default.** ``True`` asks the
                target for labels, ``False`` disables it, or pass an array. Required
                because it changes the numbers you report and leaves no trace when
                wrong -- an unstratified 20% split of a few hundred patients can land
                several points off on the event rate.

        Returns:
            tuple[list[str], list[str]]: Train and test identifiers, each in cohort order.

        Raises:
            TypeError: If ``stratify=True`` but no target can supply labels.
        """
        labels = self._stratify_labels(stratify)
        train_pos, test_pos = train_test_split(
            np.arange(len(self)), test_size=test_size, random_state=random_state, stratify=labels
        )
        return self._ids_at(np.sort(train_pos)), self._ids_at(np.sort(test_pos))

    def _ids_at(self, positions: Sequence[int] | np.ndarray) -> list[str]:
        """Identifiers at the given positions. The only place positions are read."""
        return [self.identifiers[i] for i in positions]

    def _stratify_labels(self, stratify: bool | np.ndarray) -> np.ndarray | None:
        """Resolve the ``stratify`` argument of :meth:`split` to labels or ``None``."""
        if stratify is False:
            return None
        if not isinstance(stratify, bool):
            return np.asarray(stratify)

        if self.target is None:
            raise TypeError(
                "stratify=True needs a target to take labels from, and this cohort has "
                "none. Pass stratify=False to split at random, or pass an array to "
                "balance on something of your own choosing."
            )

        # Optional extension, not part of the Target contract: what is worth
        # stratifying on is task-specific and has no universal answer.
        labels_for = getattr(self.target, "stratify_labels", None)
        if labels_for is None:
            raise TypeError(
                f"stratify=True needs a target providing stratify_labels(identifiers), "
                f"which {type(self.target).__name__} does not. Pass stratify=False, or "
                f"pass an array to stratify on directly."
            )
        return np.asarray(labels_for(self.identifiers))

    def check_identifiers(self, identifiers: Identifiers) -> list[str]:
        """Validate a caller's identifier list, returning it as a plain list.

        Both failures it catches are silent otherwise. An identifier this cohort does
        not hold means a split was taken from somewhere else; a repeated one means a
        patient is counted twice, which inflates ``n`` and puts the same sample on both
        sides of a comparison.

        Raises:
            ValueError: If any identifier is unknown or appears more than once.
        """
        ids = list(identifiers)
        unknown = [i for i in ids if i not in self._row_of]
        if unknown:
            raise ValueError(
                f"{len(unknown)} identifier(s) are not in this cohort, e.g. {unknown[:5]}. "
                f"Identifiers must come from this cohort -- cohort.split() or "
                f"cohort.identifiers -- not from another cohort or an earlier version of "
                f"this one."
            )
        if len(set(ids)) != len(ids):
            counts = Counter(ids)
            repeated = sorted(i for i, n in counts.items() if n > 1)
            raise ValueError(
                f"{len(repeated)} identifier(s) appear more than once, e.g. {repeated[:5]}. "
                f"A repeated sample is counted twice in every statistic it reaches."
            )
        return ids

    def index_of(self, identifiers: Identifiers) -> np.ndarray:
        """Positions for ``identifiers``. The identifier-to-position bridge.

        For subclasses that store their payload as a block and must reach into it.
        Nothing outside a cohort should need this.
        """
        return np.array([self._row_of[i] for i in self.check_identifiers(identifiers)], dtype=int)

    def __len__(self) -> int:
        return len(self.identifiers)

    def __contains__(self, identifier: str) -> bool:
        return identifier in self._row_of

    def _reindex(self) -> None:
        """Rebuild the identifier-to-row lookup. Call after changing identifiers."""
        self._row_of = {identifier: i for i, identifier in enumerate(self.identifiers)}

    def __repr__(self) -> str:
        parts = [f"{len(self)} samples", f"name={self.name!r}"]
        if self.target is not None:
            summarise = getattr(self.target, "summarise", None)
            parts.append(summarise(self.identifiers) if summarise else type(self.target).__name__)
        return f"{type(self).__name__}({' | '.join(parts)})"


# --------------------------------------------------------------------------- #
# The view: one fold's rows, under one preprocessor
# --------------------------------------------------------------------------- #


class CohortView(TorchDataset):
    """A torch Dataset over some of a cohort's rows, under one preprocessor.

    Prefer :meth:`Cohort.view` to constructing this directly.

    Args:
        cohort (Cohort): Held by reference and never mutated, so several folds' views
            share one cohort safely.
        identifiers (Identifiers): Which samples, named. Validated against the cohort.
        preprocessor (Preprocessor | None): This fold's fitted state.
    """

    def __init__(
        self,
        cohort: Cohort,
        identifiers: Identifiers,
        preprocessor: Preprocessor | None,
    ):
        self.cohort = cohort
        self.identifiers = cohort.check_identifiers(identifiers)
        self.preprocessor = preprocessor

        # Caching is the cohort's decision, not the view's: one with a stochastic
        # payload never offers a bulk path, so a tile draw can never be frozen here.
        self._cache = cohort.payload_bulk(self.identifiers, preprocessor)
        if self._cache is not None:
            self._check_cache_alignment()

    def __len__(self) -> int:
        return len(self.identifiers)

    def __getitem__(self, i: int) -> PatientSample:
        """Return one training item.

        ``i`` is a position within this view -- torch's ``Dataset`` contract, and the
        only place in the API where a sample is reached for by number.
        """
        identifier = self.identifiers[i]
        if self._cache is None:
            modalities = self.cohort.payload(identifier, self.preprocessor)
        else:
            modalities = {name: values[i] for name, values in self._cache.items()}
        return PatientSample(
            patient_id=identifier,
            modalities=modalities,
            present=self.cohort.present(identifier),
            target={} if self.cohort.target is None else self.cohort.target.for_(identifier),
        )

    def batch(self) -> PatientBatch:
        """Every sample in this view, collated into one :class:`PatientBatch`.

        The bulk counterpart to iterating: features per modality, the target, the
        identifiers and the padding masks, all in one object -- the same one a
        ``DataLoader`` produces, so code written against a batch works either way.

        For fitting an embedder's context, for a full-batch loss, or for anything
        scikit-learn shaped. **Materialises the whole view**, so it suits a clinical
        table and not a cohort of slides.

        A cohort offering a bulk path already holds the block this would rebuild, so
        that case is served from it rather than sliced apart and stacked back together.
        The two routes must agree; a test pins them field for field.

        Raises:
            ValueError: If the view is empty.
        """
        if self._cache is None or not self.identifiers:
            return collate_samples([self[i] for i in range(len(self))])

        target = self.cohort.target
        return PatientBatch(
            patient_id=list(self.identifiers),
            # Cloned, not handed out: the block is read again by every __getitem__,
            # so an in-place op on the batch would rewrite this view's own features.
            modalities={name: block.clone() for name, block in self._cache.items()},
            present=self._present_bulk(),
            # No pad_mask: a bulk block arrives as one stacked tensor, so its rows are
            # fixed-width by construction and there is nothing ragged to mask.
            target={} if target is None else target.values_for(self.identifiers),
        )

    def _present_bulk(self) -> dict[str, Tensor]:
        """Availability per modality, stacked over this view's samples."""
        flags = [self.cohort.present(identifier) for identifier in self.identifiers]
        return {name: torch.stack([flag[name] for flag in flags]) for name in flags[0]}

    @property
    def feature_names(self) -> dict[str, list[str]]:
        """Post-encoding feature names per modality, from this fold's preprocessor.

        Passed through unchanged -- the preprocessor already keys them by modality, so
        there is nothing to reshape and nothing to guess.
        """
        if self.preprocessor is None:
            return {}
        return {name: list(values) for name, values in self.preprocessor.feature_names.items()}

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _check_cache_alignment(self) -> None:
        """Verify an eager cache lines up, row for row, with this view's patients.

        The cache is read positionally while the rows were asked for by name, so a
        cohort returning them in another order would pair patients with the wrong
        features -- silently. This is the one door in the design that has to be
        positional, so it is the one that gets checked.

        Row count always, plus the first row against ``payload()``. The spot check is
        valid because implementing ``payload_bulk`` asserts ``payload`` is deterministic.
        """
        assert self._cache is not None  # only called when there is one
        identifiers = self.identifiers

        for name, values in self._cache.items():
            if len(values) != len(identifiers):
                raise ValueError(
                    f"{type(self.cohort).__name__}.payload_bulk returned {len(values)} rows "
                    f"for modality '{name}' but was asked for {len(identifiers)}. Rows are "
                    f"read positionally, so a mismatched block pairs patients with the wrong "
                    f"features. Return exactly the identifiers you were given, in order."
                )

        if not identifiers:
            return
        expected = self.cohort.payload(identifiers[0], self.preprocessor)
        for name, values in self._cache.items():
            if name in expected and not _rows_equal(values[0], expected[name]):
                raise ValueError(
                    f"{type(self.cohort).__name__}.payload_bulk disagrees with payload() on "
                    f"the first row of modality '{name}' (patient {identifiers[0]!r}). The "
                    f"bulk block must return the identifiers it was given, in that order, and "
                    f"must agree with the per-sample path."
                )

    def __repr__(self) -> str:
        status = "no transforms" if self.preprocessor is None else "fitted"
        return f"{type(self).__name__}({len(self)} of {len(self.cohort)} samples | {status})"


def _rows_equal(left: Tensor, right: Tensor) -> bool:
    """Exact equality, counting NaN as equal to NaN.

    ``torch.equal`` calls any NaN tensor unequal to itself, and NaN is reachable --
    ``StandardScaler`` with no imputer propagates it -- so a plain comparison would
    raise on perfectly aligned data.
    """
    if left.shape != right.shape:
        return False
    return bool(((left == right) | (left.isnan() & right.isnan())).all())


# --------------------------------------------------------------------------- #
# The tabular implementation
# --------------------------------------------------------------------------- #


def _read_table(path: Path, id_column: str) -> pd.DataFrame:
    """Read a tabular file into a DataFrame, dispatching on suffix."""
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported table format '{path.suffix}'. Expected one of {sorted(_READERS)}.")
    return _prepare_frame(reader(path, {id_column: str}), id_column)


def _prepare_frame(frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Normalise a freshly-sourced table: fresh row index, identifiers as strings.

    Identifiers are forced to string because pandas reads ``"001"`` as ``1``, which a
    slide manifest keyed on ``"001"`` then silently fails to match. This can only
    *keep* padding, never recover it -- a frame read without ``dtype={id: str}`` has
    already lost it.
    """
    frame = frame.reset_index(drop=True)
    if id_column in frame.columns:
        frame[id_column] = frame[id_column].astype(str)
    return frame


class TabularCohort(Cohort):
    """Covariates from a flat table, keyed by a sample identifier.

    Column roles and their preprocessing are declared with plain scikit-learn
    transformers. Nothing is fitted here and this object is never mutated:
    :meth:`fit_preprocessor` fits on the rows you name and hands back a separate
    artifact, which :meth:`~kalecancer.loaddata.tabular_access.Cohort.view` pairs with a row
    subset to make a dataset.

    Args:
        source (str | Path | pd.DataFrame): A ``.csv``, ``.tsv`` or ``.json`` file, or
            a loaded ``DataFrame`` -- the route for anything else pandas can read.
            Read it with ``dtype={identifier: str}`` or zero-padded ids are already
            lost. The frame is copied.
        identifier (str): Column holding sample identifiers. Must be unique.
        target (Target | None, optional): ``None`` for a pure feature provider used as
            a component of a larger cohort.
        continuous (Sequence[str], optional): Numeric feature columns.
        categorical (Sequence[str], optional): Categorical feature columns.
        continuous_transform (optional): A scikit-learn transformer, or a list applied
            in order. ``None`` passes the columns through. No default pipeline:
            imputation and scaling are modelling decisions and belong in your script.
        categorical_transform (optional): As above, for the categorical columns.
        name (str, optional): Key for this modality in ``PatientSample.modalities``.
            Defaults to ``"clinical"``.

    Example:
        >>> from sklearn.impute import SimpleImputer
        >>> from sklearn.preprocessing import OneHotEncoder, StandardScaler
        >>> cohort = TabularCohort(
        ...     "data/my_cohort.csv",
        ...     identifier="patient_id",
        ...     target=SurvivalTarget(time="os_days", event="vital_status",
        ...                           event_value="dead"),
        ...     continuous=["age", "bmi"],
        ...     continuous_transform=[SimpleImputer(strategy="median"), StandardScaler()],
        ...     categorical=["sex", "stage"],
        ...     categorical_transform=OneHotEncoder(handle_unknown="ignore",
        ...                                         sparse_output=False),
        ... )
        >>> train_ids, test_ids = cohort.split(test_size=0.2, random_state=0, stratify=True)
        >>> prep = cohort.fit_preprocessor(train_ids)     # fitted on train rows only
        >>> train, test = cohort.view(train_ids, prep), cohort.view(test_ids, prep)

        A frame is the seam for anything between reading the file and binding the
        target, so the decision stays visible in your script:

        >>> frame = pd.read_csv("data/my_cohort.csv", dtype={"patient_id": str})
        >>> frame = frame[frame.vital_status.notna()]   # 24 patients, status not recorded
        >>> cohort = TabularCohort(frame, identifier="patient_id", target=target)
    """

    def __init__(
        self,
        source: str | Path | pd.DataFrame,
        identifier: str,
        target: Target | None = None,
        continuous: Sequence[str] = (),
        categorical: Sequence[str] = (),
        continuous_transform: Any = None,
        categorical_transform: Any = None,
        name: str = "clinical",
    ):
        # Set before super().__init__, which triggers _load_index().
        self.identifier = identifier
        self.continuous = list(continuous)
        self.categorical = list(categorical)
        self.continuous_transform = continuous_transform
        self.categorical_transform = categorical_transform

        # Seeded from a caller's frame, or overwritten by _load_index from self.path.
        # The empty placeholder is never read.
        self._frame: pd.DataFrame = source.copy() if isinstance(source, pd.DataFrame) else pd.DataFrame()
        self._spec: ColumnTransformer | None = None

        super().__init__(path=None if isinstance(source, pd.DataFrame) else source, name=name, target=target)

    def _has_index_source(self) -> bool:
        """Always: ``source`` is either a path to read or a frame already in hand."""
        return True

    # ------------------------------------------------------------------ #
    # index loading
    # ------------------------------------------------------------------ #

    def _load_index(self) -> None:
        """Read the table, validate the declared columns, bind the target.

        Reads everything, because a clinical table is small and index and payload are
        the same read here. WSI is what motivates keeping the phases apart.
        """
        self._frame = (
            _prepare_frame(self._frame, self.identifier)
            if self.path is None
            else _read_table(self.path, self.identifier)
        )
        self._validate_columns()

        self.identifiers = self._frame[self.identifier].tolist()
        duplicates = self._frame[self.identifier].duplicated()
        if duplicates.any():
            raise ValueError(
                f"Column '{self.identifier}' must be unique; found "
                f"{int(duplicates.sum())} duplicate(s), e.g. "
                f"{self._frame.loc[duplicates, self.identifier].unique()[:5].tolist()}"
            )

        self._bind_target()
        self._spec = self._build_spec()
        self._check_untransformed_roles()

    def _bind_target(self) -> None:
        """Hand the target the columns it declared it needs, as arrays.

        Checked here rather than inside ``bind`` so a missing column fails at
        construction, named in the target's own vocabulary.
        """
        if self.target is None:
            return
        missing = [c for c in self.target.required_columns if c not in self._frame.columns]
        if missing:
            raise ValueError(
                f"{type(self.target).__name__} requires column(s) {missing}, which are not "
                f"in this table. Available: {sorted(self._frame.columns)}"
            )
        values = {c: self._frame[c].to_numpy() for c in self.target.required_columns}
        self.target.bind(self.identifiers, values)

    def _validate_columns(self) -> None:
        columns = set(self._frame.columns)
        if self.identifier not in columns:
            raise ValueError(f"Identifier column '{self.identifier}' not found. Available: {sorted(columns)}")
        missing = [c for c in self.feature_columns if c not in columns]
        if missing:
            raise ValueError(f"Feature column(s) {missing} not found. Available: {sorted(columns)}")
        overlap = set(self.continuous) & set(self.categorical)
        if overlap:
            raise ValueError(f"Column(s) {sorted(overlap)} declared as both continuous and categorical.")
        if self.identifier in self.feature_columns:
            raise ValueError(f"Identifier column '{self.identifier}' cannot also be a feature.")

    def _build_spec(self) -> ColumnTransformer | None:
        """Assemble an *unfitted* ColumnTransformer, or ``None`` if no role is stateful."""
        blocks, stateful = [], False
        for columns, spec, label in self._roles():
            if not columns:
                continue
            steps = self._resolve(spec)
            if steps:
                blocks.append((label, make_pipeline(*steps), columns))
                stateful = True
            else:
                # Passthrough, or ColumnTransformer would drop these columns.
                blocks.append((label, "passthrough", columns))

        if not stateful:
            return None
        return ColumnTransformer(blocks, remainder="drop", verbose_feature_names_out=False)

    def _roles(self):
        """The (columns, transform spec, label) triple for each declared column role."""
        return (
            (self.continuous, self.continuous_transform, "continuous"),
            (self.categorical, self.categorical_transform, "categorical"),
        )

    @staticmethod
    def _resolve(spec: Any) -> list:
        """Normalise a transform spec to a list of unfitted transformer instances."""
        if spec is None:
            return []
        if isinstance(spec, str):
            raise ValueError(
                f"Transforms must be scikit-learn transformer instances, not the string "
                f"'{spec}'. Pass e.g. SimpleImputer(strategy='median'), a list of them, or "
                f"None to pass the columns through untouched. There is no shorthand: what "
                f"you preprocess with is a modelling decision and belongs in your script."
            )
        return list(spec) if isinstance(spec, list | tuple) else [spec]

    # ------------------------------------------------------------------ #
    # fold-local fitting
    # ------------------------------------------------------------------ #

    def fit_preprocessor(self, identifiers: Identifiers) -> TabularPreprocessor:
        """Fit the declared transforms on the named samples only, and return the artifact.

        The cohort is not modified, so folds are independent and can be fitted in
        parallel.

        Returns:
            TabularPreprocessor: Fitted transforms, the resulting feature names, and
            the identifiers they were fitted on.
        """
        ids = self.check_identifiers(identifiers)
        columns = self.feature_columns
        if self._spec is None:
            # Nothing stateful was declared. fitted_on stays empty: a passthrough
            # carries no row's information, so it cannot leak one.
            return TabularPreprocessor(None, columns, feature_names={self.name: list(columns)})

        transformer = sk_clone(self._spec)
        transformer.fit(self._frame.iloc[self.index_of(ids)][columns])
        return TabularPreprocessor(
            transformer,
            columns,
            feature_names={self.name: list(transformer.get_feature_names_out())},
            fitted_on=frozenset(ids),
        )

    # ------------------------------------------------------------------ #
    # payload
    # ------------------------------------------------------------------ #

    def payload_bulk(self, identifiers, prep) -> dict[str, Tensor]:
        """Transform every row at once, which is what lets a view cache.

        Overridden because scikit-learn's per-call overhead would dominate the loop if
        ``transform`` ran once per row.
        """
        self._require(prep)
        rows = self.index_of(identifiers)
        return {self.name: prep.transform(self._frame.iloc[rows])}

    def payload(self, identifier: str, prep) -> dict[str, Tensor]:
        """Return the ``(n_features,)`` feature vector for one sample."""
        self._require(prep)
        row = self._row_of[identifier]
        return {self.name: prep.transform(self._frame.iloc[[row]])[0]}

    def _require(self, prep) -> None:
        """Refuse to serve values without a preprocessor.

        Refused even for a passthrough, so there is one rule rather than two: accepting
        ``None`` would let a cohort *with* declared transforms serve raw values.
        """
        if prep is None:
            raise NotFittedError(
                f"{type(self).__name__} was asked for values with no preprocessor.\n"
                f"  Fit one on this fold's training rows first:\n"
                f"      prep = cohort.fit_preprocessor(train_ids)\n"
                f"      train = cohort.view(train_ids, prep)\n"
                f"      val   = cohort.view(val_ids, prep)   # same prep, never refitted\n"
                f"  For the untransformed values, use cohort.frame."
            )

    #: Suggested fix per role, quoted back at the user in the messages below.
    _ENCODE_HINT = "categorical_transform=OneHotEncoder(handle_unknown='ignore', sparse_output=False)"
    _IMPUTE_HINT = {
        "continuous": "continuous_transform=SimpleImputer(strategy='median')",
        "categorical": "categorical_transform=SimpleImputer(strategy='most_frequent')",
    }

    def _check_untransformed_roles(self) -> None:
        """Refuse to build a cohort whose features no transform was declared to clean up.

        Checked **per role**, because a transform on one role does nothing for the
        other. A half-declared spec -- scaled numbers beside raw strings -- would
        otherwise die much later as ``could not convert string to float: 'F'``, and
        missing values would sail through as NaN into the model.
        """
        for columns, spec, label in self._roles():
            if not columns or self._resolve(spec):
                continue  # nothing declared for this role is the only case to police
            frame = self._frame[columns]

            non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
            if non_numeric:
                fix = (
                    f"Add e.g. {self._ENCODE_HINT}."
                    if label == "categorical"
                    else f"Declare them as categorical instead, with {self._ENCODE_HINT}."
                )
                raise ValueError(
                    f"{label.capitalize()} column(s) {non_numeric} are not numeric and no "
                    f"{label}_transform was declared for them. {fix}"
                )

            missing = frame.isna().sum()
            missing = missing[missing > 0]
            if not missing.empty:
                raise ValueError(
                    f"Missing values in {missing.to_dict()} and no {label}_transform was "
                    f"declared to handle them. Add e.g. {self._IMPUTE_HINT[label]}, or resolve "
                    f"them upstream. Passing NaN into a model silently is never right."
                )

    # ------------------------------------------------------------------ #
    # access
    # ------------------------------------------------------------------ #

    @property
    def frame(self) -> pd.DataFrame:
        """The untransformed table. The exploration hatch."""
        return self._frame

    @property
    def feature_columns(self) -> list[str]:
        """Declared feature columns, continuous first."""
        return self.continuous + self.categorical

    def describe_transforms(self) -> str:
        """Summary of the transforms this cohort *declares*.

        What a fold actually applied is its ``TabularPreprocessor``'s to answer.
        """
        lines = []
        for columns, spec, label in self._roles():
            if not columns:
                continue
            steps = self._resolve(spec)
            rendered = " -> ".join(type(s).__name__ for s in steps) if steps else "passthrough"
            lines.append(f"{label:<12}({len(columns)} cols): {rendered}")
        if not lines:
            return "no feature columns declared"
        return "\n".join(lines)

    def __repr__(self) -> str:
        parts = [f"{len(self)} samples", f"{len(self.feature_columns)} columns"]
        if self.target is not None:
            summarise = getattr(self.target, "summarise", None)
            parts.append(summarise(self.identifiers) if summarise else type(self.target).__name__)
        return f"{type(self).__name__}({' | '.join(parts)})"
