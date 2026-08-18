"""TabICL as a frozen feature extractor.

TabICL predicts by placing labelled training rows in context alongside the rows being
predicted, in one forward pass: a column-wise transformer, a row-wise transformer
giving one vector per row, then a dataset-wise transformer that predicts. This wrapper
stops after the second stage and returns the row vectors. There is no public API for
that upstream, so it reaches into ``TabICLClassifier.model_``.

Two measured properties shape the design (512 dims, checkpoint v2):

1. **A row's representation depends on its context.** The same 100 rows embedded
   against a 200-row and a 100-row context differ by ~0.4 per dimension. The context
   is fitted state exactly as a ``StandardScaler`` mean is, hence :meth:`fit` and
   :meth:`encode` rather than one stateless transform.
2. **Context rows do not see the rows being encoded**, with ``embed_with_test=False``:
   a context row's representation is bit-identical whatever is encoded against it
   (difference exactly 0.0). Flipping the flag moves that to ~0.27, so it is
   hard-coded rather than exposed.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import Tensor

from kalecancer.loaddata.base import NotFittedError
from kalecancer.loaddata.view import CohortView

#: Context rows must never attend to the rows being encoded. See the module docstring.
_EMBED_WITH_TEST = False


def _require_tabicl():
    """Import TabICLClassifier, or explain how to install it."""
    try:
        from tabicl import TabICLClassifier
    except ImportError as exc:  # pragma: no cover - exercised by install state, not tests
        raise ImportError(
            "TabICLEncoder requires the 'tabicl' package, which is not installed. "
            "Install it with: pip install 'kalecancer[tabular]'  (or: pip install tabicl). "
            "The pretrained checkpoint is downloaded from Hugging Face on first use."
        ) from exc
    return TabICLClassifier


#: Attributes ``TabICLClassifier._load_model`` sets from a checkpoint. All three are
#: read-only once loaded, which is what makes them shareable; see :func:`_shared_load`.
_CHECKPOINT_ATTRS = ("model_", "model_config_", "model_path_")

#: One loaded checkpoint per ``(checkpoint, device)``, shared by every encoder asking
#: for it. Keyed rather than global because two encoders may legitimately want
#: different weights or devices in one process.
_LOADED: dict[tuple, dict] = {}


def _shared_load(clf, key: tuple) -> None:
    """Point ``clf`` at a shared copy of the checkpoint instead of loading its own.

    110MB of frozen weights, and a fresh estimator is built per fold to keep contexts
    independent -- so five folds would hold five identical copies.

    Safe only because nothing mutates them: no fine-tuning, no gradients, and
    ``kv_cache`` left ``False`` so the module's one mutable field stays empty.
    **Fine-tuning must not use this path**, or folds would train each other's weights.
    """
    cached = _LOADED.get(key)
    if cached is None:
        return

    def _install() -> None:
        clf.__dict__.update(cached)

    clf._load_model = _install


def _remember_load(clf, key: tuple) -> None:
    """Record this estimator's freshly-loaded checkpoint for later encoders to share."""
    if key in _LOADED:
        # _load_model was not routed through; share the cached module anyway so the
        # memory saving holds even if the skip above stopped working.
        clf.__dict__.update(_LOADED[key])
        return
    _LOADED[key] = {name: getattr(clf, name) for name in _CHECKPOINT_ATTRS if hasattr(clf, name)}


def _matrix(view: CohortView) -> np.ndarray:
    """Stack a view's feature vectors into ``(n_samples, n_features)``.

    Goes through the view's own ``__getitem__`` in its natural order rather than
    touching any internal matrix, so rows stay aligned with ``view.identifiers``
    and any cohort implementing the modality contract works here unchanged.
    """
    if len(view) == 0:
        raise ValueError("Cannot encode an empty view.")
    rows = []
    for i in range(len(view)):
        modalities = view[i].modalities
        if len(modalities) != 1:
            raise ValueError(
                f"TabICLEncoder needs a single-modality view; this one carries "
                f"{sorted(modalities)}. Encode each modality with its own encoder and "
                f"fuse the representations afterwards."
            )
        rows.append(next(iter(modalities.values())).numpy())
    return np.stack(rows)


class TabICLEncoder:
    """Frozen TabICL row representations for a tabular dataset.

    Not fine-tuned and not trainable: the pretrained weights are used as-is, and the
    only fitted state is the in-context example set. :meth:`fit` returns a **new**
    encoder and leaves this one untouched, so folds cannot contaminate one another.

    Args:
        context_label (str, optional): Which supervision value conditions the
            representation. TabICL's column embedder is target-aware, so this is a
            modelling choice, not a formatting one. Only ``"event"`` is implemented;
            ``"time"`` needs ``TabICLRegressor`` and raises.
        checkpoint (str | None, optional): Checkpoint version. ``None`` takes the
            upstream default, downloaded from Hugging Face on first use.
        n_estimators (int, optional): TabICL's internal ensemble over normalisations and
            feature permutations, averaged across members. Defaults to 1, because the
            ensemble was designed to average *predictions*; averaging *representations*
            is a different and unvalidated operation and should not happen silently.
        device (str | None, optional): Torch device. ``None`` lets TabICL choose.
        random_state (int | None, optional): Seed for the ensemble generator.

    Attributes:
        out_dim (int): Representation width, known once fitted.

    Example:
        >>> train_idx, test_idx = cohort.split(test_size=0.2, random_state=0, stratify=True)
        >>> fitted_train = train.fit_transform()
        >>> fitted_test = fitted_train.transform(test)
        >>> encoder = TabICLEncoder().fit(fitted_train)
        >>> Z_test = encoder.encode(fitted_test)      # (n_test, 512)

    Note:
        Context rows see their own label when encoded, so ``encode(train)`` is
        optimistic relative to ``encode(test)``. That is inherent to in-context
        learning, but a head fitted on training representations and applied to held-out
        ones is crossing a distribution boundary. Fixing it properly needs cross-fitted
        or leave-one-out contexts, which this class does not do.
    """

    _CONTEXT_LABELS = ("event", "time")

    def __init__(
        self,
        context_label: str = "event",
        checkpoint: str | None = None,
        n_estimators: int = 1,
        device: str | None = None,
        random_state: int | None = None,
    ):
        if context_label not in self._CONTEXT_LABELS:
            raise ValueError(f"Unknown context_label '{context_label}'. Expected one of {list(self._CONTEXT_LABELS)}.")
        if context_label == "time":
            raise NotImplementedError(
                "context_label='time' requires the TabICL regressor checkpoint, which is not "
                "wired up yet. Use context_label='event' to condition on the event indicator."
            )
        self.context_label = context_label
        self.checkpoint = checkpoint
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state

        self._clf: Any = None
        self._context_size: int = 0

    # ------------------------------------------------------------------ #
    # fold discipline -- fit returns a new object, never mutates in place
    # ------------------------------------------------------------------ #

    @property
    def is_fitted(self) -> bool:
        """Whether an in-context example set has been absorbed."""
        return self._clf is not None

    @property
    def out_dim(self) -> int:
        """Width of the representation, read from the loaded checkpoint.

        Raises:
            NotFittedError: Before :meth:`fit`, which is when the checkpoint loads.
        """
        self._check_fitted()
        model = self._clf.model_
        return int(model.embed_dim * model.row_num_cls)

    def fit(self, dataset: CohortView) -> TabICLEncoder:
        """Absorb ``dataset`` as the in-context example set.

        No gradient step: TabICL conditions on context rows, so "fitting" is storing
        them. Pass a *training fold only* -- a context drawn from the whole cohort leaks
        held-out rows into every representation, and no amount of correct
        ``ColumnTransformer`` handling catches it.

        Returns:
            TabICLEncoder: A new, fitted encoder. ``self`` is unchanged.

        Raises:
            ValueError: If ``dataset`` has no target to draw context labels from.
            TypeError: If the target cannot supply the requested ``context_label``.
        """
        target = dataset.cohort.target
        if target is None:
            raise ValueError(
                "TabICLEncoder.fit needs a dataset with a target: TabICL's column embedder "
                "is target-aware, so the context rows must carry labels. Construct the "
                "dataset with target=SurvivalTarget(...)."
            )

        # Asked for by capability, not type: event language is not part of the Target
        # contract, so a survival-specific need is checked where it is needed.
        labels_for = getattr(target, "events_for", None)
        if labels_for is None:
            raise TypeError(
                f"context_label='{self.context_label}' needs a target providing events_for(), "
                f"which {type(target).__name__} does not. Use a SurvivalTarget, or add "
                f"that method to your target."
            )

        X = _matrix(dataset)
        y = np.asarray(labels_for(dataset.identifiers)).astype(int)

        TabICLClassifier = _require_tabicl()
        kwargs: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "device": self.device,
            "random_state": self.random_state,
        }
        if self.checkpoint is not None:
            kwargs["checkpoint_version"] = self.checkpoint

        # Each fold gets its own estimator, so its context stays its own -- but they can
        # all point at one copy of the frozen weights. See _shared_load.
        key = (self.checkpoint, self.device)
        clf = TabICLClassifier(**kwargs)
        _shared_load(clf, key)
        clf.fit(X, y)
        _remember_load(clf, key)

        # Shallow copy is safe: _clf is replaced wholesale rather than mutated, so no
        # fitted state is shared between the clone and self.
        fitted = copy.copy(self)
        fitted._clf = clf
        fitted._context_size = len(dataset)
        return fitted

    # ------------------------------------------------------------------ #
    # encoding
    # ------------------------------------------------------------------ #

    def encode(self, dataset: CohortView) -> Tensor:
        """Return one representation per sample, in ``dataset.identifiers`` order.

        Rows are appended after the context and embedded against it in one forward
        pass; the context's own representations are discarded.

        Returns:
            Tensor: ``(len(dataset), out_dim)``, float32.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
            ValueError: If ``dataset`` has a different feature width than the context.
        """
        self._check_fitted()
        X = _matrix(dataset)

        # Going straight to the model skips TabICL's own input validation, and width
        # mismatches are routine -- a one-hot encoder fitted on one fold can emit a
        # column the next one's does not. Caught here rather than deep in the transformer.
        expected = int(self._clf.n_features_in_)
        if X.shape[1] != expected:
            raise ValueError(
                f"{type(self).__name__} was fitted on {expected} features but was given "
                f"{X.shape[1]}. Held-out data must be prepared with the training fold's "
                f"own transforms -- fitted_train.transform(other) -- not fitted separately."
            )

        encoded = self._representations(X)
        if encoded.shape[0] != len(dataset):
            raise RuntimeError(
                f"Encoded {encoded.shape[0]} rows for a dataset of {len(dataset)}; "
                f"representations would not line up with identifiers."
            )
        return encoded

    def _representations(self, X: np.ndarray) -> Tensor:
        """Run the column embedder and row interactor over context + query rows."""
        clf = self._clf
        model = clf.model_

        # ColEmbedding.forward branches on self.training, and the module is shared with
        # every other encoder on this checkpoint. Asserted, not assumed.
        model.eval()

        # TabICL's own preprocessing: dtypes and missingness, then the normalisation the
        # checkpoint was trained with, re-attaching the stored context rows.
        prepared = clf.X_encoder_.transform(X)
        data = clf.ensemble_generator_.transform(prepared, mode="both")

        members = []
        for _norm_method, (Xs, ys) in data.items():
            Xb = torch.from_numpy(Xs).float().to(clf.device_)
            yb = torch.from_numpy(ys).float().to(clf.device_)
            train_size = yb.shape[1]
            with torch.no_grad():
                embeddings = model.col_embedder(
                    Xb,
                    y_train=yb,
                    embed_with_test=_EMBED_WITH_TEST,
                    mgr_config=clf.inference_config_.COL_CONFIG,
                )
                reps = model.row_interactor(embeddings, mgr_config=clf.inference_config_.ROW_CONFIG)
            members.append(reps[:, train_size:, :].float().cpu())

        # (n_members, n_query, out_dim) -> mean over members. With the default
        # n_estimators=1 this is a no-op.
        return torch.cat(members, dim=0).mean(dim=0)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _check_fitted(self) -> None:
        """Guard the unfitted case, naming the likeliest cause: a *discarded* fit.

        ``fit`` returns a new encoder, so the sklearn reflex of ``encoder.fit(train)``
        on its own line leaves ``encoder`` unfitted and fails one call later.
        """
        if not self.is_fitted:
            raise NotFittedError(
                f"{type(self).__name__} has no in-context example set.\n"
                f"  If you called fit() already, note that it returns a *new* encoder and "
                f"leaves this one untouched:\n"
                f"      encoder = encoder.fit(train)     # not: encoder.fit(train)\n"
                f"  That is what keeps cross-validation folds from sharing a context. Fit on a "
                f"training fold only -- a context drawn from the whole cohort leaks held-out "
                f"rows into every representation."
            )

    def __repr__(self) -> str:
        parts = [f"context_label={self.context_label!r}"]
        if self.n_estimators != 1:
            parts.append(f"n_estimators={self.n_estimators}")
        parts.append(f"context {self._context_size} rows -> {self.out_dim}d" if self.is_fitted else "unfitted")
        return f"{type(self).__name__}({' | '.join(parts)})"
