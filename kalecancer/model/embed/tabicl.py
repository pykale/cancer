"""TabICL as a modality embedder.

TabICL predicts by placing labelled rows in context beside the rows it is asked
about, in a single forward pass: a column-wise transformer, a row-wise transformer
giving one vector per row, then a dataset-wise transformer that predicts. This
wrapper stops after the second stage and returns the row vectors. There is no public
API for that upstream, so it reaches into the loaded ``TabICL`` module.

The context is **data this object holds**, handed to it at construction. Nothing here
knows what a cohort, a view or a survival target is; it takes two arrays.

Three properties of the real checkpoint shape the design (512 dims, v2):

1. **A row's representation depends on its context.** The same 100 rows embedded
   against a 200-row and a 100-row context differ by ~0.4 per dimension. The context
   belongs to one fold in exactly the sense a ``StandardScaler`` mean does.
2. **Context rows do not see the rows being embedded**, with ``embed_with_test=False``:
   a context row's representation is bit-identical whatever is embedded against it.
   Flipping the flag moves that to ~0.27, so it is hard-coded rather than exposed.
3. **The rows being embedded do not see each other.** Embedding 3 rows alone matches
   their values from a 60-row block to ~3e-5, against ~5.0 for the same rows under a
   different context. Cross-row mixing happens in the dataset-wise transformer, the
   stage this wrapper stops before; ``RowInteraction`` attends across features within
   a row. Mini-batching is therefore not *wrong* -- it is wasteful, which is what
   :attr:`TabICLEmbedder.needs_full_batch` records.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor, nn

#: Context rows must never attend to the rows being embedded. See the module docstring.
_EMBED_WITH_TEST = False

#: One loaded checkpoint per ``(checkpoint, device)``, shared by every *frozen*
#: embedder asking for it -- 110MB of weights, and cross-validation builds one
#: embedder per fold. Keyed rather than global because two embedders may legitimately
#: want different weights or devices in one process.
_BACKBONES: dict[tuple, Any] = {}


def _require_tabicl():
    """Import TabICLClassifier, or explain how to install it."""
    try:
        from tabicl import TabICLClassifier
    except ImportError as exc:  # pragma: no cover - exercised by install state, not tests
        raise ImportError(
            "TabICLEmbedder requires the 'tabicl' package, which is not installed. "
            "Install it with: uv sync --all-extras  (or: pip install tabicl). "
            "The pretrained checkpoint is downloaded from Hugging Face on first use."
        ) from exc
    return TabICLClassifier


def _as_matrix(values: ArrayLike | Tensor, name: str) -> np.ndarray:
    """Coerce ``(n, d)`` features to a plain float array."""
    array = values.detach().cpu().numpy() if isinstance(values, Tensor) else np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features); got shape {array.shape}.")
    if len(array) == 0:
        raise ValueError(f"{name} is empty. TabICL needs rows to condition on.")
    return array


def _as_labels(values: ArrayLike | Tensor, n_rows: int) -> np.ndarray:
    """Coerce context labels to integer classes, or say why they cannot be."""
    array = values.detach().cpu().numpy() if isinstance(values, Tensor) else np.asarray(values)
    array = array.ravel()
    if len(array) != n_rows:
        raise ValueError(f"context_y has {len(array)} labels for {n_rows} context rows; they must line up.")
    numeric = array.astype(float)
    if not np.array_equal(numeric, np.round(numeric)):
        raise ValueError(
            "context_y holds continuous values. TabICL conditions on a classification "
            "label, so a survival *time* cannot be used here -- pass the event indicator, "
            "or bin the times yourself and pass the bin. Conditioning on time directly "
            "needs the TabICL regressor checkpoint, which this class does not load."
        )
    return numeric.astype(int)


class TabICLEmbedder(nn.Module):
    """TabICL row representations, conditioned on a context you supply.

    One embedder belongs to one fold, because its context does. Build a new one for
    the next fold from that fold's training rows; there is no ``fit`` and no state to
    reset.

    Args:
        context_x (Tensor | ArrayLike): ``(n, d)`` in-context example features, already
            through this fold's preprocessor. Pass **training rows only** -- a context
            drawn from the whole cohort leaks held-out rows into every representation,
            and no amount of correct ``ColumnTransformer`` handling catches it.
        context_y (Tensor | ArrayLike): ``(n,)`` labels for those rows. Whatever you
            want the representation conditioned on: TabICL's column embedder is
            target-aware, so this is a modelling choice. For survival, the event
            indicator -- ``view.batch().target["event"]``.
        trainable (bool): **Required, no default.** ``False`` freezes the weights and
            keeps them in eval mode; ``True`` lets gradients reach them. Fine-tuning a
            pretrained model on a few hundred patients can be destructive, and which
            you did changes the result with no trace in the output.
        random_state (int | None): **Required, no default.** Seeds TabICL's feature
            shuffling. ``None`` is a legitimate choice but not a silent one: with it
            the representations differ between runs and nothing says so.
        checkpoint (str | None, optional): Checkpoint version. ``None`` takes the
            upstream default, downloaded from Hugging Face on first use.
        n_estimators (int, optional): TabICL's internal ensemble over normalisations
            and feature permutations, averaged across members. Defaults to 1, because
            the ensemble was designed to average *predictions*; averaging
            *representations* is a different and unvalidated operation.
        device (str | None, optional): Torch device. ``None`` lets TabICL choose.

    Attributes:
        out_dim (int): Representation width, read from the checkpoint. Known at
            construction, so the projection layer that consumes it can be built in the
            same breath -- the half of the ``Embedder`` contract a caller reads.
        context_size (int): How many in-context examples this embedder holds.
        n_features (int): Width the context was built at; ``forward`` refuses anything else.

    Example:
        >>> prep = cohort.fit_preprocessor(train_ids)
        >>> train, test = cohort.view(train_ids, prep), cohort.view(test_ids, prep)
        >>> context = train.batch()
        >>> embedder = TabICLEmbedder(
        ...     context_x=context.modalities["clinical"],
        ...     context_y=context.target["event"],
        ...     trainable=False,
        ...     random_state=0,
        ... )
        >>> z_test = embedder(test.batch().modalities["clinical"])      # (n_test, 512)

    Note:
        Context rows see their own label when embedded, so embedding the training rows
        is optimistic relative to held-out ones. That is inherent to in-context
        learning; fixing it properly needs cross-fitted or leave-one-out contexts,
        which this class does not do.

    Note:
        Two things to expect when ``trainable=True``. ``ColEmbedding.forward`` branches
        on ``self.training`` and takes a different route through the network in each
        mode, so training and evaluation representations are not identical -- the same
        contract as dropout, but larger. And only the ~143 of 391 parameters this
        wrapper actually calls receive gradient; the rest belong to the dataset-wise
        transformer, which stops one stage past what we read.
    """

    #: Every call re-embeds the whole context alongside the query rows, so splitting a
    #: split into ``B`` mini-batches does the context work ``B`` times and perturbs the
    #: result at ~1e-5 through chunking. Give it a whole split at once.
    needs_full_batch = True

    def __init__(
        self,
        context_x: Tensor | ArrayLike,
        context_y: Tensor | ArrayLike,
        *,
        trainable: bool,
        random_state: int | None,
        checkpoint: str | None = None,
        n_estimators: int = 1,
        device: str | None = None,
    ):
        super().__init__()
        X = _as_matrix(context_x, "context_x")
        y = _as_labels(context_y, len(X))

        self.trainable = trainable
        self.checkpoint = checkpoint
        self.n_estimators = n_estimators
        self.context_size = len(X)
        self.n_features = X.shape[1]

        TabICLClassifier = _require_tabicl()
        kwargs: dict[str, Any] = {
            "n_estimators": n_estimators,
            "device": device,
            "random_state": random_state,
        }
        if checkpoint is not None:
            kwargs["checkpoint_version"] = checkpoint

        # No gradient step: fitting a TabICLClassifier loads the checkpoint and builds
        # the input pipeline around these rows. It is the supported way in.
        clf = TabICLClassifier(**kwargs)
        clf.fit(X, y)
        self._clf = clf

        self.backbone = self._share_backbone(clf)
        self.out_dim = int(self.backbone.embed_dim * self.backbone.row_num_cls)

        # Loading leaves the backbone in eval mode, and ColEmbedding.forward branches
        # on self.training: the eval branch runs under torch.no_grad(), so a trainable
        # embedder left as loaded would take gradients from nothing and train nothing.
        # Restoring train mode also restores plain nn.Module semantics -- a fresh
        # module is training, and .eval() is how you ask for inference.
        self.backbone.requires_grad_(trainable)
        self.backbone.train(trainable)

    def _share_backbone(self, clf) -> Any:
        """Reuse one copy of the weights across folds, unless they are being trained.

        Typed ``Any`` because the parts of the loaded module this wrapper reaches for
        -- ``col_embedder``, ``row_interactor``, ``embed_dim`` -- are attributes of an
        untyped upstream class, which ``nn.Module`` widens to ``Tensor | Module``.

        Fine-tuning folds must each own their weights, or they would train each
        other's. Frozen ones are read-only, so five folds can share one 110MB module.

        The duplicate this fold just loaded is discarded rather than avoided:
        intercepting the load means monkeypatching a private upstream method, and the
        cost of not doing so is a second of disk I/O.
        """
        if self.trainable:
            return clf.model_
        key = (self.checkpoint, str(clf.device_))
        shared = _BACKBONES.setdefault(key, clf.model_)
        clf.model_ = shared  # drop this fold's copy, or nothing is saved
        return shared

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Embed rows against the stored context.

        Rows are appended after the context and embedded in one pass; the context's
        own representations are discarded. ``mask`` is accepted and ignored, so one
        call site serves ragged and fixed-width modalities alike.

        Args:
            x (Tensor): ``(B, n_features)``. Give it a whole split: the rows do not
                influence each other, but each call re-embeds the entire context
                beside them -- see :attr:`needs_full_batch`.

        Returns:
            Tensor: ``(B, out_dim)``, in the order the rows were given.

        Raises:
            ValueError: If ``x`` is not 2-D, or is a different width than the context.
        """
        if x.ndim != 2:
            raise ValueError(f"{type(self).__name__} expects (n_samples, n_features); got shape {tuple(x.shape)}.")
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"{type(self).__name__} holds a context of {self.n_features} features but was "
                f"given {x.shape[1]}. Every view must be built from the same fold's "
                f"preprocessor -- prep = cohort.fit_preprocessor(train_ids), then "
                f"cohort.view(ids, prep) for each split -- because a one-hot encoder fitted "
                f"on one fold can emit a column another's does not."
            )

        # TabICL's own input handling, then the normalisation the checkpoint was
        # trained with, re-attaching the stored context rows. numpy and fitted on the
        # context, so gradients reach the weights but not x. Nothing upstream of a
        # tabular embedder learns, so that costs nothing today; it is the one seam to
        # replace if it ever does.
        prepared = self._clf.X_encoder_.transform(x.detach().cpu().numpy())
        members = self._clf.ensemble_generator_.transform(prepared, mode="both")

        device = next(self.backbone.parameters()).device
        embedded = []
        for block, labels in members.values():
            rows = torch.as_tensor(block, dtype=torch.float32, device=device)
            context_labels = torch.as_tensor(labels, dtype=torch.float32, device=device)
            columns = self.backbone.col_embedder(
                rows,
                y_train=context_labels,
                embed_with_test=_EMBED_WITH_TEST,
                mgr_config=self._clf.inference_config_.COL_CONFIG,
            )
            interacted = self.backbone.row_interactor(columns, mgr_config=self._clf.inference_config_.ROW_CONFIG)
            embedded.append(interacted[:, context_labels.shape[1] :, :])

        # (n_members, B, out_dim) -> mean over members. A no-op at n_estimators=1.
        return torch.cat(embedded, dim=0).mean(dim=0).float()

    def train(self, mode: bool = True) -> TabICLEmbedder:
        """Keep frozen weights in eval mode when the surrounding model switches to train.

        ``ColEmbedding.forward`` branches on ``self.training``, and a frozen backbone is
        shared between folds -- so without this, one fold entering training would change
        what every other fold computes.
        """
        super().train(mode)
        if not self.trainable:
            self.backbone.eval()
        return self

    def __repr__(self) -> str:
        parts = [
            f"context {self.context_size}x{self.n_features} -> {self.out_dim}d",
            "trainable" if self.trainable else "frozen",
        ]
        if self.n_estimators != 1:
            parts.append(f"n_estimators={self.n_estimators}")
        return f"{type(self).__name__}({' | '.join(parts)})"
