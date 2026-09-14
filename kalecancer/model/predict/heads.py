"""Prediction heads: a fixed-width embedding in, one score per patient out.

A head is the last thing a model does and the only part that knows what is being
predicted, which is why it is chosen by a
:class:`~kalecancer.pipeline.task.PredictionTask` rather than fixed by a trainer.

Two heads cover the endpoints this package targets, and the difference between them
is worth knowing because it is forced by the objective, not chosen:

* :class:`LinearHead` keeps its bias. Cross-entropy is not shift-invariant, and the
  intercept is what sets the predicted base rate.
* :class:`CoxHead` drops it. The Cox partial likelihood is invariant to an additive
  shift in log-hazard, so an intercept would be unidentifiable and never train.

The loss each is trained with lives in :mod:`kalecancer.model.predict.losses`.

**On reusing PyKale's head.** ``kale.predict.decode.LinearClassifier`` is the same
layer, and reusing it would be the convention here. It is not imported because
``kale.predict.decode`` pulls in ``GripNet`` and therefore ``torch_geometric``, which
this package does not depend on -- a graph library is a large thing to require for a
single linear layer. The initialisers it applies come from ``kale.utils``, which
imports cleanly, so those are reused instead.
"""

from __future__ import annotations

from kale.utils.initialize_nn import bias_init, xavier_init
from torch import Tensor, nn

__all__ = ["CoxHead", "LinearHead"]


class LinearHead(nn.Module):
    """Linear head with a bias, for a binary or continuous outcome.

    Xavier-normal weights and a zeroed bias, matching what PyKale gives its own
    linear heads: a head reads a fused representation whose scale depends on how many
    modalities were concatenated, so setting the gain from the fan-in beats a fixed
    range.

    Emits a raw score, not a probability. For a binary endpoint that is a logit: the
    fused sigmoid-and-log form of the loss is the numerically stable one, and ROC-AUC
    is unchanged by the sigmoid.

    Args:
        in_features: Width of the embedding the head reads.
        out_features: Number of scores per patient. 1 for a binary endpoint.
    """

    def __init__(self, in_features: int, out_features: int = 1) -> None:
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.linear.apply(xavier_init)
        self.linear.apply(bias_init)

    def forward(self, z: Tensor) -> Tensor:
        """Score embeddings.

        Args:
            z: Embeddings, shape ``(B, D)`` with ``D == self.in_features``.

        Returns:
            Scores, shape ``(B, out_features)``.
        """
        return self.linear(z)


class CoxHead(nn.Module):
    """Linear Cox proportional-hazards head.

    Maps a fixed-width embedding to a single log-hazard score. Bias-free:
    the Cox partial likelihood is invariant to an additive shift in
    log-hazard, so a bias term would be unidentifiable and untrained.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, 1, bias=False)

    def forward(self, z: Tensor) -> Tensor:
        """Compute log-hazard scores.

        Args:
            z: Embeddings, shape ``(B, D)`` with ``D == self.in_features``.

        Returns:
            Log-hazard, shape ``(B, 1)``.

        Raises:
            ValueError: If ``z`` is not 2-D or its last dimension does not
                match ``in_features``.
        """
        if z.dim() != 2 or z.shape[1] != self.in_features:
            raise ValueError(f"CoxHead expects input of shape (B, {self.in_features}), got {tuple(z.shape)}")
        return self.linear(z)
