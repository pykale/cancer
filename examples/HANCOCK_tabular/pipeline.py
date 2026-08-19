"""Config-driven HANCOCK tabular demo: load, encode, and train a Cox head.

The same objects as ``quickstart.py``, wired from ``configs/config.yaml``, plus a
small trainable head so the run ends in a number.

Run with:

    uv run python examples/HANCOCK_tabular/pipeline.py
    uv run python examples/HANCOCK_tabular/pipeline.py --config my_config.yaml

Paths in the config are relative to the directory you run this from.
"""

import torch
from config_handler import (
    build_cohort,
    build_embedder,
    build_head,
    build_optimiser,
    c_index,
    config_from_cli,
    cox_ph_loss,
    section,
    set_seed,
    split_identifiers,
    supervision,
)

config = config_from_cli()
set_seed(config)


# --------------------------------------------------------------------------- #
# 1. Cohort and split
# --------------------------------------------------------------------------- #
# An index over patients: no fitted state, never mutated, built once and read by
# every fold.

cohort = build_cohort(config)
print(cohort)

train_ids, test_ids = split_identifiers(cohort, config)


# --------------------------------------------------------------------------- #
# 2. Prepare the fold
# --------------------------------------------------------------------------- #
# Fit on the training rows, then build both views from that same preprocessor: a
# view cannot be built without naming the statistics it uses.

prep = cohort.fit_preprocessor(train_ids)
train = cohort.view(train_ids, prep)
test = cohort.view(test_ids, prep)

print(f"\n{prep.describe()}")
assert set(test.identifiers).isdisjoint(prep.fitted_on), "test rows leaked into the fitted statistics"


# --------------------------------------------------------------------------- #
# 3. Embed
# --------------------------------------------------------------------------- #
# TabICL conditions on context rows, so its context is fold state like a scaler's
# mean -- built from the training view only. The embedder is handed the rows and the
# labels; `context_label` is resolved by the config layer, not by the embedder.

embedder = build_embedder(config, train)
with torch.no_grad():
    # .cpu() is the demo head's requirement, not the embedder's: like any nn.Module
    # it returns on the device it lives on, so a GPU training loop needs no move.
    encoded_train = embedder(train.batch().modalities["clinical"]).cpu()
    encoded_test = embedder(test.batch().modalities["clinical"]).cpu()

print(f"\n{embedder}")
print(f"encoded_train {tuple(encoded_train.shape)} | encoded_test {tuple(encoded_test.shape)}")


# --------------------------------------------------------------------------- #
# 4. Demo only: a trainable head on the frozen representations
# --------------------------------------------------------------------------- #
# Not the intended API: the head, loss and metric belong in kalecancer.survival and
# the loop in kalecancer.evaluate, once those exist. Kept so the demo ends in a
# c-index rather than a shape.
#
# No CohortDataModule here for the same reason -- the head trains on encoded
# representations, not cohort rows. See quickstart.py for that seam.

train_time, train_event = supervision(cohort, train)
test_time, test_event = supervision(cohort, test)

head = section(config, "head")
model = build_head(encoded_train.shape[1], config)
optimiser = build_optimiser(model, config)

# Full batch: the Cox risk set covers every patient still being followed, so a
# minibatch would silently restrict it to whoever landed in that batch.
log_every = head.get("log_every", 50)
for epoch in range(1, head.get("epochs", 200) + 1):
    optimiser.zero_grad()
    loss = cox_ph_loss(model(encoded_train), train_time, train_event)
    loss.backward()
    optimiser.step()

    if epoch % log_every == 0:
        with torch.no_grad():
            train_c = c_index(model(encoded_train), train_time, train_event)
            test_c = c_index(model(encoded_test), test_time, test_event)
        print(f"epoch {epoch:3d} | loss {loss.item():.4f} | c-index train {train_c:.3f} test {test_c:.3f}")

with torch.no_grad():
    print(f"\nfinal test c-index: {c_index(model(encoded_test), test_time, test_event):.3f}")
