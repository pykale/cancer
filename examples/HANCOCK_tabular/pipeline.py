import torch

from config_handler import (
    build_dataset,
    build_encoder,
    build_head,
    build_optimiser,
    c_index,
    config_from_cli,
    cox_ph_loss,
    section,
    set_seed,
    split_dataset,
)

# Loading, transforming and encdoding tabular data
# Run with your own config using:
#     python examples/HANCOCK_tabular/main.py --config my_config.yaml
config = config_from_cli()
set_seed(config)

cohort = build_dataset(config)

train, test = split_dataset(cohort, config)

train_transformed = train.fit_transform()
test_transformed = train_transformed.transform(test)

encoder = build_encoder(config)

encoder = encoder.fit(train_transformed)

encoded_train = encoder.encode(train_transformed)
encoded_test = encoder.encode(test_transformed)

print(f"encoded_train.shape: {encoded_train.shape}")
print(f"encoded_test.shape: {encoded_test.shape}")


# --------------------------------------------------------------------------- #
# Demo only: a trainable head on top of the frozen TabICL representations.
# Not the intended API -- this all belongs in kalecancer.survival / .evaluate
# eventually. The head, loss and metric are in pipeline.py; the loop stays here.
# --------------------------------------------------------------------------- #

# Supervision, in the same row order the encoder returned.
if train.target is None or test.target is None:
    raise ValueError("This pipeline needs a survival target; set 'target:' in the config.")

train_time = torch.tensor(train.target.times_for(train.identifiers), dtype=torch.float32)
train_event = torch.tensor(train.target.events_for(train.identifiers), dtype=torch.float32)
test_time = torch.tensor(test.target.times_for(test.identifiers), dtype=torch.float32)
test_event = torch.tensor(test.target.events_for(test.identifiers), dtype=torch.float32)


# The trainable linear layer, which takes the frozen 512d TabICL embeddings.
head = section(config, "head")
model = build_head(encoded_train.shape[1], config)
optimiser = build_optimiser(model, config)

# Full batch: the Cox risk set is defined over the whole cohort, so minibatching it
# would be an approximation. A few hundred patients easily fit.
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
