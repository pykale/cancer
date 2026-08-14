# API overview

### Not a concrete design decision, but rather an idea on how I currently envision the API working

```python
from kalecancer.loaddata import TabularDataset, SurvivalTarget, WSIDataset, MultiModalDataset
from kalecancer.model import TabICL, ABMIL, FineTune
from kalecancer.survival import RiskModel, CoxHead
from kalecancer.evaluate import compare, evaluate, cross_validate
from kalecancer.interpret import explain
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer

SEED = 0

# ---- 1. point at your data ----------------------------------------------
cohort = TabularDataset(
    "data/my_cohort.csv",
    identifier="patient_id",
    target=SurvivalTarget(time="os_days", event="vital_status", event_value="dead"),

    continuous=["age"],
    continuous_transform=[SimpleImputer(strategy="median", add_indicator=True),
                      QuantileTransformer(output_distribution="normal")],

    categorical=["sex", "stage", "smoking_status"],
    categorical_transform=OneHotEncoder(handle_unknown="ignore", min_frequency=5),
)
print(cohort)

# # for multimodal, we can eventually do something like this:
# slides = WSIDataset(
#     "cache/uni_20x",
#     identifier="patient_id",
#     tiles_per_slide=2048,          # sampled per epoch — fixes variable-length bags
# )

# cohort = MultiModalDataset(
#     {"clinical": clinical, "wsi": slides},
#     missing="mask",                # keep patients who have no usable slide
# )

# ---- 2. hold out a test set, once ---------------------------------------
train, test = cohort.split(test_size=0.2, random_state=SEED)

# ---- 3. encoder, now fine-tuned -----------------------------------------
encoder = TabICL(
    checkpoint="tabicl-v1",
    out_dim=256,
    finetune=FineTune(
        lr=1e-5,                    # encoder lr, well below the head's
        unfreeze="last:4",          # "all" | "none" | "last:N" | ["blocks.10", ...]
        epochs=30,
        early_stopping="c_index",
        patience=5,
        context_size=256,           # rows sampled into context per step
    ),
)

model = RiskModel(encoders={"clinical": encoder}, head=CoxHead(lr=1e-3))

# # for multimodal, we can eventually do something like this:
# model = RiskModel(
#     encoders={
#         "clinical": TabICL(checkpoint="tabicl-v1", out_dim=256, freeze=True),
#         "wsi":      ABMIL(in_dim=1024, out_dim=256),
#     },
#     fusion="concat",                       # | "bilinear" | "lrtf" | "coattention"
#     head=DiscreteTimeHead(n_bins=4),       # not CoxHead — see below
#     modality_dropout=0.25,
# )


# ---- 4. CV, with an inner split for early stopping ----------------------
report = cross_validate(
    model, train,
    n_splits=5, n_repeats=1, random_state=SEED,
    inner_val=0.15,                        # <- carved from each train fold
    keep_models="runs/tabicl_ft/folds",
)
print(report.c_index, report.best_epoch())     # 0.706 ± 0.045   |   12

# ---- 5. refit on all of train, for a fixed number of epochs -------------
final_model = report.best_model().refit(train, epochs=report.best_epoch())

# alternative, usually more stable:
# final_model = report.ensemble()        # average risk over the K fold models, no refit

# ---- 6. evaluate once on test -------------------------------------------
final = evaluate(final_model, test, metrics=["c_index", "ibs", "auc@24mo"])
print(final.c_index)

final_model.save("runs/tabicl/final")


# ── 7. save and explain ──────────────────────────────────────────────────
final_model.save("runs/final")
result.to_frame().to_csv("runs/test_metrics.csv")
explain(final_model, test).summary()      # hazard ratios / per-feature attribution


# Note: Finetuning on a small cohort can be destructive. Hence I propose this too:
report = compare(
    {
        "frozen":    RiskModel(encoders={"clinical": TabICL(freeze=True)},      head=CoxHead()),
        "finetuned": RiskModel(encoders={"clinical": TabICL(finetune=FineTune(...))}, head=CoxHead()),
        "cox_only":  RiskModel(head=CoxHead()),                    # no encoder, plain covariates
    },
    train, n_splits=5, random_state=SEED, inner_val=0.15,
)
report.summary()
```
