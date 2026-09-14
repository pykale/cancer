# API overview

### Not a concrete design decision, but rather an idea on how I currently envision the API working

***Workflow encompasses:***

1. loading data
2. preprocess/transform data
3. embedding each modality
4. projecting to a common dimension (MLP)
5. fusing embeddings
6. prediction head
7. evaluation

The training/fine-tuning loop involves steps 3 to 6

```python
from kalecancer.loaddata import TabularDataset, WSIDataset, MultiModalDataset
from kalecancer.model import TabICL, ABMIL
from kalecancer.survival import CoxHead, SurvivalTarget
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

SEED = 0


# Example of TabularDataset object invocation
cohort = TabularDataset(
    source='loc/to/data',
    identifier="patient_id",
    target=SurvivalTarget(
        time="days_to_last_information",
        event="survival_status",
        event_value="deceased",
        unit="days",
    ),

    continuous=["age_at_initial_diagnosis"],
    continuous_transform=[SimpleImputer(strategy="median"), StandardScaler()],

    categorical=["sex", "smoking_status", "primarily_metastasis"],
    categorical_transform=[
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    ]
)


# Pipeline object overview
# The way I envision the PredictionPipeline class is as more of a plumbing
# tool which connects each part in the correct way. It makes it easy for the user
# to use the pipeline (instead of them having to write their own training loops and
# learn our whole API), and also allows for an easy interface for a yaml configuration
# file to be wired up to.
# NOTE: this example is for late fusion only
pipeline = PredictionPipeline(

    # experiment setup
    # NOTE: this could go in the PredictionPipeline.fit(...) function
    experiment_params = {
        'model_output' = 'output/dest/'
        'cross-validation' : KFoldsCV(k = 5),

        # once all cv folds have trained, how do we choose the final model.
        # could be an ensemble of all 5 models (5 folds  produces 5 models), or
        # could be the best model best fold (which model from the 5 folds performs best).
        'cv-model-selection' : 'ensemble',

        'train-test-split' : 0.2,
        'epochs' : 10,
        'metrics' : ['brier', 'c-index'],
        'loss' : 'cox_ph_loss', # cox proportional hazard loss
        ...
    }

    # step 1 and 2 - data loading and transforming
    data = MultiModalDataset(
        datasets = {
            'clinical': clincal, # where clinical is a TabularDataset object
            'wsi' : WSIDataset(...)
        },

        # point to the dataset which contains the Target (must be a TabularDataset?)
        primary_dataset = 'clinical'
    ),

    # step 3 - encoding
    encoders = {
        'clinical': TabICL(finetune = False),
        'wsi': ABMIL(...) # Uses frozen UNI backbone, and can train/finetune MIL
    },

    # step 4 - dimension projection
    projection = Projector(
        common_dim = 128,   # The common dimension to project all embeddings to
        hidden_layers = ...,
        drop_out = 0.2,
        ...
    ),

    # step 5 - fusion
    fusion = Fuser(method = 'poe'),

    # setp 6 - prediciton head
    prediction_head = CoxHead(
        lr = 1e-3,
        ...
    )
)

pipeline.fit() # could have experiment setup in the fit() func

pipeline.evaluate()
print(pipeline.results) # includes a summary, i.e. hazard ratios

feature_importance = pipeline.explain()
print(feature_importance)
```

# Old API design (no longer preferred):

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
    continuous_transform=[
        SimpleImputer(strategy="median", add_indicator=True),
        QuantileTransformer(output_distribution="normal"),
    ],
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
        lr=1e-5,  # encoder lr, well below the head's
        unfreeze="last:4",  # "all" | "none" | "last:N" | ["blocks.10", ...]
        epochs=30,
        early_stopping="c_index",
        patience=5,
        context_size=256,  # rows sampled into context per step
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
    model,
    train,
    n_splits=5,
    n_repeats=1,
    random_state=SEED,
    inner_val=0.15,  # <- carved from each train fold
    keep_models="runs/tabicl_ft/folds",
)
print(report.c_index, report.best_epoch())  # 0.706 ± 0.045   |   12

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
explain(final_model, test).summary()  # hazard ratios / per-feature attribution


# Note: Finetuning on a small cohort can be destructive. Hence I propose this too:
report = compare(
    {
        "frozen": RiskModel(encoders={"clinical": TabICL(freeze=True)}, head=CoxHead()),
        "finetuned": RiskModel(encoders={"clinical": TabICL(finetune=FineTune(...))}, head=CoxHead()),
        "cox_only": RiskModel(head=CoxHead()),  # no encoder, plain covariates
    },
    train,
    n_splits=5,
    random_state=SEED,
    inner_val=0.15,
)
report.summary()
```
