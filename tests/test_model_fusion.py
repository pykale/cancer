import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from kalecancer.loaddata import Classification, MultimodalDataset, PatchFeatures, TimeToEvent
from kalecancer.model import (
    ABMIL,
    ClassificationHead,
    Concat,
    CoxHead,
    EarlyFusion,
    InContextModule,
    IntermediateFusion,
    LateFusion,
    MajorityVote,
    MaskedMean,
    StageList,
    Unimodal,
)


class RecordingStage(InContextModule):
    """Pass-through stage that records the ids it sees and the data it was fitted on."""

    def __init__(self, width: int):
        super().__init__(context_label="event", context_folds=2, random_state=0)
        self.out_dim = width
        self.seen_ids: list[list[str]] = []
        self.fit_ids: list[str] | None = None
        self.fit_x = None
        self.fit_target = None

    def fit(self, x, target, ids):
        self.fit_x, self.fit_target, self.fit_ids = x, target, ids

    def forward(self, x, ids):
        self.seen_ids.append(list(ids))
        return x


def tables(cohort):
    rng = np.random.default_rng(1)
    clinical = pd.DataFrame(
        {"age": (cohort.clinical["age"] - 60) / 10, "noise": rng.normal(size=len(cohort.ids))},
        index=cohort.clinical.index,
    )
    lab = pd.DataFrame(rng.normal(size=(len(cohort.ids), 3)), index=cohort.clinical.index, columns=["a", "b", "c"])
    return clinical, lab


def make_data(cohort, required, task="survival", with_lab=False):
    clinical, lab = tables(cohort)
    modalities = {"clinical": clinical, "wsi": PatchFeatures(cohort.wsi_files, multiple_files="concatenate")}
    if with_lab:
        modalities["lab"] = lab
    target = (
        TimeToEvent(cohort.time, cohort.event)
        if task == "survival"
        else Classification(cohort.labels, classes=["low", "high"])
    )
    return MultimodalDataset(modalities, target=target, required_modalities=list(required))


def whole_batch(data):
    return data.collate([data[i] for i in range(len(data))])


def intermediate(fusion=None, head_dim=8):
    return IntermediateFusion(
        encoding={
            "clinical": [nn.Linear(2, 4)],
            "wsi": [ABMIL(in_dim=16, hidden_dim=6, attention_dim=4, dropout=0.0), nn.Linear(6, 4)],
        },
        fusion=fusion or Concat(),
        head=CoxHead(in_dim=head_dim, ties="efron"),
    )


# ---------------------------------------------------------------- stage lists


def test_stage_list_checks_links_fit_position_and_type():
    with pytest.raises(ValueError, match=r"\[1\] Linear expects width 7 but receives 6"):
        StageList([ABMIL(in_dim=16, hidden_dim=6, attention_dim=4, dropout=0.0), nn.Linear(7, 3)], "wsi")
    with pytest.raises(ValueError, match="only the first stage may be fitted"):
        StageList([nn.Linear(2, 2), RecordingStage(2)], "clinical")
    with pytest.raises(TypeError, match="list of stages"):
        StageList(nn.Linear(2, 2), "clinical")
    assert StageList([nn.Linear(2, 4), nn.ReLU(), nn.Dropout(0.1)], "clinical").output_width(2) == 4


def test_head_width_is_checked_at_construction():
    with pytest.raises(ValueError, match="expects width 7 but receives 8"):
        intermediate(head_dim=7)


def test_stage_errors_name_the_stage_and_its_input():
    stages = StageList([nn.Linear(16, 4)], "wsi")
    with pytest.raises(TypeError) as caught:
        stages([torch.zeros(5, 16), torch.zeros(3, 16)], ["001", "002"])
    assert caught.value.__notes__ == [
        "in encoding['wsi'][0] Linear, called on a list of 2 tensors shaped (5, 16), (3, 16)"
    ]


# ---------------------------------------------------------------- intermediate fusion


def test_intermediate_concat_forward_loss_and_gradients(cohort):
    data = make_data(cohort, required=["clinical", "wsi"])
    model = intermediate()
    model.check(data)
    output = model(whole_batch(data))
    assert output.prediction.shape == (30, 1) and bool(output.defined.all())
    loss = model.loss(output, whole_batch(data)["target"])["loss"]
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_id_aware_stages_are_fitted_on_and_see_present_patients_only(cohort):
    data = make_data(cohort, required=["clinical"])
    recorder = RecordingStage(16)
    model = IntermediateFusion(
        encoding={
            "clinical": [nn.Linear(2, 4)],
            "wsi": [recorder, ABMIL(in_dim=16, hidden_dim=4, attention_dim=4, dropout=0.0)],
        },
        fusion=MaskedMean(),
        head=CoxHead(in_dim=4, ties="efron"),
    )
    with_slides = [f"{k:03d}" for k in range(1, 31)]
    model.fit(data)
    assert recorder.fit_ids == with_slides and len(recorder.fit_x) == 30
    assert recorder.fit_target["event"].shape == (30,)
    model(whole_batch(data))
    assert recorder.seen_ids == [with_slides]


def test_masked_mean_keeps_patients_missing_a_modality_with_finite_gradients(cohort):
    data = make_data(cohort, required=["clinical"])
    model = intermediate(fusion=MaskedMean(), head_dim=4)
    model.check(data)
    batch = whole_batch(data)
    output = model(batch)
    assert bool(output.defined.all()) and torch.isfinite(output.prediction).all()
    model.loss(output, batch["target"])["loss"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_concat_with_a_missing_modality_raises_at_check(cohort):
    data = make_data(cohort, required=["clinical"])
    with pytest.raises(
        ValueError,
        match=r"Concat cannot combine 10 patients \(.*lack each of the modalities: \{'clinical': 0, 'wsi': 10\}",
    ):
        intermediate().check(data)


def test_stage_lists_must_end_with_vectors(cohort):
    data = make_data(cohort, required=["clinical", "wsi"])
    model = IntermediateFusion(encoding={"wsi": []}, fusion=Concat(), head=CoxHead(in_dim=16, ties="efron"))
    with pytest.raises(TypeError, match=r"encoding\['wsi'\] must end with \(n, d\) vectors, got list"):
        model(whole_batch(data))


def test_head_must_match_target_kind(cohort):
    data = make_data(cohort, required=["clinical", "wsi"], task="classification")
    with pytest.raises(TypeError, match="time-to-event"):
        intermediate().check(data)


# ---------------------------------------------------------------- unimodal and early fusion


def test_standalone_unimodal_requires_its_modality(cohort):
    model = Unimodal("wsi", [ABMIL(in_dim=16, hidden_dim=4, attention_dim=4, dropout=0.0)], CoxHead(4, "efron"))
    with pytest.raises(ValueError, match="10 patients lack 'wsi'"):
        model.check(make_data(cohort, required=["clinical"]))
    model.check(make_data(cohort, required=["wsi"]))


def test_early_fusion_fits_and_runs_on_concatenated_rows_in_forward_order(cohort):
    data = make_data(cohort, required=["clinical", "lab"], with_lab=True)
    recorder = RecordingStage(5)
    model = EarlyFusion(["clinical", "lab"], Concat(), [recorder, nn.Linear(5, 4)], CoxHead(4, "efron"))
    model.check(data)
    train = data.subset(data.ids[:20])
    model.fit(train)
    clinical, lab = tables(cohort)
    expected = torch.tensor(np.concatenate([clinical.loc[train.ids].to_numpy(), lab.loc[train.ids].to_numpy()], axis=1))
    torch.testing.assert_close(recorder.fit_x, expected.float())
    assert recorder.fit_ids == train.ids
    assert model(whole_batch(train)).prediction.shape == (20, 1)


# ---------------------------------------------------------------- late fusion


def late(fusion, task="classification"):
    head = (lambda d: ClassificationHead(d, 2)) if task == "classification" else (lambda d: CoxHead(d, "efron"))
    return LateFusion(
        branches={
            "clinical": Unimodal("clinical", [nn.Linear(2, 4)], head(4)),
            "wsi": Unimodal("wsi", [ABMIL(in_dim=16, hidden_dim=4, attention_dim=4, dropout=0.0)], head(4)),
        },
        fusion=fusion,
    )


def test_late_fusion_loss_is_sum_of_branch_losses_and_prediction_uses_mean_logits(cohort):
    data = make_data(cohort, required=["clinical", "wsi"], task="classification")
    model = late(MaskedMean())
    model.check(data)
    batch = whole_batch(data)
    output = model(batch)
    losses = model.loss(output, batch["target"])
    branch_sum = sum(model.branches[n].loss(output.branches[n], batch["target"])["loss"] for n in model.branches)
    torch.testing.assert_close(losses["loss"], branch_sum)
    mean_logits = (output.branches["clinical"].output + output.branches["wsi"].output) / 2
    torch.testing.assert_close(output.prediction, torch.softmax(mean_logits, dim=-1))
    assert model.columns(data.target.info()) == ["probability[low]", "probability[high]"]


def test_late_fusion_cox_masked_mean_needs_complete_branches(cohort):
    with pytest.raises(ValueError, match="arbitrary offset"):
        late(MaskedMean(), task="survival").check(make_data(cohort, required=["clinical"]))
    late(MaskedMean(), task="survival").check(make_data(cohort, required=["clinical", "wsi"]))


def test_late_fusion_with_missing_branch_rows(cohort):
    data = make_data(cohort, required=["clinical"], task="classification")
    model = late(MajorityVote(tie_break="mean_probability"))
    model.check(data)
    batch = whole_batch(data)
    output = model(batch)
    wsi = output.branches["wsi"]
    assert int(wsi.defined.sum()) == 30 and torch.isnan(wsi.prediction[~wsi.defined]).all()
    assert bool(output.defined.all()) and torch.isfinite(output.prediction).all()
    assert model.columns(data.target.info()) == ["vote_score[low]", "vote_score[high]"]
    model.loss(output, batch["target"])["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_late_fusion_rejects_mixed_heads_and_resolves_stages():
    with pytest.raises(TypeError, match="one kind"):
        LateFusion(
            {
                "a": Unimodal("clinical", [nn.Linear(2, 1)], CoxHead(1, "efron")),
                "b": Unimodal("clinical", [nn.Linear(2, 1)], ClassificationHead(1, 2)),
            },
            MaskedMean(),
        )
    hybrid = LateFusion(
        {
            "tables": Unimodal("clinical", [nn.Linear(2, 2)], ClassificationHead(2, 2)),
            "both": IntermediateFusion({"clinical": [nn.Linear(2, 2)], "wsi": []}, Concat(), ClassificationHead(18, 2)),
        },
        MaskedMean(),
    )
    with pytest.raises(ValueError, match="2 branches match modality 'clinical'"):
        hybrid.stages_for("clinical")
    assert isinstance(hybrid.stages_for("clinical", branch="both")[0], nn.Linear)


def test_models_reject_fusion_methods_for_another_stage_at_construction():
    with pytest.raises(TypeError, match="MajorityVote is for late fusion, not intermediate fusion"):
        intermediate(fusion=MajorityVote(tie_break="error"))
    with pytest.raises(TypeError, match="MajorityVote is for late fusion, not early fusion"):
        EarlyFusion(["clinical", "lab"], MajorityVote(tie_break="error"), [nn.Linear(5, 4)], CoxHead(4, "efron"))
    with pytest.raises(TypeError, match="Concat is for early or intermediate fusion, not late fusion"):
        late(Concat())
    with pytest.raises(TypeError, match="fusion must be a FusionMethod"):
        intermediate(fusion=nn.Identity())


def test_late_fusion_branch_rejects_patients_with_only_some_of_its_modalities(cohort):
    data = make_data(cohort, required=["clinical"], task="classification")

    def model(fusion, width):
        both = IntermediateFusion(
            {"clinical": [nn.Linear(2, 4)], "wsi": [ABMIL(in_dim=16, hidden_dim=4, attention_dim=4, dropout=0.0)]},
            fusion,
            ClassificationHead(width, 2),
        )
        return LateFusion(
            {"clinical": Unimodal("clinical", [nn.Linear(2, 2)], ClassificationHead(2, 2)), "both": both}, MaskedMean()
        )

    with pytest.raises(ValueError, match="Concat cannot combine 10 patients"):
        model(Concat(), 8).check(data)
    model(MaskedMean(), 4).check(data)
