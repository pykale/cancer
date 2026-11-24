from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kalecancer.loaddata import Classification, MultimodalDataset, PatchFeatures, Tabular, TimeToEvent
from kalecancer.model import (
    ABMIL,
    ClassificationEndpoint,
    ClassProbabilities,
    Concat,
    CoxEndpoint,
    CoxOutput,
    EarlyFusion,
    EncoderList,
    FittedEncoder,
    IntermediateFusion,
    LateFusion,
    MajorityVote,
    MaskedMean,
    RiskScore,
    Unimodal,
    VoteScores,
)


class RecordingEncoder(FittedEncoder):
    """Pass-through fitted encoder that records the ids it sees and the data it was fitted on."""

    def __init__(self):
        super().__init__()
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
    modalities = {"clinical": Tabular(clinical), "wsi": PatchFeatures(cohort.wsi_files, multiple_files="concatenate")}
    if with_lab:
        modalities["lab"] = Tabular(lab)
    target = (
        TimeToEvent(cohort.time, cohort.event)
        if task == "survival"
        else Classification(cohort.labels, classes=["low", "high"])
    )
    return MultimodalDataset(modalities, target=target, required_modalities=list(required))


def whole_batch(data):
    torch_data = data.as_torch()
    return torch_data.collate([torch_data[i] for i in range(len(torch_data))])


def intermediate(fusion=None, head_dim=8):
    return IntermediateFusion(
        encoders={
            "clinical": [nn.Linear(2, 4)],
            "wsi": [ABMIL(in_dim=16, hidden_dim=6, attention_dim=4, dropout=0.0), nn.Linear(6, 4)],
        },
        fusion=fusion or Concat(),
        endpoint=CoxEndpoint(nn.Linear(head_dim, 1, bias=False), "efron"),
    )


# ---------------------------------------------------------------- encoder lists


def test_encoder_list_checks_fit_position_and_type():
    with pytest.raises(ValueError, match="only the first encoder may be fitted"):
        EncoderList([nn.Linear(2, 2), RecordingEncoder()], "clinical")
    with pytest.raises(TypeError, match="list of encoders"):
        EncoderList(nn.Linear(2, 2), "clinical")


def test_widths_are_not_checked_at_construction():
    # widths are checked by the dry run in Pipeline.fit, against the real embeddings
    intermediate(head_dim=7)
    EncoderList([ABMIL(in_dim=16, hidden_dim=6, attention_dim=4, dropout=0.0), nn.Linear(7, 3)], "wsi")


def test_encoder_errors_name_the_encoder_and_its_input():
    encoders = EncoderList([nn.Linear(16, 4)], "wsi")
    with pytest.raises(TypeError) as caught:
        encoders([torch.zeros(5, 16), torch.zeros(3, 16)], ["001", "002"])
    assert caught.value.__notes__ == [
        "in encoders['wsi'][0] Linear, called on a list of 2 tensors shaped (5, 16), (3, 16)"
    ]


# ---------------------------------------------------------------- intermediate fusion


def test_intermediate_concat_forward_loss_and_gradients(cohort):
    data = make_data(cohort, required=["clinical", "wsi"])
    model = intermediate()
    model.check(data)
    output = model(whole_batch(data))
    assert output.risk.shape == (30,) and bool(output.defined.all())
    loss = model.loss(output, whole_batch(data)["target"])["loss"]
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_fitted_encoders_are_fitted_on_and_see_present_patients_only(cohort):
    data = make_data(cohort, required=["clinical"])
    recorder = RecordingEncoder()
    model = IntermediateFusion(
        encoders={
            "clinical": [nn.Linear(2, 4)],
            "wsi": [recorder, ABMIL(in_dim=16, hidden_dim=4, attention_dim=4, dropout=0.0)],
        },
        fusion=MaskedMean(),
        endpoint=CoxEndpoint(nn.Linear(4, 1, bias=False), "efron"),
    )
    with_slides = [f"{k:03d}" for k in range(1, 31)]
    model.fit(data)
    assert recorder.fit_ids == with_slides and len(recorder.fit_x) == 30
    assert recorder.fit_target is data.target
    model(whole_batch(data))
    assert recorder.seen_ids == [with_slides]


def test_masked_mean_keeps_patients_missing_a_modality_with_finite_gradients(cohort):
    data = make_data(cohort, required=["clinical"])
    model = intermediate(fusion=MaskedMean(), head_dim=4)
    model.check(data)
    batch = whole_batch(data)
    output = model(batch)
    assert bool(output.defined.all()) and torch.isfinite(output.risk).all()
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


def test_encoder_lists_must_end_with_vectors(cohort):
    data = make_data(cohort, required=["clinical", "wsi"])
    model = IntermediateFusion(
        encoders={"wsi": []}, fusion=Concat(), endpoint=CoxEndpoint(nn.Linear(16, 1, bias=False), "efron")
    )
    with pytest.raises(TypeError, match=r"encoders\['wsi'\] must output an \(n, d\) tensor, got a list"):
        model(whole_batch(data))


def test_endpoint_must_match_target_type(cohort):
    data = make_data(cohort, required=["clinical", "wsi"], task="classification")
    with pytest.raises(TypeError, match="time-to-event"):
        intermediate().check(data)


# ---------------------------------------------------------------- unimodal and early fusion


def test_standalone_unimodal_requires_its_modality(cohort):
    model = Unimodal(
        "wsi",
        [ABMIL(in_dim=16, hidden_dim=4, attention_dim=4, dropout=0.0)],
        CoxEndpoint(nn.Linear(4, 1, bias=False), "efron"),
    )
    with pytest.raises(ValueError, match="10 patients lack 'wsi'"):
        model.check(make_data(cohort, required=["clinical"]))
    model.check(make_data(cohort, required=["wsi"]))


def test_early_fusion_fits_and_runs_on_concatenated_rows_in_forward_order(cohort):
    data = make_data(cohort, required=["clinical", "lab"], with_lab=True)
    recorder = RecordingEncoder()
    model = EarlyFusion(
        ["clinical", "lab"], Concat(), [recorder, nn.Linear(5, 4)], CoxEndpoint(nn.Linear(4, 1, bias=False), "efron")
    )
    model.check(data)
    train = data.subset(data.ids[:20])
    model.fit(train)
    clinical, lab = tables(cohort)
    expected = torch.tensor(np.concatenate([clinical.loc[train.ids].to_numpy(), lab.loc[train.ids].to_numpy()], axis=1))
    torch.testing.assert_close(recorder.fit_x, expected.float())
    assert recorder.fit_ids == list(train.ids)
    assert model(whole_batch(train)).risk.shape == (20,)


class GatedConcat(Concat):
    """Concat with a trainable gate, standing in for a trainable early fusion method."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, values, present):
        return self.gate * super().forward(values, present)


def test_early_fusion_refuses_a_trainable_fusion_method_before_a_fitted_encoder(cohort):
    data = make_data(cohort, required=["clinical", "lab"], with_lab=True)
    train = data.subset(data.ids[:20])

    def model(fusion, first):
        return EarlyFusion(
            ["clinical", "lab"], fusion, [first, nn.Linear(5, 4)], CoxEndpoint(nn.Linear(4, 1, bias=False), "efron")
        )

    # the fitted encoder would be fitted on a fused input that training then changes
    with pytest.raises(ValueError, match="GatedConcat has trainable parameters"):
        model(GatedConcat(), RecordingEncoder()).fit(train)
    # a frozen fusion method, or no fitted encoder, is fine; the fitting data carries no gradient
    frozen = GatedConcat().requires_grad_(False)
    recorder = RecordingEncoder()
    model(frozen, recorder).fit(train)
    assert not recorder.fit_x.requires_grad
    model(GatedConcat(), nn.Identity()).fit(train)


def test_models_load_training_inputs_only_for_a_fitted_first_encoder(cohort, monkeypatch):
    data = make_data(cohort, required=["clinical", "wsi"])
    loads = []
    real = data.modalities["wsi"].tensors
    monkeypatch.setattr(data.modalities["wsi"], "tensors", lambda ids: loads.append(len(ids)) or real(ids))
    intermediate().fit(data)
    assert loads == []


def test_early_fusion_rejects_bag_modalities_at_check(cohort):
    model = EarlyFusion(
        ["clinical", "wsi"], Concat(), [nn.Linear(18, 4)], CoxEndpoint(nn.Linear(4, 1, bias=False), "efron")
    )
    with pytest.raises(TypeError, match=r"must be FixedShapeModality\. \['wsi'\] are not"):
        model.check(make_data(cohort, required=["clinical", "wsi"]))


# ---------------------------------------------------------------- late fusion


def late(fusion, task="classification", endpoint=None):
    head = (
        (lambda d: ClassificationEndpoint(nn.Linear(d, 2), nn.CrossEntropyLoss()))
        if task == "classification"
        else (lambda d: CoxEndpoint(nn.Linear(d, 1, bias=False), "efron"))
    )
    return LateFusion(
        branches={
            "clinical": Unimodal("clinical", [nn.Linear(2, 4)], head(4)),
            "wsi": Unimodal("wsi", [ABMIL(in_dim=16, hidden_dim=4, attention_dim=4, dropout=0.0)], head(4)),
        },
        fusion=fusion,
        endpoint=endpoint,
    )


def fused_classification():
    return ClassificationEndpoint(None, nn.CrossEntropyLoss())


def test_late_fusion_loss_is_fused_plus_branch_losses_and_prediction_uses_mean_logits(cohort):
    data = make_data(cohort, required=["clinical", "wsi"], task="classification")
    model = late(MaskedMean(), endpoint=fused_classification())
    model.check(data)
    model.fit(data)
    batch = whole_batch(data)
    output = model(batch)
    losses = model.loss(output, batch["target"])
    branch_sum = sum(model.branches[n].loss(output.branches[n], batch["target"])["loss"] for n in model.branches)
    fused = F.cross_entropy(output.logits, batch["target"]["label"])
    torch.testing.assert_close(losses["loss/fused"], fused)
    torch.testing.assert_close(losses["loss"], fused + branch_sum)
    mean_logits = (output.branches["clinical"].logits + output.branches["wsi"].logits) / 2
    torch.testing.assert_close(output.probabilities, torch.softmax(mean_logits, dim=-1))
    assert isinstance(output, ClassProbabilities) and model.output_type is ClassProbabilities
    assert list(output.detach_cpu().frame().columns) == ["probability[low]", "probability[high]"]


def test_late_fusion_cox_masked_mean_needs_complete_branches(cohort):
    with pytest.raises(ValueError, match="arbitrary offset"):
        late(MaskedMean(), task="survival").check(make_data(cohort, required=["clinical"]))
    late(MaskedMean(), task="survival").check(make_data(cohort, required=["clinical", "wsi"]))


def test_late_fusion_rejects_branch_predicting_for_none(tmp_path):
    # no patient has slides: the wsi branch predicts for nobody, fits nothing, and would later fail
    # (e.g. Breslow baseline fit on zero patients) with an unclear error -> fail early in check instead
    ids = [f"{k:03d}" for k in range(1, 11)]
    clinical = pd.DataFrame(
        {"age": np.full(10, 60.0), "sex": ["female"] * 10, "smoking": ["never"] * 10},
        index=pd.Index(ids, name="patient_id"),
    )
    wsi_files = pd.Series([], dtype=object, index=pd.Index([], name="id"))
    labels = pd.Series(["low"] * 10, index=pd.Index(ids, name="patient_id"))
    cohort = SimpleNamespace(ids=ids, clinical=clinical, wsi_files=wsi_files, labels=labels)
    data = make_data(cohort, required=["clinical"], task="classification")
    with pytest.raises(ValueError, match="the following branches predict"):
        late(MaskedMean(), task="classification").check(data)


def test_the_fused_output_type_comes_from_the_fusion_method_or_the_endpoint():
    # averaged log-hazards have no baseline of their own: they are a relative risk until an endpoint fits one
    assert late(MaskedMean(), task="survival").output_type is RiskScore
    assert late(MaskedMean(), task="survival", endpoint=CoxEndpoint(None, "efron")).output_type is CoxOutput
    assert late(MajorityVote(tie_break="error")).output_type is VoteScores


def test_the_fused_endpoint_must_accept_what_the_fusion_method_outputs():
    with pytest.raises(TypeError, match="ClassificationEndpoint takes a ClassProbabilities, but MaskedMean outputs"):
        late(MaskedMean(), task="survival", endpoint=fused_classification())
    # vote scores are not logits, so no classification endpoint can take them
    with pytest.raises(TypeError, match="but MajorityVote outputs a VoteScores"):
        late(MajorityVote(tie_break="error"), endpoint=fused_classification())
    with pytest.raises(ValueError, match="head=None"):
        late(MaskedMean(), endpoint=ClassificationEndpoint(nn.Linear(2, 2), nn.CrossEntropyLoss()))
    with pytest.raises(ValueError, match="branch_weight"):
        LateFusion(late(MaskedMean()).branches, MaskedMean(), None, branch_weight=-1.0)


def test_branch_weight_scales_the_branch_losses_against_the_fused_loss(cohort):
    data = make_data(cohort, required=["clinical", "wsi"], task="classification")
    model = late(MaskedMean(), endpoint=fused_classification())
    model.fit(data)
    batch = whole_batch(data)
    output = model(batch)
    model.branch_weight = 0.25
    losses = model.loss(output, batch["target"])
    branches = losses["loss/clinical"] + losses["loss/wsi"]
    torch.testing.assert_close(losses["loss"], losses["loss/fused"] + 0.25 * branches)


class PenalisedMean(MaskedMean):
    """MaskedMean with a trainable loss term of its own, which records the output it was given."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.seen = None

    def loss(self, output, target):
        self.seen = output
        return self.weight**2


class DecomposablePenalisedMean(PenalisedMean):
    @property
    def decomposable_loss(self):
        return True


def decomposable_late(fusion):
    """``late`` with classification endpoints that all declare their loss a mean over patients."""

    def endpoint(head):
        return ClassificationEndpoint(head, nn.CrossEntropyLoss(), decomposable_loss=True)

    return LateFusion(
        branches={
            "clinical": Unimodal("clinical", [nn.Linear(2, 4)], endpoint(nn.Linear(4, 2))),
            "wsi": Unimodal(
                "wsi", [ABMIL(in_dim=16, hidden_dim=4, attention_dim=4, dropout=0.0)], endpoint(nn.Linear(4, 2))
            ),
        },
        fusion=fusion,
        endpoint=endpoint(None),
    )


def test_a_fusion_method_without_a_loss_adds_nothing_and_stays_decomposable(cohort):
    data = make_data(cohort, required=["clinical", "wsi"], task="classification")
    model = late(MaskedMean(), endpoint=fused_classification())
    model.fit(data)
    batch = whole_batch(data)
    assert "loss/regulariser" not in model.loss(model(batch), batch["target"])
    assert MaskedMean().decomposable_loss and decomposable_late(MaskedMean()).decomposable_loss


def test_only_late_fusion_adds_the_fusion_methods_loss_term(cohort):
    # in intermediate fusion the output comes from the head, not the method, so the method's term is never asked for
    data = make_data(cohort, required=["clinical", "wsi"])
    fusion = PenalisedMean()
    model = intermediate(fusion, head_dim=4)
    batch = whole_batch(data)
    assert set(model.loss(model(batch), batch["target"])) == {"loss"}
    assert fusion.seen is None
    assert not fusion.decomposable_loss and DecomposablePenalisedMean().decomposable_loss


def test_late_fusion_adds_the_fusion_methods_term_unweighted_and_gives_it_the_branches(cohort):
    data = make_data(cohort, required=["clinical", "wsi"], task="classification")
    fusion = PenalisedMean()
    model = late(fusion, endpoint=fused_classification())
    model.fit(data)
    model.branch_weight = 0.25
    batch = whole_batch(data)
    losses = model.loss(model(batch), batch["target"])
    branches = losses["loss/clinical"] + losses["loss/wsi"]
    torch.testing.assert_close(losses["loss/regulariser"], torch.tensor(0.25))
    torch.testing.assert_close(losses["loss"], losses["loss/fused"] + 0.25 * branches + 0.25)
    losses["loss"].backward()
    assert fusion.weight.grad is not None
    assert set(fusion.seen.branches) == {"clinical", "wsi"} and len(fusion.seen.ids) == len(data.ids)
    # a method with its own term is refused gradient accumulation unless it says the term decomposes
    assert not decomposable_late(PenalisedMean()).decomposable_loss
    assert decomposable_late(DecomposablePenalisedMean()).decomposable_loss


def test_late_fusion_branches_cannot_take_the_names_of_loss_report_parts():
    branches = dict(late(MaskedMean()).branches)
    for reserved in ("fused", "regulariser"):
        with pytest.raises(ValueError, match=f"cannot be called '{reserved}'"):
            LateFusion({**branches, reserved: branches["clinical"]}, MaskedMean(), None)


def test_late_fusion_with_missing_branch_rows(cohort):
    data = make_data(cohort, required=["clinical"], task="classification")
    model = late(MajorityVote(tie_break="mean_probability"))
    model.check(data)
    model.fit(data)
    batch = whole_batch(data)
    output = model(batch)
    wsi = output.branches["wsi"]
    assert int(wsi.defined.sum()) == 30 and torch.isnan(wsi.logits[~wsi.defined]).all()
    assert bool(output.defined.all()) and torch.isfinite(output.scores).all()
    assert type(output) is VoteScores and model.output_type is VoteScores
    assert list(output.detach_cpu().frame().columns) == ["score[low]", "score[high]"]
    model.loss(output, batch["target"])["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_late_fusion_rejects_mixed_heads_and_resolves_encoder_lists():
    with pytest.raises(TypeError, match="one output type"):
        LateFusion(
            {
                "a": Unimodal("clinical", [nn.Linear(2, 1)], CoxEndpoint(nn.Linear(1, 1, bias=False), "efron")),
                "b": Unimodal(
                    "clinical", [nn.Linear(2, 1)], ClassificationEndpoint(nn.Linear(1, 2), nn.CrossEntropyLoss())
                ),
            },
            MaskedMean(),
            None,
        )
    hybrid = LateFusion(
        {
            "tables": Unimodal(
                "clinical", [nn.Linear(2, 2)], ClassificationEndpoint(nn.Linear(2, 2), nn.CrossEntropyLoss())
            ),
            "both": IntermediateFusion(
                {"clinical": [nn.Linear(2, 2)], "wsi": []},
                Concat(),
                ClassificationEndpoint(nn.Linear(18, 2), nn.CrossEntropyLoss()),
            ),
        },
        MaskedMean(),
        None,
    )
    with pytest.raises(ValueError, match="2 branches match modality 'clinical'"):
        hybrid.encoders_for("clinical")
    assert isinstance(hybrid.encoders_for("clinical", branch="both")[0], nn.Linear)


def test_models_reject_fusion_methods_for_another_stage_at_construction():
    with pytest.raises(TypeError, match="MajorityVote is for late fusion, not intermediate fusion"):
        intermediate(fusion=MajorityVote(tie_break="error"))
    with pytest.raises(TypeError, match="MajorityVote is for late fusion, not early fusion"):
        EarlyFusion(
            ["clinical", "lab"],
            MajorityVote(tie_break="error"),
            [nn.Linear(5, 4)],
            CoxEndpoint(nn.Linear(4, 1, bias=False), "efron"),
        )
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
            ClassificationEndpoint(nn.Linear(width, 2), nn.CrossEntropyLoss()),
        )
        return LateFusion(
            {
                "clinical": Unimodal(
                    "clinical", [nn.Linear(2, 2)], ClassificationEndpoint(nn.Linear(2, 2), nn.CrossEntropyLoss())
                ),
                "both": both,
            },
            MaskedMean(),
            None,
        )

    with pytest.raises(ValueError, match="Concat cannot combine 10 patients"):
        model(Concat(), 8).check(data)
    model(MaskedMean(), 4).check(data)
