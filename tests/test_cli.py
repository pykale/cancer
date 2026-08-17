"""Tests for the command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalecancer.cli import PRESETS, build_parser, main, resolve_config


def parse(argv: list[str]):
    return build_parser().parse_args(argv)


BASE = ["wsi-survival", "--features", "f", "--clinical", "c"]


def test_flags_reach_the_configuration() -> None:
    cfg = resolve_config(parse([*BASE, "--epochs", "7", "--batch-size", "4", "--seed", "9"]))

    assert cfg.DATASET.FEATURE_ROOT == "f"
    assert cfg.DATASET.CLINICAL_PATH == "c"
    assert cfg.SOLVER.MAX_EPOCHS == 7
    assert cfg.SOLVER.BATCH_SIZE == 4
    assert cfg.SOLVER.SEED == 9


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_resolves(preset: str) -> None:
    cfg = resolve_config(parse([*BASE, "--preset", preset]))

    assert cfg.MODEL.INPUT_DIM == 1024


def test_quick_preset_shortens_the_run() -> None:
    cfg = resolve_config(parse([*BASE, "--preset", "quick"]))

    assert cfg.SOLVER.MAX_EPOCHS == 3
    assert cfg.DATASET.MAX_PATCHES == 256


def test_cv_preset_enables_cross_validation() -> None:
    assert resolve_config(parse([*BASE, "--preset", "cv"])).DATASET.NUM_FOLDS == 5


def test_dss_preset_switches_the_endpoint() -> None:
    assert resolve_config(parse([*BASE, "--preset", "dss"])).SURVIVAL.ENDPOINT == "DSS"


def test_flags_take_precedence_over_the_preset() -> None:
    cfg = resolve_config(parse([*BASE, "--preset", "quick", "--epochs", "42"]))

    assert cfg.SOLVER.MAX_EPOCHS == 42


def test_trailing_overrides_take_final_precedence() -> None:
    cfg = resolve_config(parse([*BASE, "--epochs", "5", "MODEL.DROPOUT", "0.5", "SOLVER.MAX_EPOCHS", "11"]))

    assert cfg.MODEL.DROPOUT == 0.5
    assert cfg.SOLVER.MAX_EPOCHS == 11


def test_config_file_is_applied_before_flags(tmp_path: Path) -> None:
    config_file = tmp_path / "c.yaml"
    config_file.write_text("SOLVER:\n  MAX_EPOCHS: 3\n  BATCH_SIZE: 8\n", encoding="utf-8")

    cfg = resolve_config(parse([*BASE, "--cfg", str(config_file), "--epochs", "20"]))

    assert cfg.SOLVER.BATCH_SIZE == 8
    assert cfg.SOLVER.MAX_EPOCHS == 20


def test_print_config_writes_the_resolved_config(capsys) -> None:
    assert main([*BASE, "--print-config"]) == 0

    assert "FEATURE_ROOT: f" in capsys.readouterr().out


def test_missing_paths_report_a_clear_error(caplog) -> None:
    with caplog.at_level("ERROR"):
        exit_code = main(["wsi-survival", "--features", "absent", "--clinical", "absent"])

    assert exit_code == 2
    assert "path not found" in caplog.text


def test_unmatched_cohort_exits_without_a_traceback(tmp_path: Path, caplog) -> None:
    features = tmp_path / "features"
    features.mkdir()
    clinical = tmp_path / "clinical.json"
    clinical.write_text(json.dumps([]), encoding="utf-8")

    with caplog.at_level("ERROR"):
        exit_code = main(
            ["wsi-survival", "--features", str(features), "--clinical", str(clinical), "--out", str(tmp_path / "o")]
        )

    assert exit_code == 1
    assert "no patients matched" in caplog.text


def test_command_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
