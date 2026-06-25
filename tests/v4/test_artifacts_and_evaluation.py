from __future__ import annotations

import json

import pytest

from tenflix.v4.artifacts import load_bundle, load_test, save_bundle
from tenflix.v4.cli import (
    PromotionRejected,
    REQUIRED_PROMOTION_GATES,
    _all_gates_pass,
    _assert_promoted,
    _promote,
    build_parser,
)
from tenflix.v4.evaluation import evaluate_bundle
from tenflix.v4.pipeline import _tune_temporal_calibration


def test_artifact_round_trip_and_schema_rejection(tmp_path, v4_bundle, v4_ratings):
    path = save_bundle(v4_bundle, v4_ratings.iloc[-5:], tmp_path, "v4-fixture", "evaluation")
    loaded = load_bundle(path)
    assert loaded.schema_version == 4
    assert len(load_test(path)) == 5
    old = tmp_path / "old"
    old.mkdir()
    (old / "COMPLETE").write_text("ok")
    (old / "manifest.json").write_text(json.dumps({"schema_version": 3}))
    with pytest.raises(ValueError, match="Expected V4 artifact schema 4"):
        load_bundle(old)


def test_small_evaluation_persists_honest_gate_result(v4_bundle, v4_ratings):
    context = v4_ratings.groupby("userId", group_keys=False).head(6)
    test = v4_ratings.groupby("userId", group_keys=False).tail(2)
    report = evaluate_bundle(v4_bundle, context, test)
    assert set(report["aggregate"]) == {"popularity", "static_mf", "recent_mf", "full_v4"}
    assert isinstance(report["validated"], bool)
    assert "latency" in report["gates"]
    assert "integrity" in report["gates"]
    assert "noninferiority_margin" in report["gates"]["new_start_ndcg"]
    assert "segments" in report


def test_temporal_calibration_returns_persistable_parameters(v4_bundle, v4_ratings):
    train = v4_ratings.groupby("userId", group_keys=False).head(6)
    validation = v4_ratings.groupby("userId", group_keys=False).tail(2)
    cutoff = int(train["timestamp"].max())
    result = _tune_temporal_calibration(
        v4_bundle, train, validation, cutoff, maximum_users=5
    )
    assert set(result) == {
        "sigmoid_intercept",
        "sigmoid_latent_weight",
        "sigmoid_genre_weight",
        "sigmoid_activity_weight",
    }
    assert all(isinstance(value, float) for value in result.values())


def test_unvalidated_run_cannot_be_promoted(tmp_path):
    run = tmp_path / "failed"
    run.mkdir()
    (run / "evaluation.json").write_text(json.dumps({"validated": False}))
    with pytest.raises(PromotionRejected, match="Promotion rejected"):
        _promote(run)


def test_promotion_error_identifies_failed_coverage(tmp_path):
    run = tmp_path / "coverage-failed"
    run.mkdir()
    gates = {name: {"passed": True} for name in REQUIRED_PROMOTION_GATES}
    gates["coverage"] = {"value": 0.13, "minimum": 0.15, "passed": False}
    (run / "evaluation.json").write_text(
        json.dumps({"validated": False, "gates": gates})
    )
    with pytest.raises(PromotionRejected, match=r"coverage: 0\.1300.*0\.1500"):
        _promote(run)


def test_promotion_error_identifies_new_user_noninferiority_failure(tmp_path):
    run = tmp_path / "new-start-failed"
    run.mkdir()
    gates = {name: {"passed": True} for name in REQUIRED_PROMOTION_GATES}
    gates["new_start_ndcg"] = {
        "lower": -0.02,
        "noninferiority_margin": -0.005,
        "passed": False,
    }
    (run / "evaluation.json").write_text(
        json.dumps({"validated": False, "gates": gates})
    )
    with pytest.raises(PromotionRejected, match=r"lower bound -0\.0200.*-0\.0050"):
        _promote(run)


def test_validated_flag_cannot_override_a_failed_gate():
    assert not _all_gates_pass({"coverage": {"passed": False}})
    assert not _all_gates_pass({})
    assert _all_gates_pass(
        {"accuracy": {"ndcg": {"passed": True}}, "coverage": {"passed": True}}
    )
    assert "new_start_ndcg" in REQUIRED_PROMOTION_GATES


def test_production_training_cannot_bypass_promotion(tmp_path):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--mode", "production"])
    run = tmp_path / "manual-production"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"mode": "production"}))
    with pytest.raises(ValueError, match="created by promote"):
        _assert_promoted(run)
