from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aic_signal_harness import (
    BackendKind,
    HarnessIOError,
    PromotionDecision,
    read_ledger_entries,
    read_json,
)
from aic_signal_harness.live_eval import LiveEvalFinalization, finalize_live_eval_run


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_scoring_yaml(path: Path, *, total: float = 7.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
total: {total}
trial_1:
  tier_1:
    score: 1
  tier_2:
    score: 2.5
  tier_3:
    score: {total - 3.5}
""",
        encoding="utf-8",
    )


def _write_policy_trace(path: Path, *, run_id: str = "live-run-a") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "schema_version": 1,
            "run_id": run_id,
            "trial_id": "task_1__policy_call_0001",
            "event_index": 0,
            "event_type": "task_started",
            "elapsed_sec": 0.0,
            "emitted_at_utc": "2026-04-24T00:00:00Z",
            "source": "pytest",
            "leakage_class": "legal_policy_input",
            "payload": {"task_id": "task_1"},
        },
        {
            "schema_version": 1,
            "run_id": run_id,
            "trial_id": "task_1__policy_call_0001",
            "event_index": 1,
            "event_type": "action_published",
            "elapsed_sec": 0.1,
            "emitted_at_utc": "2026-04-24T00:00:01Z",
            "source": "pytest",
            "leakage_class": "legal_policy_input",
            "payload": {
                "linear": [0.0, 0.0, 0.0],
                "angular": [0.0, 0.0, 0.0],
                "frame_id": "base_link",
            },
        },
    ]
    path.write_text("\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n")


def test_live_eval_module_executes_with_warnings_as_errors() -> None:
    result = subprocess.run(
        [sys.executable, "-W", "error", "-m", "aic_signal_harness.live_eval", "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Finalize a live AIC eval" in result.stdout


def test_finalize_live_eval_run_writes_neutral_harness_artifacts(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    policy_trace = harness_root / "policy_trace.jsonl"
    ledger_path = harness_root / "ledger.jsonl"
    baseline_path = harness_root / "baseline.json"
    _write_scoring_yaml(scoring_yaml)
    _write_policy_trace(policy_trace)

    finalization = finalize_live_eval_run(
        run_id="live-run-a",
        result_root=result_root,
        harness_root=harness_root,
        scoring_yaml=scoring_yaml,
        policy_trace=policy_trace,
        ledger_path=ledger_path,
        update_baseline_path=baseline_path,
        bootstrap_promotion=True,
        model_image="aic-lewm-learned:live-run-a",
        model_image_id="sha256:" + "a" * 64,
        runtime_env={
            "AIC_LEWM_PLANNER_MODE": "lewm_mpc",
            "AIC_LEWM_CONTROL_HZ": "4",
            "AIC_LEWM_CHECKPOINT": "/opt/aic_lewm/aic_lewm_epoch_100_object.ckpt",
        },
        generated_at_utc="2026-04-24T00:00:02Z",
    )

    assert isinstance(finalization, LiveEvalFinalization)
    assert finalization.manifest.backend.backend_kind is BackendKind.lewm_world_model
    assert finalization.manifest.backend.runtime_boundary is not None
    assert finalization.manifest.backend.runtime_boundary.policy_artifact is not None
    assert finalization.manifest.backend.runtime_boundary.policy_artifact.sha256 == "a" * 64
    assert "AIC_LEWM_CHECKPOINT" in finalization.manifest.backend.provenance["runtime_env"]
    assert "AIC_LEWM_CHECKPOINT" not in finalization.manifest.backend.config["runtime_env"]
    assert finalization.manifest_artifact.sha256 is not None

    for path in (
        harness_root / "score_report.json",
        harness_root / "scoring_yaml_artifact.json",
        harness_root / "policy_trace_report.json",
        harness_root / "policy_trace_artifact.json",
        harness_root / "episode_trace.json",
        harness_root / "training_signal_report.json",
        harness_root / "run_manifest.json",
        harness_root / "promotion_report.json",
        harness_root / "reward_report.json",
        harness_root / "failure_report.json",
        harness_root / "next_experiment.json",
        harness_root / "ledger_entry.json",
        harness_root / "live_eval_summary.json",
        baseline_path,
    ):
        assert path.is_file(), path

    manifest = read_json(harness_root / "run_manifest.json")
    assert manifest["score"]["total"] == 7.5
    assert {artifact["kind"] for artifact in manifest["artifacts"]} == {
        "scoring_yaml",
        "policy_trace_jsonl",
    }
    assert len(read_ledger_entries(ledger_path)) == 1
    assert PromotionDecision.from_dict(read_json(baseline_path)).decision.value == "bootstrap"
    episode_trace = read_json(harness_root / "episode_trace.json")
    assert episode_trace["run_id"] == "live-run-a"
    assert episode_trace["trials"][0]["event_count"] == 2
    training_signals = read_json(harness_root / "training_signal_report.json")
    assert training_signals["source_trace"]["kind"] == "episode_trace"
    assert {signal["kind"] for signal in training_signals["signals"]} >= {
        "official_score_term",
        "reward_term",
        "failure_label",
    }
    summary = read_json(harness_root / "live_eval_summary.json")
    assert summary["next_experiment_path"] == str(harness_root / "next_experiment.json")
    assert summary["episode_trace_path"] == str(harness_root / "episode_trace.json")
    assert summary["training_signal_report_path"] == str(
        harness_root / "training_signal_report.json"
    )


def test_finalize_live_eval_run_rejects_missing_declared_baseline(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    _write_scoring_yaml(scoring_yaml)

    with pytest.raises(HarnessIOError, match="baseline does not exist"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            baseline_path=harness_root / "missing-baseline.json",
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not harness_root.exists()


def test_finalize_live_eval_run_rejects_missing_declared_baseline_with_bootstrap(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    _write_scoring_yaml(scoring_yaml)

    with pytest.raises(HarnessIOError, match="baseline does not exist"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            baseline_path=harness_root / "stale-baseline.json",
            bootstrap_promotion=True,
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not harness_root.exists()


def test_finalize_live_eval_run_rejects_missing_declared_policy_trace_without_outputs(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    _write_scoring_yaml(scoring_yaml)

    with pytest.raises(HarnessIOError, match="policy trace does not exist"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            policy_trace=harness_root / "missing-policy-trace.jsonl",
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not harness_root.exists()


def test_finalize_live_eval_run_rejects_empty_declared_policy_trace_without_outputs(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    policy_trace = harness_root / "policy_trace.jsonl"
    _write_scoring_yaml(scoring_yaml)
    policy_trace.parent.mkdir(parents=True, exist_ok=True)
    policy_trace.write_text("", encoding="utf-8")

    with pytest.raises(HarnessIOError, match="policy trace is empty"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            policy_trace=policy_trace,
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not (harness_root / "score_report.json").exists()


def test_finalize_live_eval_run_rejects_policy_trace_from_another_run_without_outputs(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    policy_trace = result_root / "traces" / "policy_trace.jsonl"
    _write_scoring_yaml(scoring_yaml)
    _write_policy_trace(policy_trace, run_id="other-run")

    with pytest.raises(HarnessIOError, match="policy trace run_id"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            policy_trace=policy_trace,
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not harness_root.exists()


def test_finalize_live_eval_run_updates_baseline_only_after_durable_outputs(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    baseline_path = harness_root / "baseline.json"
    _write_scoring_yaml(scoring_yaml)
    harness_root.mkdir(parents=True)
    (harness_root / "reward_report.json").write_text("occupied\n", encoding="utf-8")

    with pytest.raises(HarnessIOError):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            update_baseline_path=baseline_path,
            bootstrap_promotion=True,
            model_image_id="sha256:" + "a" * 64,
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not baseline_path.exists()
    assert not (harness_root / "promotion_report.json").exists()


def test_finalize_live_eval_run_rejects_duplicate_ledger_when_overwriting_outputs(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    ledger_path = harness_root / "ledger.jsonl"
    _write_scoring_yaml(scoring_yaml)

    finalize_live_eval_run(
        run_id="live-run-a",
        result_root=result_root,
        harness_root=harness_root,
        scoring_yaml=scoring_yaml,
        ledger_path=ledger_path,
        planner_mode="replay",
        write_next_experiment=False,
        generated_at_utc="2026-04-24T00:00:02Z",
    )

    with pytest.raises(HarnessIOError, match="ledger already contains run_id=live-run-a"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            ledger_path=ledger_path,
            planner_mode="replay",
            write_next_experiment=False,
            overwrite=True,
            generated_at_utc="2026-04-24T00:00:03Z",
        )
    assert len(read_ledger_entries(ledger_path)) == 1


def test_finalize_live_eval_run_requires_immutable_image_id_for_lewm_policy(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    _write_scoring_yaml(scoring_yaml)

    with pytest.raises(HarnessIOError, match="model_image_id"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            model_image="aic-lewm-learned:live-run-a",
            write_next_experiment=False,
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not harness_root.exists()


def test_finalize_live_eval_run_can_skip_next_experiment_without_promotion(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    _write_scoring_yaml(scoring_yaml, total=50.0)

    finalization = finalize_live_eval_run(
        run_id="live-run-a",
        result_root=result_root,
        harness_root=harness_root,
        scoring_yaml=scoring_yaml,
        planner_mode="replay",
        write_next_experiment=False,
        append_ledger=False,
        generated_at_utc="2026-04-24T00:00:02Z",
    )

    assert finalization.manifest.backend.backend_kind is BackendKind.replay_servo
    assert finalization.next_experiment_path is None
    assert finalization.episode_trace_path is None
    assert finalization.training_signal_report_path is None
    assert not (harness_root / "next_experiment.json").exists()
    assert not (harness_root / "episode_trace.json").exists()
    assert not (harness_root / "training_signal_report.json").exists()
