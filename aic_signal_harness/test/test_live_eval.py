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
import aic_signal_harness.live_eval as live_eval_module
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


def _write_policy_trace(
    path: Path,
    *,
    run_id: str = "live-run-a",
    include_official_trial_id: bool = True,
) -> None:
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
            "leakage_class": "legal_policy_action_output",
            "payload": {
                "linear": [0.0, 0.0, 0.0],
                "angular": [0.0, 0.0, 0.0],
                "frame_id": "base_link",
            },
        },
    ]
    if include_official_trial_id:
        for event in events:
            event["official_trial_id"] = "trial_1"
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
            "AIC_LEWM_POLICY_TRACE_TRUST_TASK_OFFICIAL_TRIAL_ID": "1",
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

    runtime_env = finalization.manifest.backend.provenance["runtime_env"]
    assert runtime_env["AIC_LEWM_POLICY_TRACE_TRUST_TASK_OFFICIAL_TRIAL_ID"] == "1"
    assert (
        "AIC_LEWM_POLICY_TRACE_TRUST_TASK_OFFICIAL_TRIAL_ID"
        not in finalization.manifest.backend.config["runtime_env"]
    )

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
        harness_root / "next_experiment_plan.json",
        harness_root / "ledger_entry.json",
        harness_root / "live_eval_summary.json",
        baseline_path,
    ):
        assert path.is_file(), path

    manifest = read_json(harness_root / "run_manifest.json")
    assert manifest["score"]["total"] == 7.5
    artifacts_by_kind = {artifact["kind"]: artifact for artifact in manifest["artifacts"]}
    assert set(artifacts_by_kind) == {
        "scoring_yaml",
        "policy_trace_jsonl",
        "reward_report",
        "failure_report",
        "episode_trace",
        "training_signal_report",
    }
    assert artifacts_by_kind["reward_report"]["provenance"]["source_artifacts"] == [
        {
            "kind": "scoring_yaml",
            "sha256": artifacts_by_kind["scoring_yaml"]["sha256"],
        }
    ]
    assert artifacts_by_kind["failure_report"]["provenance"]["source_artifacts"] == [
        {
            "kind": "scoring_yaml",
            "sha256": artifacts_by_kind["scoring_yaml"]["sha256"],
        }
    ]
    assert len(read_ledger_entries(ledger_path)) == 1
    assert PromotionDecision.from_dict(read_json(baseline_path)).decision.value == "bootstrap"
    episode_trace = read_json(harness_root / "episode_trace.json")
    assert episode_trace["run_id"] == "live-run-a"
    assert episode_trace["trials"][0]["event_count"] == 2
    assert {artifact["kind"] for artifact in episode_trace["source_artifacts"]} == {
        "scoring_yaml",
        "policy_trace_jsonl",
        "reward_report",
        "failure_report",
    }
    training_signals = read_json(harness_root / "training_signal_report.json")
    assert training_signals["source_trace"]["kind"] == "episode_trace"
    assert training_signals["source_trace"]["sha256"]
    assert {signal["kind"] for signal in training_signals["signals"]} >= {
        "official_score_term",
        "failure_label",
    }
    assert all(signal["offline_only"] is True for signal in training_signals["signals"])
    assert all(signal["runtime_allowed"] is False for signal in training_signals["signals"])
    next_plan = read_json(harness_root / "next_experiment_plan.json")
    assert next_plan["source_next_experiment"]["kind"] == "next_experiment_report"
    assert next_plan["source_manifest"]["kind"] == "run_manifest"
    assert all(candidate["offline_only"] is True for candidate in next_plan["candidates"])
    assert all(candidate["autonomous_launch_allowed"] is False for candidate in next_plan["candidates"])
    summary = read_json(harness_root / "live_eval_summary.json")
    assert summary["next_experiment_path"] == str(harness_root / "next_experiment.json")
    assert summary["next_experiment_plan_path"] == str(harness_root / "next_experiment_plan.json")
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


def test_finalize_live_eval_run_requires_scoring_bound_to_result_root(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    bound_scoring_yaml = result_root / "eval" / "scoring.yaml"
    other_scoring_yaml = tmp_path / "other" / "eval" / "scoring.yaml"
    _write_scoring_yaml(bound_scoring_yaml)
    _write_scoring_yaml(other_scoring_yaml)

    with pytest.raises(HarnessIOError, match="result_root/eval/scoring.yaml"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=other_scoring_yaml,
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
            model_image_id="sha256:" + "a" * 64,
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
            model_image_id="sha256:" + "a" * 64,
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not harness_root.exists()


def test_finalize_live_eval_run_rejects_scored_policy_trace_without_official_trial_ids(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    policy_trace = result_root / "traces" / "policy_trace.jsonl"
    _write_scoring_yaml(scoring_yaml)
    _write_policy_trace(policy_trace, include_official_trial_id=False)

    with pytest.raises(HarnessIOError, match="official_trial_id"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            policy_trace=policy_trace,
            model_image_id="sha256:" + "a" * 64,
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not (harness_root / "episode_trace.json").exists()


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


def test_finalize_live_eval_run_appends_ledger_last_after_promotion_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    ledger_path = harness_root / "ledger.jsonl"
    _write_scoring_yaml(scoring_yaml)

    def fail_promotion_write(*args, **kwargs):
        raise HarnessIOError("simulated promotion write failure")

    monkeypatch.setattr(live_eval_module, "write_promotion_decision", fail_promotion_write)

    with pytest.raises(HarnessIOError, match="simulated promotion"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            ledger_path=ledger_path,
            bootstrap_promotion=True,
            model_image_id="sha256:" + "a" * 64,
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not ledger_path.exists()


def test_finalize_live_eval_run_does_not_append_ledger_when_promotion_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    ledger_path = harness_root / "ledger.jsonl"
    _write_scoring_yaml(scoring_yaml)
    original_replace = Path.replace

    def fail_promotion_publish(self: Path, target: Path | str) -> Path:
        target_path = Path(target)
        if self.name.startswith(".promotion_report.json.") and target_path.name == "promotion_report.json":
            raise OSError("simulated promotion replace failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_promotion_publish)

    with pytest.raises(HarnessIOError, match="simulated promotion replace"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            ledger_path=ledger_path,
            bootstrap_promotion=True,
            model_image_id="sha256:" + "a" * 64,
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not ledger_path.exists()
    assert not (harness_root / "promotion_report.json").exists()
    assert not tuple(harness_root.glob(".promotion_report.json.*.tmp"))


def test_finalize_live_eval_run_does_not_update_baseline_when_ledger_append_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    ledger_path = harness_root / "ledger.jsonl"
    baseline_path = harness_root / "baseline.json"
    _write_scoring_yaml(scoring_yaml)

    def fail_ledger_append(*args, **kwargs):
        raise HarnessIOError("simulated ledger append failure")

    monkeypatch.setattr(live_eval_module, "append_ledger_entry", fail_ledger_append)

    with pytest.raises(HarnessIOError, match="simulated ledger"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            ledger_path=ledger_path,
            update_baseline_path=baseline_path,
            bootstrap_promotion=True,
            model_image_id="sha256:" + "a" * 64,
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not baseline_path.exists()
    assert not (harness_root / "promotion_report.json").exists()
    assert not tuple(harness_root.glob(".promotion_report.json.*.tmp"))


def test_finalize_live_eval_run_restores_existing_baseline_when_ledger_append_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_result_root = tmp_path / "seed-result"
    seed_harness_root = seed_result_root / "harness"
    seed_scoring_yaml = seed_result_root / "eval" / "scoring.yaml"
    ledger_path = tmp_path / "ledger.jsonl"
    baseline_path = tmp_path / "baseline.json"
    _write_scoring_yaml(seed_scoring_yaml, total=7.5)
    finalize_live_eval_run(
        run_id="seed-run",
        result_root=seed_result_root,
        harness_root=seed_harness_root,
        scoring_yaml=seed_scoring_yaml,
        ledger_path=ledger_path,
        update_baseline_path=baseline_path,
        bootstrap_promotion=True,
        planner_mode="replay",
        write_next_experiment=False,
        generated_at_utc="2026-04-24T00:00:02Z",
    )
    original_baseline = baseline_path.read_text(encoding="utf-8")
    original_ledger_entries = read_ledger_entries(ledger_path)

    candidate_result_root = tmp_path / "candidate-result"
    candidate_harness_root = candidate_result_root / "harness"
    candidate_scoring_yaml = candidate_result_root / "eval" / "scoring.yaml"
    _write_scoring_yaml(candidate_scoring_yaml, total=9.0)

    def fail_ledger_append(*args, **kwargs):
        raise HarnessIOError("simulated ledger append failure")

    monkeypatch.setattr(live_eval_module, "append_ledger_entry", fail_ledger_append)

    with pytest.raises(HarnessIOError, match="simulated ledger"):
        finalize_live_eval_run(
            run_id="candidate-run",
            result_root=candidate_result_root,
            harness_root=candidate_harness_root,
            scoring_yaml=candidate_scoring_yaml,
            ledger_path=ledger_path,
            baseline_path=baseline_path,
            update_baseline_path=baseline_path,
            min_improvement=1.0,
            planner_mode="replay",
            write_next_experiment=True,
            generated_at_utc="2026-04-24T00:00:03Z",
        )

    assert baseline_path.read_text(encoding="utf-8") == original_baseline
    assert read_ledger_entries(ledger_path) == original_ledger_entries
    assert not (candidate_harness_root / "promotion_report.json").exists()
    assert not tuple(tmp_path.glob(".baseline.json.*.bak"))
    assert not tuple(candidate_harness_root.glob(".promotion_report.json.*.tmp"))


def test_finalize_live_eval_run_stages_baseline_before_ledger_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    ledger_path = harness_root / "ledger.jsonl"
    baseline_path = harness_root / "baseline.json"
    _write_scoring_yaml(scoring_yaml)

    def fail_baseline_write(*args, **kwargs):
        raise HarnessIOError("simulated baseline write failure")

    monkeypatch.setattr(live_eval_module, "write_baseline_decision", fail_baseline_write)

    with pytest.raises(HarnessIOError, match="simulated baseline"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            ledger_path=ledger_path,
            update_baseline_path=baseline_path,
            bootstrap_promotion=True,
            model_image_id="sha256:" + "a" * 64,
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not ledger_path.exists()
    assert not baseline_path.exists()
    assert not (harness_root / "promotion_report.json").exists()
    assert not tuple(harness_root.glob(".baseline.json.*.tmp"))
    assert not tuple(harness_root.glob(".promotion_report.json.*.tmp"))


def test_finalize_live_eval_run_rejects_unpublishable_baseline_before_ledger_append(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    ledger_path = harness_root / "ledger.jsonl"
    baseline_path = harness_root / "baseline.json"
    _write_scoring_yaml(scoring_yaml)
    baseline_path.mkdir(parents=True)

    with pytest.raises(HarnessIOError, match="baseline decision path must not be a directory"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            ledger_path=ledger_path,
            update_baseline_path=baseline_path,
            bootstrap_promotion=True,
            model_image_id="sha256:" + "a" * 64,
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not ledger_path.exists()
    assert not (harness_root / "promotion_report.json").exists()


def test_finalize_live_eval_run_rejects_baseline_update_generated_output_collision(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    ledger_path = harness_root / "ledger.jsonl"
    baseline_path = harness_root / "run_manifest.json"
    _write_scoring_yaml(scoring_yaml)

    with pytest.raises(HarnessIOError, match="generated output"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            ledger_path=ledger_path,
            update_baseline_path=baseline_path,
            bootstrap_promotion=True,
            model_image_id="sha256:" + "a" * 64,
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not ledger_path.exists()
    assert not (harness_root / "promotion_report.json").exists()
    assert not (harness_root / "score_report.json").exists()


def test_finalize_live_eval_run_rejects_baseline_update_ledger_path_collision(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    ledger_path = harness_root / "ledger.jsonl"
    _write_scoring_yaml(scoring_yaml)

    with pytest.raises(HarnessIOError, match="append ledger path"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            ledger_path=ledger_path,
            update_baseline_path=ledger_path,
            bootstrap_promotion=True,
            model_image_id="sha256:" + "a" * 64,
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not harness_root.exists()


def test_finalize_live_eval_run_rejects_ledger_generated_output_collision(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    ledger_path = harness_root / "run_manifest.json"
    _write_scoring_yaml(scoring_yaml)

    with pytest.raises(HarnessIOError, match="append ledger path"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            ledger_path=ledger_path,
            bootstrap_promotion=True,
            model_image_id="sha256:" + "a" * 64,
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not harness_root.exists()


def test_finalize_live_eval_run_requires_ledger_append_for_promotion(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    _write_scoring_yaml(scoring_yaml)

    with pytest.raises(HarnessIOError, match="promotion decisions require"):
        finalize_live_eval_run(
            run_id="live-run-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            bootstrap_promotion=True,
            append_ledger=False,
            model_image_id="sha256:" + "a" * 64,
            generated_at_utc="2026-04-24T00:00:02Z",
        )
    assert not harness_root.exists()


def test_finalize_live_eval_run_rejects_baseline_update_without_ledger_append(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    baseline_path = harness_root / "baseline.json"
    _write_scoring_yaml(scoring_yaml)

    with pytest.raises(HarnessIOError, match="baseline update requires"):
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


def test_finalize_live_eval_run_compares_baseline_and_writes_next_experiment(
    tmp_path: Path,
) -> None:
    baseline_result_root = tmp_path / "baseline-result"
    baseline_harness_root = baseline_result_root / "harness"
    baseline_scoring_yaml = baseline_result_root / "eval" / "scoring.yaml"
    ledger_path = tmp_path / "ledger.jsonl"
    baseline_path = tmp_path / "baseline.json"
    _write_scoring_yaml(baseline_scoring_yaml, total=7.5)

    finalize_live_eval_run(
        run_id="baseline-run",
        result_root=baseline_result_root,
        harness_root=baseline_harness_root,
        scoring_yaml=baseline_scoring_yaml,
        ledger_path=ledger_path,
        update_baseline_path=baseline_path,
        bootstrap_promotion=True,
        planner_mode="replay",
        write_next_experiment=False,
        generated_at_utc="2026-04-24T00:00:02Z",
    )

    candidate_result_root = tmp_path / "candidate-result"
    candidate_harness_root = candidate_result_root / "harness"
    candidate_scoring_yaml = candidate_result_root / "eval" / "scoring.yaml"
    _write_scoring_yaml(candidate_scoring_yaml, total=9.0)

    finalization = finalize_live_eval_run(
        run_id="candidate-run",
        result_root=candidate_result_root,
        harness_root=candidate_harness_root,
        scoring_yaml=candidate_scoring_yaml,
        ledger_path=ledger_path,
        baseline_path=baseline_path,
        update_baseline_path=baseline_path,
        min_improvement=1.0,
        planner_mode="replay",
        write_next_experiment=True,
        generated_at_utc="2026-04-24T00:00:03Z",
    )

    next_experiment_path = finalization.next_experiment_path
    assert next_experiment_path is not None
    assert next_experiment_path == candidate_harness_root / "next_experiment.json"
    assert next_experiment_path.is_file()
    promotion = PromotionDecision.from_dict(read_json(candidate_harness_root / "promotion_report.json"))
    assert promotion.decision.value == "accepted"
    assert promotion.metric.baseline_value == pytest.approx(7.5)
    reward_report = read_json(candidate_harness_root / "reward_report.json")
    reward_terms = {term["name"] for term in reward_report["terms"]}
    assert "promotion.metric.improvement" in reward_terms
    assert "promotion.metric.margin_gap" in reward_terms
    reward_artifact = next(
        artifact
        for artifact in read_json(candidate_harness_root / "run_manifest.json")["artifacts"]
        if artifact["kind"] == "reward_report"
    )
    assert {
        source["kind"]
        for source in reward_artifact["provenance"]["source_artifacts"]
    } >= {"scoring_yaml", "promotion_decision_snapshot"}
    assert len(read_ledger_entries(ledger_path)) == 2


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
