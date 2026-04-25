from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from aic_signal_harness import (
    AIC_POLICY_TRAINING_OUTPUT_CHECKPOINT,
    AIC_POLICY_TRAINING_SOURCE_DATASET_REPORT,
    ArtifactRef,
    BackendKind,
    CandidatePolicyTemplate,
    HarnessIOError,
    LeakageClass,
    PromotionDecision,
    PolicyTrainingCommandResult,
    PolicyTrainingRunReport,
    RuntimeRole,
    SimulatorKind,
    TrainerInvocation,
    TrainingSignal,
    TrainingSignalKind,
    TrainingSignalReport,
    TrainingSourceKind,
    derive_training_dataset_report,
    read_json,
    read_ledger_entries,
    run_policy_training_command,
    sha256_file,
    write_json,
)
from aic_signal_harness.train_eval_promote import (
    TrainedPolicyEvalBindingReport,
    finalize_trained_policy_eval,
    stage_policy_training_report_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_train_eval_promote_module_executes_with_warnings_as_errors() -> None:
    result = subprocess.run(
        [sys.executable, "-W", "error", "-m", "aic_signal_harness.train_eval_promote", "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "trained policy eval" in result.stdout


def test_stage_policy_training_report_bundle_rewrites_byte_bound_paths(
    tmp_path: Path,
) -> None:
    training = _run_training(tmp_path)
    bundle_root = tmp_path / "remote_bundle"

    staged = stage_policy_training_report_bundle(
        policy_training_report=_required_report_path(training),
        bundle_root=bundle_root,
        runtime_bundle_root=bundle_root,
    )

    staged_payload = read_json(staged.policy_training_report_path)
    staged_report = PolicyTrainingRunReport.from_dict(staged_payload)
    resolved_bundle = bundle_root.resolve(strict=False)
    assert staged_report.policy_artifact.path == str(staged.policy_checkpoint_path)
    assert staged_report.policy_artifact.sha256 == training.report.policy_artifact.sha256
    assert staged_report.source_training_dataset_report.path == str(
        resolved_bundle / "training_dataset_report.json"
    )
    assert staged_report.policy_training_execution_report is not None
    assert staged_report.policy_training_execution_report.path == str(
        resolved_bundle / "policy_training_execution_report.json"
    )
    assert staged_report.policy_artifact.provenance[
        "source_training_dataset_report_sha256"
    ] == staged_report.source_training_dataset_report.sha256
    assert staged_report.policy_artifact.provenance[
        "policy_training_execution_report_sha256"
    ] == staged_report.policy_training_execution_report.sha256
    assert sha256_file(staged.policy_training_report_path) != sha256_file(
        _required_report_path(training)
    )


def test_stage_policy_training_report_bundle_rejects_malformed_report_typed(
    tmp_path: Path,
) -> None:
    malformed_report = tmp_path / "policy_training_report.json"
    write_json(malformed_report, {"schema_version": 1})

    with pytest.raises(HarnessIOError, match="policy training report"):
        stage_policy_training_report_bundle(
            policy_training_report=malformed_report,
            bundle_root=tmp_path / "bundle",
        )


def test_stage_policy_training_report_bundle_rejects_missing_report_typed(
    tmp_path: Path,
) -> None:
    with pytest.raises(HarnessIOError, match="policy_training_report does not exist"):
        stage_policy_training_report_bundle(
            policy_training_report=tmp_path / "missing_policy_training_report.json",
            bundle_root=tmp_path / "bundle",
        )


def test_finalize_trained_policy_eval_binds_training_eval_and_promotion(
    tmp_path: Path,
) -> None:
    training = _run_training(tmp_path)
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    ledger_path = harness_root / "ledger.jsonl"
    baseline_path = harness_root / "baseline.json"
    runtime_checkpoint = tmp_path / "runtime" / "aic_lewm_epoch_100_object.ckpt"
    runtime_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    runtime_checkpoint.write_bytes(Path(training.report.policy_artifact.path or "").read_bytes())
    _write_scoring_yaml(scoring_yaml, total=11.25)

    finalization = finalize_trained_policy_eval(
        run_id="eval-trained-a",
        result_root=result_root,
        harness_root=harness_root,
        scoring_yaml=scoring_yaml,
        policy_training_report=_required_report_path(training),
        runtime_policy_checkpoint=runtime_checkpoint,
        ledger_path=ledger_path,
        update_baseline_path=baseline_path,
        bootstrap_promotion=True,
        model_image="aic-lewm-learned:eval-trained-a",
        model_image_id="sha256:" + "a" * 64,
        backend_kind=BackendKind.lewm_world_model,
        runtime_env={"AIC_LEWM_PLANNER_MODE": "lewm_mpc"},
        generated_at_utc="2026-04-24T02:00:00Z",
    )

    assert finalization.binding_path.is_file()
    assert finalization.summary_path.is_file()
    assert finalization.binding_report.train_run_id == training.report.run_id
    assert finalization.binding_report.eval_run_id == "eval-trained-a"
    assert finalization.binding_report.runtime_policy_artifact.path == str(
        runtime_checkpoint.resolve(strict=False)
    )
    assert finalization.binding_report.runtime_policy_artifact.sha256 == (
        training.report.policy_artifact.sha256
    )
    assert finalization.binding_report.runtime_image_artifact.kind == "docker_image"
    assert finalization.binding_report.runtime_image_artifact.sha256 == "a" * 64
    assert finalization.live_eval.manifest.backend.runtime_role is RuntimeRole.live_policy
    assert finalization.live_eval.manifest.backend.runtime_boundary is not None
    assert finalization.live_eval.manifest.backend.runtime_boundary.policy_artifact == (
        finalization.binding_report.runtime_policy_artifact
    )
    assert (
        finalization.live_eval.manifest.backend.provenance["source_policy_training_report_sha256"]
        == finalization.binding_artifact.provenance["source_policy_training_report_sha256"]
    )

    manifest = read_json(harness_root / "run_manifest.json")
    artifacts_by_kind = {artifact["kind"]: artifact for artifact in manifest["artifacts"]}
    assert {
        "policy_training_report",
        "policy_training_execution_report",
        "policy_checkpoint",
        "trained_policy_eval_binding",
        "scoring_yaml",
        "reward_report",
        "failure_report",
    } <= set(artifacts_by_kind)
    assert artifacts_by_kind["trained_policy_eval_binding"]["sha256"] == sha256_file(
        finalization.binding_path
    )
    assert artifacts_by_kind["policy_checkpoint"]["sha256"] == training.report.policy_artifact.sha256
    assert artifacts_by_kind["docker_image"]["sha256"] == "a" * 64

    ledger_entries = read_ledger_entries(ledger_path)
    assert len(ledger_entries) == 1
    assert ledger_entries[0].run_id == "eval-trained-a"
    assert ledger_entries[0].manifest.sha256 == finalization.live_eval.manifest_artifact.sha256
    promotion = PromotionDecision.from_dict(read_json(harness_root / "promotion_report.json"))
    assert promotion.manifest.sha256 == finalization.live_eval.manifest_artifact.sha256
    assert PromotionDecision.from_dict(read_json(baseline_path)).run_id == "eval-trained-a"
    summary = read_json(finalization.summary_path)
    assert summary["binding_report"]["sha256"] == finalization.binding_artifact.sha256
    assert summary["manifest"]["sha256"] == finalization.live_eval.manifest_artifact.sha256
    assert summary["score_total"] == 11.25


def test_finalize_trained_policy_eval_rejects_runtime_checkpoint_mismatch_without_outputs(
    tmp_path: Path,
) -> None:
    training = _run_training(tmp_path)
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    runtime_checkpoint = tmp_path / "runtime" / "aic_lewm_epoch_100_object.ckpt"
    runtime_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    runtime_checkpoint.write_bytes(b"wrong checkpoint bytes")
    _write_scoring_yaml(scoring_yaml)

    with pytest.raises(HarnessIOError, match="sha256 does not match trained policy artifact"):
        finalize_trained_policy_eval(
            run_id="eval-trained-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            policy_training_report=_required_report_path(training),
            runtime_policy_checkpoint=runtime_checkpoint,
            ledger_path=harness_root / "ledger.jsonl",
            bootstrap_promotion=True,
            model_image_id="sha256:" + "a" * 64,
            generated_at_utc="2026-04-24T02:00:00Z",
        )

    assert not (harness_root / "trained_policy_eval_binding.json").exists()
    assert not (harness_root / "ledger_entry.json").exists()
    assert not (harness_root / "promotion_report.json").exists()


def test_finalize_trained_policy_eval_rejects_missing_model_image_id_without_outputs(
    tmp_path: Path,
) -> None:
    training = _run_training(tmp_path)
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    runtime_checkpoint = tmp_path / "runtime" / "aic_lewm_epoch_100_object.ckpt"
    runtime_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    runtime_checkpoint.write_bytes(Path(training.report.policy_artifact.path or "").read_bytes())
    _write_scoring_yaml(scoring_yaml)

    with pytest.raises(HarnessIOError, match="model_image_id"):
        finalize_trained_policy_eval(
            run_id="eval-trained-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            policy_training_report=_required_report_path(training),
            runtime_policy_checkpoint=runtime_checkpoint,
            ledger_path=harness_root / "ledger.jsonl",
            bootstrap_promotion=True,
            generated_at_utc="2026-04-24T02:00:00Z",
        )

    assert not (harness_root / "trained_policy_eval_binding.json").exists()
    assert not (harness_root / "ledger_entry.json").exists()


def test_finalize_trained_policy_eval_preflights_live_outputs_before_binding(
    tmp_path: Path,
) -> None:
    training = _run_training(tmp_path)
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    runtime_checkpoint = tmp_path / "runtime" / "aic_lewm_epoch_100_object.ckpt"
    runtime_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    runtime_checkpoint.write_bytes(Path(training.report.policy_artifact.path or "").read_bytes())
    _write_scoring_yaml(scoring_yaml)
    harness_root.mkdir(parents=True)
    (harness_root / "reward_report.json").write_text("occupied\n", encoding="utf-8")

    with pytest.raises(HarnessIOError, match="output already exists"):
        finalize_trained_policy_eval(
            run_id="eval-trained-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            policy_training_report=_required_report_path(training),
            runtime_policy_checkpoint=runtime_checkpoint,
            ledger_path=harness_root / "ledger.jsonl",
            bootstrap_promotion=True,
            model_image="aic-lewm-learned:eval-trained-a",
            model_image_id="sha256:" + "a" * 64,
            generated_at_utc="2026-04-24T02:00:00Z",
        )

    assert not (harness_root / "trained_policy_eval_binding.json").exists()
    assert not (harness_root / "ledger_entry.json").exists()


def test_trained_policy_eval_binding_rejects_forged_eval_backend(tmp_path: Path) -> None:
    training = _run_training(tmp_path)
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    runtime_checkpoint = tmp_path / "runtime" / "aic_lewm_epoch_100_object.ckpt"
    runtime_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    runtime_checkpoint.write_bytes(Path(training.report.policy_artifact.path or "").read_bytes())
    _write_scoring_yaml(scoring_yaml)
    finalization = finalize_trained_policy_eval(
        run_id="eval-trained-a",
        result_root=result_root,
        harness_root=harness_root,
        scoring_yaml=scoring_yaml,
        policy_training_report=_required_report_path(training),
        runtime_policy_checkpoint=runtime_checkpoint,
        ledger_path=harness_root / "ledger.jsonl",
        bootstrap_promotion=True,
        model_image="aic-lewm-learned:eval-trained-a",
        model_image_id="sha256:" + "a" * 64,
        generated_at_utc="2026-04-24T02:00:00Z",
    )
    forged = finalization.binding_report.to_dict()
    forged["eval_backend"]["runtime_boundary"]["policy_artifact"]["sha256"] = "b" * 64

    with pytest.raises(HarnessIOError, match="runtime_boundary.policy_artifact"):
        TrainedPolicyEvalBindingReport.from_dict(forged)


def test_finalize_trained_policy_eval_rejects_runtime_planner_mode_mismatch(
    tmp_path: Path,
) -> None:
    training = _run_training(tmp_path)
    payload = training.report.to_dict()
    payload["candidate_policy"]["config"]["planner_mode"] = "replay"
    report_path = tmp_path / "planner_mismatch_policy_training_report.json"
    write_json(report_path, payload)
    runtime_checkpoint = tmp_path / "runtime" / "aic_lewm_epoch_100_object.ckpt"
    runtime_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    runtime_checkpoint.write_bytes(Path(training.report.policy_artifact.path or "").read_bytes())
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    _write_scoring_yaml(scoring_yaml)

    with pytest.raises(HarnessIOError, match="planner_mode"):
        finalize_trained_policy_eval(
            run_id="eval-trained-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            policy_training_report=report_path,
            runtime_policy_checkpoint=runtime_checkpoint,
            ledger_path=harness_root / "ledger.jsonl",
            bootstrap_promotion=True,
            model_image_id="sha256:" + "a" * 64,
            runtime_env={"AIC_LEWM_PLANNER_MODE": "lewm_mpc"},
            generated_at_utc="2026-04-24T02:00:00Z",
        )

    assert not (harness_root / "trained_policy_eval_binding.json").exists()
    assert not (harness_root / "ledger_entry.json").exists()


def test_finalize_trained_policy_eval_rejects_backend_kind_override_mismatch(
    tmp_path: Path,
) -> None:
    training = _run_training(tmp_path)
    runtime_checkpoint = tmp_path / "runtime" / "aic_lewm_epoch_100_object.ckpt"
    runtime_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    runtime_checkpoint.write_bytes(Path(training.report.policy_artifact.path or "").read_bytes())
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    _write_scoring_yaml(scoring_yaml)

    with pytest.raises(HarnessIOError, match="backend_kind override"):
        finalize_trained_policy_eval(
            run_id="eval-trained-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            policy_training_report=_required_report_path(training),
            runtime_policy_checkpoint=runtime_checkpoint,
            ledger_path=harness_root / "ledger.jsonl",
            bootstrap_promotion=True,
            model_image_id="sha256:" + "a" * 64,
            backend_kind=BackendKind.replay_servo,
            generated_at_utc="2026-04-24T02:00:00Z",
        )

    assert not (harness_root / "trained_policy_eval_binding.json").exists()
    assert not (harness_root / "ledger_entry.json").exists()


def test_finalize_trained_policy_eval_rejects_uncontrolled_training_report_without_outputs(
    tmp_path: Path,
) -> None:
    training = _run_training(tmp_path)
    payload = training.report.to_dict()
    del payload["policy_training_execution_report"]
    payload["policy_artifact"]["provenance"].pop("policy_training_execution_report_sha256")
    payload["candidate_policy"]["provenance"].pop("policy_training_execution_report_sha256")
    payload["candidate_policy"]["runtime_boundary"]["policy_artifact"]["provenance"].pop(
        "policy_training_execution_report_sha256"
    )
    report_path = tmp_path / "uncontrolled_policy_training_report.json"
    write_json(report_path, payload)
    runtime_checkpoint = tmp_path / "runtime" / "aic_lewm_epoch_100_object.ckpt"
    runtime_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    runtime_checkpoint.write_bytes(Path(training.report.policy_artifact.path or "").read_bytes())
    result_root = tmp_path / "result"
    harness_root = result_root / "harness"
    scoring_yaml = result_root / "eval" / "scoring.yaml"
    _write_scoring_yaml(scoring_yaml)

    with pytest.raises(HarnessIOError, match="controlled policy_training_execution_report"):
        finalize_trained_policy_eval(
            run_id="eval-trained-a",
            result_root=result_root,
            harness_root=harness_root,
            scoring_yaml=scoring_yaml,
            policy_training_report=report_path,
            runtime_policy_checkpoint=runtime_checkpoint,
            ledger_path=harness_root / "ledger.jsonl",
            bootstrap_promotion=True,
            model_image_id="sha256:" + "a" * 64,
            generated_at_utc="2026-04-24T02:00:00Z",
        )

    assert not (harness_root / "trained_policy_eval_binding.json").exists()
    assert not (harness_root / "ledger_entry.json").exists()


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


def _required_report_path(training: PolicyTrainingCommandResult) -> str:
    assert training.report_path is not None
    return training.report_path


def _run_training(tmp_path: Path) -> PolicyTrainingCommandResult:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    output_dir = tmp_path / "training"
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_policy_training_command(
        run_id="train-runner-a",
        generated_at_utc="2026-04-24T01:00:00Z",
        source_training_dataset_report=source_dataset_artifact,
        trainer=_trainer(_write_checkpoint_script()),
        candidate_policy_template=_candidate_policy_template(),
        policy_checkpoint_path=output_dir / "policy.ckpt",
        stdout_path=output_dir / "trainer.stdout.log",
        stderr_path=output_dir / "trainer.stderr.log",
        execution_report_path=output_dir / "policy_training_execution_report.json",
        report_path=output_dir / "policy_training_report.json",
        timeout_seconds=5.0,
        metrics={"train.loss": 0.05},
    )


def _trainer(script: str) -> TrainerInvocation:
    return TrainerInvocation(
        trainer_id="lewm-runner-trainer-v1",
        trainer_name="LEWM controlled runner trainer",
        backend_kind=BackendKind.lewm_world_model,
        command=(sys.executable, "-c", script),
        config={"learning_rate": 0.0001, "max_epochs": 1},
        seed=11,
        offline_only=True,
        runtime_allowed=False,
        uses_online_language_model_control=False,
    )


def _candidate_policy_template() -> CandidatePolicyTemplate:
    return CandidatePolicyTemplate(
        backend_kind=BackendKind.lewm_world_model,
        name="lewm-runner-candidate",
        training_sources=(TrainingSourceKind.official_demo,),
        simulator_sources=(SimulatorKind.offline_replay,),
        legal_observation_contract="official aic_model observations only",
        description="Offline-trained LEWM policy candidate.",
        config={"planner_mode": "lewm_mpc"},
        provenance={"experiment_family": "pytest"},
        deterministic=True,
    )


def _write_checkpoint_script() -> str:
    return (
        "import os, pathlib; "
        f"source = pathlib.Path(os.environ[{AIC_POLICY_TRAINING_SOURCE_DATASET_REPORT!r}]); "
        f"checkpoint = pathlib.Path(os.environ[{AIC_POLICY_TRAINING_OUTPUT_CHECKPOINT!r}]); "
        "checkpoint.write_bytes(b'runner checkpoint:' + source.read_bytes()[:16]); "
        "print('checkpoint ready')"
    )


def _training_dataset_artifact(tmp_path: Path) -> ArtifactRef:
    signal_report = TrainingSignalReport(
        run_id="source-runner-a",
        generated_at_utc="2026-04-24T00:30:00Z",
        source_trace=ArtifactRef(
            kind="episode_trace",
            uri="memory://pytest/episode_trace.json",
            sha256="a" * 64,
            provenance={
                "producer": "pytest",
                "run_id": "source-runner-a",
                "derivation": "derive_episode_trace",
            },
        ),
        signals=(
            TrainingSignal(
                signal_id="tsig_runner_action_1",
                kind=TrainingSignalKind.behavior_clone_action,
                target="policy.action",
                weight=1.0,
                source="episode_trace",
                source_event_indices=(1,),
                extraction_method="episode_trace.v1.behavior_clone_action",
                leakage_class=LeakageClass.legal_policy_action_output,
                evidence={"payload": {"linear": [0.1, 0.0, 0.0]}},
                trial_id="trial-a",
            ),
        ),
        notes=("Signals are offline-only.",),
    )
    signal_path = tmp_path / "training_signal_report.json"
    write_json(signal_path, signal_report.to_dict())
    signal_artifact = ArtifactRef(
        kind="training_signal_report",
        path=str(signal_path),
        sha256=sha256_file(signal_path),
        provenance={
            "producer": "pytest",
            "run_id": signal_report.run_id,
            "derivation": "derive_training_signal_report",
        },
    )
    dataset_report = derive_training_dataset_report(
        signal_report,
        source_training_signal_report=signal_artifact,
    )
    dataset_path = tmp_path / "training_dataset_report.json"
    write_json(dataset_path, dataset_report.to_dict())
    return ArtifactRef(
        kind="training_dataset_report",
        path=str(dataset_path),
        sha256=sha256_file(dataset_path),
        provenance={
            "producer": "pytest",
            "run_id": dataset_report.run_id,
            "derivation": "derive_training_dataset_report",
        },
    )
