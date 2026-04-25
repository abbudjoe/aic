from __future__ import annotations

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
    PolicyTrainingRunReport,
    PolicyTrainingExecutionReport,
    PolicyTrainingExecutionStatus,
    SimulatorKind,
    TrainerInvocation,
    TrainingSignal,
    TrainingSignalKind,
    TrainingSignalReport,
    TrainingSourceKind,
    derive_training_dataset_report,
    read_json,
    run_policy_training_command,
    sha256_file,
    write_json,
)


def test_policy_training_runner_writes_checkpoint_logs_and_report(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    checkpoint_path = tmp_path / "policy.ckpt"
    stdout_path = tmp_path / "trainer.stdout.log"
    stderr_path = tmp_path / "trainer.stderr.log"
    report_path = tmp_path / "policy_training_report.json"
    execution_report_path = tmp_path / "policy_training_execution_report.json"

    result = run_policy_training_command(
        run_id="train-runner-a",
        generated_at_utc="2026-04-24T01:00:00Z",
        source_training_dataset_report=source_dataset_artifact,
        trainer=_trainer(_write_checkpoint_script()),
        candidate_policy_template=_candidate_policy_template(),
        policy_checkpoint_path=checkpoint_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        execution_report_path=execution_report_path,
        report_path=report_path,
        timeout_seconds=5.0,
        metrics={"train.loss": 0.05},
        notes=("pytest controlled runner",),
    )

    assert checkpoint_path.exists()
    assert stdout_path.read_text(encoding="utf-8").strip() == "checkpoint ready"
    assert stderr_path.read_text(encoding="utf-8") == ""
    assert result.report.run_id == "train-runner-a"
    assert result.report.policy_artifact.path == str(checkpoint_path.resolve(strict=False))
    assert result.report.policy_artifact.sha256 == sha256_file(checkpoint_path)
    assert result.report.candidate_policy.runtime_boundary is not None
    assert result.report.candidate_policy.runtime_boundary.policy_artifact == result.report.policy_artifact
    assert result.artifact_environment[AIC_POLICY_TRAINING_OUTPUT_CHECKPOINT] == str(
        checkpoint_path.resolve(strict=False)
    )
    assert result.artifact_environment[AIC_POLICY_TRAINING_SOURCE_DATASET_REPORT] == str(
        Path(source_dataset_artifact.path or "").resolve(strict=True)
    )
    assert PolicyTrainingRunReport.from_dict(read_json(report_path)) == result.report
    assert PolicyTrainingExecutionReport.from_dict(read_json(execution_report_path)) == result.execution_report
    assert result.execution_report.status is PolicyTrainingExecutionStatus.completed
    assert result.report.trainer.working_directory == str(Path.cwd().resolve(strict=True))
    assert result.execution_report.working_directory == result.report.trainer.working_directory
    assert result.execution_report_artifact.sha256 == sha256_file(execution_report_path)
    assert result.report.policy_training_execution_report == result.execution_report_artifact
    assert (
        result.report.policy_artifact.provenance["policy_training_execution_report_sha256"]
        == result.execution_report_artifact.sha256
    )
    missing_execution_artifact = result.report.to_dict()
    del missing_execution_artifact["policy_training_execution_report"]
    with pytest.raises(HarnessIOError, match="policy_training_execution_report must be set"):
        PolicyTrainingRunReport.from_dict(missing_execution_artifact)

    contradictory_execution_payload = result.execution_report.to_dict()
    contradictory_execution_payload["status"] = "failed"
    contradictory_execution_payload["ok"] = False
    contradictory_execution_payload["errors"] = ["synthetic failure after success"]
    contradictory_execution_payload["returncode"] = 9
    contradictory_execution_path = tmp_path / "contradictory_policy_training_execution_report.json"
    write_json(contradictory_execution_path, contradictory_execution_payload)
    contradictory_sha = sha256_file(contradictory_execution_path)
    contradictory_report_payload = result.report.to_dict()
    contradictory_report_payload["policy_training_execution_report"]["path"] = str(
        contradictory_execution_path
    )
    contradictory_report_payload["policy_training_execution_report"]["sha256"] = contradictory_sha
    contradictory_report_payload["policy_artifact"]["provenance"][
        "policy_training_execution_report_sha256"
    ] = contradictory_sha
    contradictory_report_payload["candidate_policy"]["provenance"][
        "policy_training_execution_report_sha256"
    ] = contradictory_sha
    with pytest.raises(HarnessIOError, match="status must be completed"):
        PolicyTrainingRunReport.from_dict(contradictory_report_payload)

    contradictory_status_payload = result.execution_report.to_dict()
    contradictory_status_payload["ok"] = False
    contradictory_status_payload["errors"] = ["contradictory completed status"]
    with pytest.raises(HarnessIOError, match="completed execution reports must set ok=true"):
        PolicyTrainingExecutionReport.from_dict(contradictory_status_payload)

    contradictory_leakage_payload = result.report.to_dict()
    contradictory_leakage_payload["policy_artifact"]["provenance"]["dataset_leakage_summary"] = {}
    with pytest.raises(HarnessIOError, match="dataset_leakage_summary"):
        PolicyTrainingRunReport.from_dict(contradictory_leakage_payload)

    contradictory_env_payload = result.report.to_dict()
    contradictory_execution_env_payload = result.execution_report.to_dict()
    contradictory_execution_env_payload["artifact_environment"][
        AIC_POLICY_TRAINING_SOURCE_DATASET_REPORT
    ] = str(tmp_path / "other_training_dataset_report.json")
    contradictory_execution_env_path = tmp_path / "contradictory_env_execution_report.json"
    write_json(contradictory_execution_env_path, contradictory_execution_env_payload)
    contradictory_env_sha = sha256_file(contradictory_execution_env_path)
    contradictory_env_payload["policy_training_execution_report"]["path"] = str(
        contradictory_execution_env_path
    )
    contradictory_env_payload["policy_training_execution_report"]["sha256"] = contradictory_env_sha
    contradictory_env_payload["policy_artifact"]["provenance"][
        "policy_training_execution_report_sha256"
    ] = contradictory_env_sha
    contradictory_env_payload["candidate_policy"]["provenance"][
        "policy_training_execution_report_sha256"
    ] = contradictory_env_sha
    with pytest.raises(HarnessIOError, match="source dataset path must match"):
        PolicyTrainingRunReport.from_dict(contradictory_env_payload)


def test_policy_training_runner_does_not_emit_report_when_command_fails(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    report_path = tmp_path / "policy_training_report.json"
    execution_report_path = tmp_path / "policy_training_execution_report.json"

    with pytest.raises(HarnessIOError, match="exited with code 7"):
        run_policy_training_command(
            run_id="train-runner-failed",
            generated_at_utc="2026-04-24T01:00:00Z",
            source_training_dataset_report=source_dataset_artifact,
            trainer=_trainer("import sys; print('bad'); print('boom', file=sys.stderr); sys.exit(7)"),
            candidate_policy_template=_candidate_policy_template(),
            policy_checkpoint_path=tmp_path / "policy.ckpt",
            stdout_path=tmp_path / "trainer.stdout.log",
            stderr_path=tmp_path / "trainer.stderr.log",
            execution_report_path=execution_report_path,
            report_path=report_path,
            timeout_seconds=5.0,
        )

    assert not report_path.exists()
    assert not (tmp_path / "policy.ckpt").exists()
    assert PolicyTrainingExecutionReport.from_dict(read_json(execution_report_path)).status is PolicyTrainingExecutionStatus.failed
    assert (tmp_path / "trainer.stdout.log").read_text(encoding="utf-8").strip() == "bad"
    assert (tmp_path / "trainer.stderr.log").read_text(encoding="utf-8").strip() == "boom"


def test_policy_training_runner_rejects_missing_checkpoint_after_success(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    report_path = tmp_path / "policy_training_report.json"
    execution_report_path = tmp_path / "policy_training_execution_report.json"

    with pytest.raises(HarnessIOError, match="policy checkpoint path must exist"):
        run_policy_training_command(
            run_id="train-runner-missing-checkpoint",
            generated_at_utc="2026-04-24T01:00:00Z",
            source_training_dataset_report=source_dataset_artifact,
            trainer=_trainer("print('no checkpoint')"),
            candidate_policy_template=_candidate_policy_template(),
            policy_checkpoint_path=tmp_path / "policy.ckpt",
            stdout_path=tmp_path / "trainer.stdout.log",
            stderr_path=tmp_path / "trainer.stderr.log",
            execution_report_path=execution_report_path,
            report_path=report_path,
            timeout_seconds=5.0,
        )

    assert not report_path.exists()
    assert (
        PolicyTrainingExecutionReport.from_dict(read_json(execution_report_path)).status
        is PolicyTrainingExecutionStatus.checkpoint_missing
    )
    assert (tmp_path / "trainer.stdout.log").read_text(encoding="utf-8").strip() == "no checkpoint"


def test_policy_training_runner_times_out_without_completed_report(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    report_path = tmp_path / "policy_training_report.json"
    execution_report_path = tmp_path / "policy_training_execution_report.json"

    with pytest.raises(HarnessIOError, match="timed out"):
        run_policy_training_command(
            run_id="train-runner-timeout",
            generated_at_utc="2026-04-24T01:00:00Z",
            source_training_dataset_report=source_dataset_artifact,
            trainer=_trainer("import time; time.sleep(5)"),
            candidate_policy_template=_candidate_policy_template(),
            policy_checkpoint_path=tmp_path / "policy.ckpt",
            stdout_path=tmp_path / "trainer.stdout.log",
            stderr_path=tmp_path / "trainer.stderr.log",
            execution_report_path=execution_report_path,
            report_path=report_path,
            timeout_seconds=0.1,
        )

    assert not report_path.exists()
    assert not (tmp_path / "policy.ckpt").exists()
    assert PolicyTrainingExecutionReport.from_dict(read_json(execution_report_path)).status is PolicyTrainingExecutionStatus.timed_out


def test_policy_training_runner_rejects_ambiguous_outputs_before_launch(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    shared_output_path = tmp_path / "shared.out"

    with pytest.raises(HarnessIOError, match="output paths must be distinct"):
        run_policy_training_command(
            run_id="train-runner-ambiguous",
            generated_at_utc="2026-04-24T01:00:00Z",
            source_training_dataset_report=source_dataset_artifact,
            trainer=_trainer(_write_checkpoint_script()),
            candidate_policy_template=_candidate_policy_template(),
            policy_checkpoint_path=tmp_path / "policy.ckpt",
            stdout_path=shared_output_path,
            stderr_path=shared_output_path,
            execution_report_path=tmp_path / "policy_training_execution_report.json",
            timeout_seconds=5.0,
        )

    assert not shared_output_path.exists()
    assert not (tmp_path / "policy.ckpt").exists()


def test_policy_training_runner_rejects_hidden_candidate_policy_artifacts() -> None:
    with pytest.raises(HarnessIOError, match="policy artifact keys"):
        CandidatePolicyTemplate(
            backend_kind=BackendKind.lewm_world_model,
            name="lewm-runner-candidate",
            training_sources=(TrainingSourceKind.official_demo,),
            simulator_sources=(SimulatorKind.offline_replay,),
            legal_observation_contract="official aic_model observations only",
            config={"policy_path": "/tmp/policy.ckpt"},
        )

    with pytest.raises(HarnessIOError, match="must not override critical fields"):
        CandidatePolicyTemplate(
            backend_kind=BackendKind.lewm_world_model,
            name="lewm-runner-candidate",
            training_sources=(TrainingSourceKind.official_demo,),
            simulator_sources=(SimulatorKind.offline_replay,),
            legal_observation_contract="official aic_model observations only",
            provenance={"run_id": "forged"},
        )

    with pytest.raises(HarnessIOError, match="model_annotation"):
        CandidatePolicyTemplate(
            backend_kind=BackendKind.lewm_world_model,
            name="lewm-runner-candidate",
            training_sources=(TrainingSourceKind.model_annotation,),
            simulator_sources=(SimulatorKind.offline_replay,),
            legal_observation_contract="official aic_model observations only",
        )


def test_policy_training_runner_rejects_hidden_command_artifacts_and_runtime_launches(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)

    with pytest.raises(HarnessIOError, match="must not hide artifact references"):
        run_policy_training_command(
            run_id="train-runner-hidden-command",
            generated_at_utc="2026-04-24T01:00:00Z",
            source_training_dataset_report=source_dataset_artifact,
            trainer=TrainerInvocation(
                trainer_id="bad-command-trainer",
                trainer_name="Bad command trainer",
                backend_kind=BackendKind.lewm_world_model,
                command=(sys.executable, "/tmp/train.py", "dataset=/tmp/data.h5"),
                offline_only=True,
                runtime_allowed=False,
                uses_online_language_model_control=False,
            ),
            candidate_policy_template=_candidate_policy_template(),
            policy_checkpoint_path=tmp_path / "policy.ckpt",
            stdout_path=tmp_path / "trainer.stdout.log",
            stderr_path=tmp_path / "trainer.stderr.log",
            execution_report_path=tmp_path / "policy_training_execution_report.json",
            timeout_seconds=5.0,
        )

    with pytest.raises(HarnessIOError, match="must not hide artifact references"):
        run_policy_training_command(
            run_id="train-runner-hidden-option",
            generated_at_utc="2026-04-24T01:00:00Z",
            source_training_dataset_report=source_dataset_artifact,
            trainer=TrainerInvocation(
                trainer_id="bad-option-trainer",
                trainer_name="Bad option trainer",
                backend_kind=BackendKind.lewm_world_model,
                command=(sys.executable, "-c", "print('unused')", "--dataset=/tmp/privileged.h5"),
                offline_only=True,
                runtime_allowed=False,
                uses_online_language_model_control=False,
            ),
            candidate_policy_template=_candidate_policy_template(),
            policy_checkpoint_path=tmp_path / "policy.ckpt",
            stdout_path=tmp_path / "trainer0.stdout.log",
            stderr_path=tmp_path / "trainer0.stderr.log",
            execution_report_path=tmp_path / "policy_training_execution_report0.json",
            timeout_seconds=5.0,
        )

    with pytest.raises(HarnessIOError, match="must not launch runtime/evaluation surfaces"):
        run_policy_training_command(
            run_id="train-runner-runtime-launch",
            generated_at_utc="2026-04-24T01:00:00Z",
            source_training_dataset_report=source_dataset_artifact,
            trainer=TrainerInvocation(
                trainer_id="bad-runtime-trainer",
                trainer_name="Bad runtime trainer",
                backend_kind=BackendKind.lewm_world_model,
                command=(sys.executable, "-c", "print('launch aic_controller')"),
                offline_only=True,
                runtime_allowed=False,
                uses_online_language_model_control=False,
            ),
            candidate_policy_template=_candidate_policy_template(),
            policy_checkpoint_path=tmp_path / "policy.ckpt",
            stdout_path=tmp_path / "trainer2.stdout.log",
            stderr_path=tmp_path / "trainer2.stderr.log",
            execution_report_path=tmp_path / "policy_training_execution_report2.json",
            timeout_seconds=5.0,
        )


def test_policy_training_runner_python_guard_blocks_child_processes(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    execution_report_path = tmp_path / "policy_training_execution_report.json"

    with pytest.raises(HarnessIOError, match="offline guard blocked audit event"):
        run_policy_training_command(
            run_id="train-runner-child-process",
            generated_at_utc="2026-04-24T01:00:00Z",
            source_training_dataset_report=source_dataset_artifact,
            trainer=_trainer(
                "import subprocess; "
                "subprocess.run(['echo', 'not offline'], check=True)"
            ),
            candidate_policy_template=_candidate_policy_template(),
            policy_checkpoint_path=tmp_path / "policy.ckpt",
            stdout_path=tmp_path / "trainer.stdout.log",
            stderr_path=tmp_path / "trainer.stderr.log",
            execution_report_path=execution_report_path,
            timeout_seconds=5.0,
        )

    execution_report = PolicyTrainingExecutionReport.from_dict(read_json(execution_report_path))
    assert execution_report.status is PolicyTrainingExecutionStatus.guard_violation
    assert "blocked audit event" in (tmp_path / "trainer.stderr.log").read_text(encoding="utf-8")
    assert not (tmp_path / "policy.ckpt").exists()


def test_policy_training_runner_rejects_swallowed_python_guard_violation(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    execution_report_path = tmp_path / "policy_training_execution_report.json"
    report_path = tmp_path / "policy_training_report.json"

    with pytest.raises(HarnessIOError, match="offline guard blocked audit event"):
        run_policy_training_command(
            run_id="train-runner-swallowed-child-process",
            generated_at_utc="2026-04-24T01:00:00Z",
            source_training_dataset_report=source_dataset_artifact,
            trainer=_trainer(
                "import os, pathlib, subprocess\n"
                "checkpoint = pathlib.Path(os.environ['AIC_POLICY_TRAINING_OUTPUT_CHECKPOINT'])\n"
                "try:\n"
                "    subprocess.run(['echo', 'blocked'], check=True)\n"
                "except RuntimeError:\n"
                "    pass\n"
                "checkpoint.write_bytes(b'should not complete')"
            ),
            candidate_policy_template=_candidate_policy_template(),
            policy_checkpoint_path=tmp_path / "policy.ckpt",
            stdout_path=tmp_path / "trainer.stdout.log",
            stderr_path=tmp_path / "trainer.stderr.log",
            execution_report_path=execution_report_path,
            report_path=report_path,
            timeout_seconds=5.0,
        )

    execution_report = PolicyTrainingExecutionReport.from_dict(read_json(execution_report_path))
    assert execution_report.status is PolicyTrainingExecutionStatus.guard_violation
    assert not (tmp_path / "policy.ckpt").exists()
    assert not report_path.exists()


def test_policy_training_runner_guard_exit_survives_closed_stderr(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    execution_report_path = tmp_path / "policy_training_execution_report.json"

    with pytest.raises(HarnessIOError, match="offline guard blocked audit event"):
        run_policy_training_command(
            run_id="train-runner-closed-stderr-guard",
            generated_at_utc="2026-04-24T01:00:00Z",
            source_training_dataset_report=source_dataset_artifact,
            trainer=_trainer(
                "import os, pathlib, subprocess\n"
                "checkpoint = pathlib.Path(os.environ['AIC_POLICY_TRAINING_OUTPUT_CHECKPOINT'])\n"
                "os.close(2)\n"
                "try:\n"
                "    subprocess.run(['echo', 'blocked'], check=True)\n"
                "except RuntimeError:\n"
                "    checkpoint.write_bytes(b'should not complete')"
            ),
            candidate_policy_template=_candidate_policy_template(),
            policy_checkpoint_path=tmp_path / "policy.ckpt",
            stdout_path=tmp_path / "trainer.stdout.log",
            stderr_path=tmp_path / "trainer.stderr.log",
            execution_report_path=execution_report_path,
            timeout_seconds=5.0,
        )

    execution_report = PolicyTrainingExecutionReport.from_dict(read_json(execution_report_path))
    assert execution_report.status is PolicyTrainingExecutionStatus.guard_violation
    assert not (tmp_path / "policy.ckpt").exists()


def test_policy_training_runner_guard_exit_survives_guard_env_tampering(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    execution_report_path = tmp_path / "policy_training_execution_report.json"
    report_path = tmp_path / "policy_training_report.json"
    checkpoint_path = tmp_path / "policy.ckpt"

    with pytest.raises(HarnessIOError, match="offline guard blocked audit event"):
        run_policy_training_command(
            run_id="train-runner-guard-env-tamper",
            generated_at_utc="2026-04-24T01:00:00Z",
            source_training_dataset_report=source_dataset_artifact,
            trainer=_trainer(
                "import os\n"
                "checkpoint_fd = os.open(\n"
                "    os.environ['AIC_POLICY_TRAINING_OUTPUT_CHECKPOINT'],\n"
                "    os.O_WRONLY | os.O_CREAT | os.O_TRUNC,\n"
                "    0o644,\n"
                ")\n"
                "os.environ['AIC_POLICY_TRAINING_OFFLINE_GUARD'] = 'disabled'\n"
                "os.write(checkpoint_fd, b'should not complete')\n"
                "os.close(checkpoint_fd)"
            ),
            candidate_policy_template=_candidate_policy_template(),
            policy_checkpoint_path=checkpoint_path,
            stdout_path=tmp_path / "trainer.stdout.log",
            stderr_path=tmp_path / "trainer.stderr.log",
            execution_report_path=execution_report_path,
            report_path=report_path,
            timeout_seconds=5.0,
        )

    execution_report = PolicyTrainingExecutionReport.from_dict(read_json(execution_report_path))
    assert execution_report.status is PolicyTrainingExecutionStatus.guard_violation
    if checkpoint_path.exists():
        assert checkpoint_path.read_bytes() != b"should not complete"
    assert not report_path.exists()


def test_policy_training_runner_rejects_fake_python_executable(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(tmp_path)
    fake_python = tmp_path / "python-trainer"
    fake_python.write_text("#!/bin/sh\necho not-python\n", encoding="utf-8")
    fake_python.chmod(0o755)

    with pytest.raises(HarnessIOError, match="this harness Python interpreter"):
        run_policy_training_command(
            run_id="train-runner-fake-python",
            generated_at_utc="2026-04-24T01:00:00Z",
            source_training_dataset_report=source_dataset_artifact,
            trainer=TrainerInvocation(
                trainer_id="fake-python-trainer",
                trainer_name="Fake Python trainer",
                backend_kind=BackendKind.lewm_world_model,
                command=(str(fake_python), "-c", "print('unused')"),
                offline_only=True,
                runtime_allowed=False,
                uses_online_language_model_control=False,
            ),
            candidate_policy_template=_candidate_policy_template(),
            policy_checkpoint_path=tmp_path / "policy.ckpt",
            stdout_path=tmp_path / "trainer.stdout.log",
            stderr_path=tmp_path / "trainer.stderr.log",
            execution_report_path=tmp_path / "policy_training_execution_report.json",
            timeout_seconds=5.0,
        )


def test_policy_training_runner_rejects_disallowed_dataset_leakage(tmp_path: Path) -> None:
    source_dataset_artifact = _training_dataset_artifact(
        tmp_path,
        leakage_class=LeakageClass.privileged_eval_signal,
    )

    with pytest.raises(HarnessIOError, match="leakage classes not allowed"):
        run_policy_training_command(
            run_id="train-runner-bad-leakage",
            generated_at_utc="2026-04-24T01:00:00Z",
            source_training_dataset_report=source_dataset_artifact,
            trainer=_trainer(_write_checkpoint_script()),
            candidate_policy_template=_candidate_policy_template(),
            policy_checkpoint_path=tmp_path / "policy.ckpt",
            stdout_path=tmp_path / "trainer.stdout.log",
            stderr_path=tmp_path / "trainer.stderr.log",
            execution_report_path=tmp_path / "policy_training_execution_report.json",
            timeout_seconds=5.0,
            allowed_training_leakage_classes=(LeakageClass.privileged_training_signal,),
        )


def _trainer(script: str) -> TrainerInvocation:
    return TrainerInvocation(
        trainer_id="lewm-runner-trainer-v1",
        trainer_name="LEWM controlled runner trainer",
        backend_kind=BackendKind.lewm_world_model,
        command=(sys.executable, "-c", script),
        config={
            "learning_rate": 0.0001,
            "max_epochs": 1,
        },
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
        config={"planner_mode": "learned_policy"},
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


def _training_dataset_artifact(
    tmp_path: Path,
    *,
    leakage_class: LeakageClass = LeakageClass.legal_policy_action_output,
) -> ArtifactRef:
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
                leakage_class=leakage_class,
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
