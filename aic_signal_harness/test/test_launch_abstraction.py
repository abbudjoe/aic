from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aic_signal_harness import (
    ArtifactRef,
    AutonomousGateConfig,
    AutonomousGateVerdict,
    BackendKind,
    CandidateEvalMode,
    CandidateEvalSpec,
    CandidatePolicyRecord,
    CandidatePolicyRegistry,
    CandidatePolicyTemplate,
    HarnessIOError,
    LEWM_TRAINER_ADAPTER_PLAN_KIND,
    LeakageClass,
    LewmTrainerAdapterConfig,
    OUTER_LOOP_LAUNCH_PLAN_KIND,
    OuterLoopLaunchConfig,
    OuterLoopLaunchOperation,
    OuterLoopLaunchPlan,
    OuterLoopLaunchTarget,
    SimulatorKind,
    TrainerInvocation,
    TrainingSignal,
    TrainingSignalKind,
    TrainingSignalReport,
    TrainingSourceKind,
    build_lewm_trainer_adapter_plan,
    build_live_eval_launch_plan_from_gate,
    build_training_launch_plan_from_lewm_adapter,
    derive_training_dataset_report,
    evaluate_autonomous_training_gate,
    materialize_training_dataset_jsonl,
    read_outer_loop_launch_plan_artifact,
    run_policy_training_command,
    sha256_file,
    write_autonomous_gate_decision,
    write_candidate_policy_registry,
    write_json,
    write_lewm_trainer_adapter_plan,
    write_outer_loop_launch_plan,
    write_training_dataset_materialization_report,
)


def test_training_launch_plan_binds_lewm_adapter_without_launch(tmp_path: Path) -> None:
    adapter_artifact = _lewm_adapter_artifact(tmp_path)

    plan = build_training_launch_plan_from_lewm_adapter(
        lewm_trainer_adapter_plan=adapter_artifact,
        output_root=tmp_path / "training",
        config=OuterLoopLaunchConfig(
            plan_id="launch-train-a",
            generated_at_utc="2026-04-25T05:00:00Z",
            launch_target=OuterLoopLaunchTarget.local,
        ),
    )
    artifact = write_outer_loop_launch_plan(tmp_path / "training_launch_plan.json", plan)
    reread = read_outer_loop_launch_plan_artifact(artifact)

    assert reread == plan
    assert artifact.kind == OUTER_LOOP_LAUNCH_PLAN_KIND
    assert plan.operation is OuterLoopLaunchOperation.policy_training
    assert plan.launch_target is OuterLoopLaunchTarget.local
    assert plan.requires_manual_authorization is True
    assert plan.autonomous_launch_allowed is False
    assert plan.source_artifacts["lewm_trainer_adapter_plan"] == adapter_artifact
    assert plan.source_artifacts["lewm_trainer_adapter_plan"].kind == LEWM_TRAINER_ADAPTER_PLAN_KIND
    assert plan.expected_outputs["policy_checkpoint"].endswith("/training/policy.ckpt")
    assert plan.environment["AIC_LEWM_OUTPUT_CHECKPOINT"] == plan.expected_outputs["policy_checkpoint"]
    assert plan.environment["AIC_TRAIN_RUN_ID"] == "train-lewm-a"


def test_training_launch_plan_rejects_target_mismatch_and_hidden_env(
    tmp_path: Path,
) -> None:
    adapter_artifact = _lewm_adapter_artifact(tmp_path)

    with pytest.raises(HarnessIOError, match="launch_target"):
        build_training_launch_plan_from_lewm_adapter(
            lewm_trainer_adapter_plan=adapter_artifact,
            output_root=tmp_path / "training",
            config=OuterLoopLaunchConfig(
                plan_id="launch-train-a",
                generated_at_utc="2026-04-25T05:00:00Z",
                launch_target=OuterLoopLaunchTarget.gcp_vm,
            ),
        )

    with pytest.raises(HarnessIOError, match="hide artifact locations"):
        build_training_launch_plan_from_lewm_adapter(
            lewm_trainer_adapter_plan=adapter_artifact,
            output_root=tmp_path / "training",
            config=OuterLoopLaunchConfig(
                plan_id="launch-train-a",
                generated_at_utc="2026-04-25T05:00:00Z",
                environment={"AIC_EXTRA_CONFIG": str(tmp_path / "undeclared.json")},
            ),
        )


def test_launch_config_rejects_command_override() -> None:
    with pytest.raises(HarnessIOError, match="unknown fields"):
        OuterLoopLaunchConfig.from_dict(
            {
                "plan_id": "launch-a",
                "generated_at_utc": "2026-04-25T05:00:00Z",
                "command": ["aic_controller"],
            }
        )


def test_live_eval_launch_plan_binds_passed_gate_without_launch(tmp_path: Path) -> None:
    gate_artifact = _passed_gate_artifact(
        tmp_path / "eval-command-override",
        eval_mode=CandidateEvalMode.gcp_live_eval,
    )

    plan = build_live_eval_launch_plan_from_gate(
        autonomous_gate_decision=gate_artifact,
        result_root=tmp_path / "eval_result",
        harness_root=tmp_path / "eval_result" / "harness",
        config=OuterLoopLaunchConfig(
            plan_id="launch-eval-a",
            generated_at_utc="2026-04-25T05:30:00Z",
            launch_target=OuterLoopLaunchTarget.gcp_vm,
            eval_run_id="eval-train-lewm-a",
            environment={"AIC_EVAL_TIMEOUT_SEC": "120"},
        ),
    )
    artifact = write_outer_loop_launch_plan(tmp_path / "eval_launch_plan.json", plan)
    reread = read_outer_loop_launch_plan_artifact(artifact)

    assert reread == plan
    assert plan.operation is OuterLoopLaunchOperation.live_eval
    assert plan.launch_target is OuterLoopLaunchTarget.gcp_vm
    assert plan.command == ("aic_lewm_policy/cloud/gcp/run_learned_eval.sh",)
    assert plan.timeout_seconds == 120.0
    assert plan.requires_manual_authorization is True
    assert plan.autonomous_launch_allowed is False
    assert plan.environment["AIC_EVAL_RUN_ID"] == "eval-train-lewm-a"
    assert plan.environment["AIC_EVAL_RESULT_ROOT"].endswith("/eval_result")
    assert plan.environment["AIC_HARNESS_ROOT"].endswith("/eval_result/harness")
    assert plan.environment["AIC_POLICY_TRAINING_REPORT_PATH"].endswith(
        "policy_training_report.json"
    )
    assert "AIC_RUNTIME_POLICY_CHECKPOINT_PATH" not in plan.environment
    assert plan.expected_outputs["run_manifest"].endswith("/eval_result/harness/run_manifest.json")


def test_local_live_eval_launch_plan_sets_runtime_checkpoint_env(tmp_path: Path) -> None:
    gate_artifact = _passed_gate_artifact(tmp_path, eval_mode=CandidateEvalMode.local_live_eval)

    plan = build_live_eval_launch_plan_from_gate(
        autonomous_gate_decision=gate_artifact,
        result_root=tmp_path / "eval_result",
        harness_root=tmp_path / "eval_result" / "harness",
        config=OuterLoopLaunchConfig(
            plan_id="launch-eval-local-a",
            generated_at_utc="2026-04-25T05:30:00Z",
            launch_target=OuterLoopLaunchTarget.local,
        ),
    )

    assert plan.command == ("aic_lewm_policy/cloud/gcp/vm_eval_learned_policy.sh",)
    assert plan.environment["AIC_RUNTIME_POLICY_CHECKPOINT_PATH"].endswith("policy.ckpt")


def test_live_eval_launch_plan_rejects_candidate_target_mismatch(
    tmp_path: Path,
) -> None:
    gate_artifact = _passed_gate_artifact(tmp_path, eval_mode=CandidateEvalMode.local_live_eval)

    with pytest.raises(HarnessIOError, match="launch_target must match"):
        build_live_eval_launch_plan_from_gate(
            autonomous_gate_decision=gate_artifact,
            result_root=tmp_path / "eval_result",
            harness_root=tmp_path / "eval_result" / "harness",
            config=OuterLoopLaunchConfig(
                plan_id="launch-eval-a",
                generated_at_utc="2026-04-25T05:30:00Z",
                launch_target=OuterLoopLaunchTarget.gcp_vm,
            ),
        )


def test_gcp_live_eval_launch_plan_rejects_runtime_checkpoint_env(
    tmp_path: Path,
) -> None:
    gate_artifact = _passed_gate_artifact(
        tmp_path / "eval-command-override",
        eval_mode=CandidateEvalMode.gcp_live_eval,
    )

    with pytest.raises(HarnessIOError, match="must not set AIC_RUNTIME_POLICY_CHECKPOINT_PATH"):
        build_live_eval_launch_plan_from_gate(
            autonomous_gate_decision=gate_artifact,
            result_root=tmp_path / "eval_result",
            harness_root=tmp_path / "eval_result" / "harness",
            config=OuterLoopLaunchConfig(
                plan_id="launch-eval-a",
                generated_at_utc="2026-04-25T05:30:00Z",
                launch_target=OuterLoopLaunchTarget.gcp_vm,
                environment={"AIC_RUNTIME_POLICY_CHECKPOINT_PATH": "/tmp/policy.ckpt"},
            ),
        )


def test_live_eval_launch_plan_rejects_blocked_gate(tmp_path: Path) -> None:
    gate_artifact = _blocked_gate_artifact(tmp_path)

    with pytest.raises(HarnessIOError, match="passed autonomous gate"):
        build_live_eval_launch_plan_from_gate(
            autonomous_gate_decision=gate_artifact,
            result_root=tmp_path / "eval_result",
            harness_root=tmp_path / "eval_result" / "harness",
            config=OuterLoopLaunchConfig(
                plan_id="launch-eval-a",
                generated_at_utc="2026-04-25T05:30:00Z",
            ),
        )


def test_launch_plan_rejects_authorization_and_output_alias_regressions(
    tmp_path: Path,
) -> None:
    adapter_artifact = _lewm_adapter_artifact(tmp_path)
    plan = build_training_launch_plan_from_lewm_adapter(
        lewm_trainer_adapter_plan=adapter_artifact,
        output_root=tmp_path / "training",
        config=OuterLoopLaunchConfig(
            plan_id="launch-train-a",
            generated_at_utc="2026-04-25T05:00:00Z",
        ),
    )

    payload = plan.to_dict()
    payload["requires_manual_authorization"] = False
    with pytest.raises(HarnessIOError, match="requires_manual_authorization"):
        OuterLoopLaunchPlan.from_dict(payload)

    payload = plan.to_dict()
    payload["expected_outputs"]["stderr"] = payload["expected_outputs"]["stdout"]
    with pytest.raises(HarnessIOError, match="must not alias each other"):
        OuterLoopLaunchPlan.from_dict(payload)

    source_path = plan.source_artifacts["materialized_dataset"].path
    assert source_path is not None
    payload = plan.to_dict()
    payload["expected_outputs"]["policy_checkpoint"] = source_path
    with pytest.raises(HarnessIOError, match="must not alias source artifact"):
        OuterLoopLaunchPlan.from_dict(payload)

    with pytest.raises(HarnessIOError, match="must not alias source artifact"):
        write_outer_loop_launch_plan(
            Path(plan.source_artifacts["materialized_dataset"].path or ""),
            plan,
            overwrite=True,
        )


def test_launch_plan_rejects_serialized_command_overrides(tmp_path: Path) -> None:
    adapter_artifact = _lewm_adapter_artifact(tmp_path)
    training_plan = build_training_launch_plan_from_lewm_adapter(
        lewm_trainer_adapter_plan=adapter_artifact,
        output_root=tmp_path / "training",
        config=OuterLoopLaunchConfig(
            plan_id="launch-train-a",
            generated_at_utc="2026-04-25T05:00:00Z",
        ),
    )
    payload = training_plan.to_dict()
    payload["command"] = ["/bin/echo", "override"]
    with pytest.raises(HarnessIOError, match="trainer.command"):
        OuterLoopLaunchPlan.from_dict(payload)

    gate_artifact = _passed_gate_artifact(
        tmp_path / "serialized-eval-command-override",
        eval_mode=CandidateEvalMode.gcp_live_eval,
    )
    eval_plan = build_live_eval_launch_plan_from_gate(
        autonomous_gate_decision=gate_artifact,
        result_root=tmp_path / "eval_result",
        harness_root=tmp_path / "eval_result" / "harness",
        config=OuterLoopLaunchConfig(
            plan_id="launch-eval-a",
            generated_at_utc="2026-04-25T05:30:00Z",
            launch_target=OuterLoopLaunchTarget.gcp_vm,
        ),
    )
    payload = eval_plan.to_dict()
    payload["command"] = ["/bin/echo", "override"]
    with pytest.raises(HarnessIOError, match="default command"):
        OuterLoopLaunchPlan.from_dict(payload)


def test_launch_plan_rejects_serialized_target_retargeting(tmp_path: Path) -> None:
    adapter_artifact = _lewm_adapter_artifact(tmp_path)
    training_plan = build_training_launch_plan_from_lewm_adapter(
        lewm_trainer_adapter_plan=adapter_artifact,
        output_root=tmp_path / "training",
        config=OuterLoopLaunchConfig(
            plan_id="launch-train-a",
            generated_at_utc="2026-04-25T05:00:00Z",
        ),
    )
    payload = training_plan.to_dict()
    payload["launch_target"] = "gcp_vm"
    with pytest.raises(HarnessIOError, match="launch_target"):
        OuterLoopLaunchPlan.from_dict(payload)

    gate_artifact = _passed_gate_artifact(
        tmp_path / "serialized-eval-target-override",
        eval_mode=CandidateEvalMode.gcp_live_eval,
    )
    eval_plan = build_live_eval_launch_plan_from_gate(
        autonomous_gate_decision=gate_artifact,
        result_root=tmp_path / "eval_target_result",
        harness_root=tmp_path / "eval_target_result" / "harness",
        config=OuterLoopLaunchConfig(
            plan_id="launch-eval-a",
            generated_at_utc="2026-04-25T05:30:00Z",
            launch_target=OuterLoopLaunchTarget.gcp_vm,
        ),
    )
    payload = eval_plan.to_dict()
    payload["launch_target"] = "local"
    payload["command"] = ["aic_lewm_policy/cloud/gcp/vm_eval_learned_policy.sh"]
    with pytest.raises(HarnessIOError, match="eval.mode"):
        OuterLoopLaunchPlan.from_dict(payload)


def test_launch_plan_artifact_rejects_provenance_regressions(tmp_path: Path) -> None:
    adapter_artifact = _lewm_adapter_artifact(tmp_path)
    plan = build_training_launch_plan_from_lewm_adapter(
        lewm_trainer_adapter_plan=adapter_artifact,
        output_root=tmp_path / "training",
        config=OuterLoopLaunchConfig(
            plan_id="launch-train-a",
            generated_at_utc="2026-04-25T05:00:00Z",
        ),
    )
    artifact = write_outer_loop_launch_plan(tmp_path / "training_launch_plan.json", plan)

    with pytest.raises(HarnessIOError, match="operation must match"):
        read_outer_loop_launch_plan_artifact(
            ArtifactRef(
                kind=artifact.kind,
                path=artifact.path,
                sha256=artifact.sha256,
                provenance={
                    **dict(artifact.provenance),
                    "operation": "live_eval",
                },
            )
        )


def _lewm_adapter_artifact(tmp_path: Path) -> ArtifactRef:
    dataset_artifact = _training_dataset_artifact(tmp_path / "dataset")
    materialization_artifact = _materialization_artifact(tmp_path, dataset_artifact)
    registry_artifact = _candidate_registry_artifact(tmp_path, dataset_artifact)
    plan = build_lewm_trainer_adapter_plan(
        candidate_registry=registry_artifact,
        dataset_materialization_report=materialization_artifact,
        config=LewmTrainerAdapterConfig(
            plan_id="lewm-plan-a",
            generated_at_utc="2026-04-25T04:00:00Z",
            train_run_id="train-lewm-a",
            timeout_seconds=120.0,
        ),
    )
    return write_lewm_trainer_adapter_plan(tmp_path / "lewm_trainer_adapter_plan.json", plan)


def _passed_gate_artifact(
    tmp_path: Path,
    *,
    eval_mode: CandidateEvalMode = CandidateEvalMode.local_live_eval,
) -> ArtifactRef:
    dataset_artifact = _training_dataset_artifact(tmp_path / "gate-dataset")
    trainer = _trainer()
    template = _policy_template()
    registry_artifact = _candidate_registry_artifact(
        tmp_path,
        dataset_artifact,
        trainer=trainer,
        template=template,
        eval_mode=eval_mode,
    )
    training = run_policy_training_command(
        run_id="train-lewm-a",
        generated_at_utc="2026-04-25T04:30:00Z",
        source_training_dataset_report=dataset_artifact,
        trainer=trainer,
        candidate_policy_template=template,
        policy_checkpoint_path=tmp_path / "training" / "policy.ckpt",
        stdout_path=tmp_path / "training" / "stdout.log",
        stderr_path=tmp_path / "training" / "stderr.log",
        execution_report_path=tmp_path / "training" / "execution.json",
        report_path=tmp_path / "training" / "policy_training_report.json",
        timeout_seconds=5.0,
        metrics={"train.loss": 0.01, "validation.loss": 0.02},
    )
    report_artifact = ArtifactRef(
        kind="policy_training_report",
        path=training.report_path,
        sha256=sha256_file(Path(training.report_path or "")),
        provenance={
            "producer": "pytest",
            "derivation": "run_policy_training_command",
            "run_id": training.report.run_id,
        },
    )
    decision = evaluate_autonomous_training_gate(
        candidate_registry=registry_artifact,
        policy_training_report=report_artifact,
        generated_at_utc="2026-04-25T04:45:00Z",
        config=AutonomousGateConfig(
            gate_id="trained_policy_pre_eval",
            max_train_validation_loss_gap=0.1,
        ),
    )
    assert decision.verdict is AutonomousGateVerdict.passed
    return write_autonomous_gate_decision(tmp_path / "autonomous_gate.json", decision)


def _blocked_gate_artifact(tmp_path: Path) -> ArtifactRef:
    dataset_artifact = _training_dataset_artifact(tmp_path / "blocked-dataset")
    trainer = _trainer()
    template = _policy_template()
    registry_artifact = _candidate_registry_artifact(
        tmp_path,
        dataset_artifact,
        trainer=trainer,
        template=template,
    )
    bad_report_path = tmp_path / "bad_policy_training_report.json"
    write_json(bad_report_path, {"schema_version": 1}, overwrite=True)
    bad_report_artifact = ArtifactRef(
        kind="policy_training_report",
        path=str(bad_report_path),
        sha256=sha256_file(bad_report_path),
        provenance={"producer": "pytest", "derivation": "malformed", "run_id": "bad"},
    )
    decision = evaluate_autonomous_training_gate(
        candidate_registry=registry_artifact,
        policy_training_report=bad_report_artifact,
        generated_at_utc="2026-04-25T04:45:00Z",
        config=AutonomousGateConfig(gate_id="trained_policy_pre_eval"),
    )
    return write_autonomous_gate_decision(tmp_path / "blocked_gate.json", decision)


def _candidate_registry_artifact(
    tmp_path: Path,
    dataset_artifact: ArtifactRef,
    *,
    trainer: TrainerInvocation | None = None,
    template: CandidatePolicyTemplate | None = None,
    eval_mode: CandidateEvalMode = CandidateEvalMode.local_live_eval,
) -> ArtifactRef:
    typed_trainer = _trainer() if trainer is None else trainer
    typed_template = _policy_template() if template is None else template
    candidate = CandidatePolicyRecord(
        generated_at_utc="2026-04-25T03:30:00Z",
        source_training_dataset_report=dataset_artifact,
        trainer=typed_trainer,
        policy_template=typed_template,
        eval=CandidateEvalSpec(
            mode=eval_mode,
            gate_id="trained_policy_pre_eval",
            planner_mode="lewm_mpc",
        ),
    )
    registry = CandidatePolicyRegistry(
        registry_id="registry-lewm-a",
        generated_at_utc="2026-04-25T03:30:00Z",
        candidates=(candidate,),
        selected_candidate_id=candidate.candidate_id,
    )
    return write_candidate_policy_registry(tmp_path / "candidate_registry.json", registry)


def _materialization_artifact(
    tmp_path: Path,
    dataset_artifact: ArtifactRef,
) -> ArtifactRef:
    report = materialize_training_dataset_jsonl(
        source_training_dataset_report=dataset_artifact,
        output_path=tmp_path / "trainer_dataset.jsonl",
        generated_at_utc="2026-04-25T03:00:00Z",
    )
    return write_training_dataset_materialization_report(
        tmp_path / "training_dataset_materialization_report.json",
        report,
    )


def _trainer() -> TrainerInvocation:
    return TrainerInvocation(
        trainer_id="lewm-trainer-v1",
        trainer_name="LEWM trainer",
        backend_kind=BackendKind.lewm_world_model,
        command=(sys.executable, "-c", _write_checkpoint_script()),
        config={"max_epochs": 1},
        seed=7,
        offline_only=True,
        runtime_allowed=False,
        uses_online_language_model_control=False,
    )


def _write_checkpoint_script() -> str:
    return (
        "import os, pathlib; "
        "path=pathlib.Path(os.environ['AIC_POLICY_TRAINING_OUTPUT_CHECKPOINT']); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text('checkpoint', encoding='utf-8'); "
        "print('checkpoint ready')"
    )


def _policy_template() -> CandidatePolicyTemplate:
    return CandidatePolicyTemplate(
        backend_kind=BackendKind.lewm_world_model,
        name="lewm-candidate",
        training_sources=(TrainingSourceKind.official_demo,),
        simulator_sources=(SimulatorKind.offline_replay,),
        legal_observation_contract="official aic_model observations only",
        description="LEWM candidate.",
        config={"planner_mode": "lewm_mpc"},
        deterministic=True,
    )


def _training_dataset_artifact(tmp_path: Path) -> ArtifactRef:
    tmp_path.mkdir(parents=True, exist_ok=True)
    signal_report = TrainingSignalReport(
        run_id="source-run-a",
        generated_at_utc="2026-04-25T02:00:00Z",
        source_trace=ArtifactRef(
            kind="episode_trace",
            uri="memory://pytest/episode_trace.json",
            sha256="a" * 64,
            provenance={
                "producer": "pytest",
                "derivation": "fixture",
                "run_id": "source-run-a",
            },
        ),
        signals=(
            TrainingSignal(
                signal_id="signal-action-1",
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
    )
    signal_path = tmp_path / "training_signal_report.json"
    write_json(signal_path, signal_report.to_dict(), overwrite=True)
    signal_artifact = ArtifactRef(
        kind="training_signal_report",
        path=str(signal_path),
        sha256=sha256_file(signal_path),
        provenance={
            "producer": "pytest",
            "derivation": "derive_training_signal_report",
            "run_id": signal_report.run_id,
        },
    )
    dataset_report = derive_training_dataset_report(
        signal_report,
        source_training_signal_report=signal_artifact,
    )
    dataset_path = tmp_path / "training_dataset_report.json"
    write_json(dataset_path, dataset_report.to_dict(), overwrite=True)
    return ArtifactRef(
        kind="training_dataset_report",
        path=str(dataset_path),
        sha256=sha256_file(dataset_path),
        provenance={
            "producer": "pytest",
            "derivation": "derive_training_dataset_report",
            "run_id": dataset_report.run_id,
        },
    )
