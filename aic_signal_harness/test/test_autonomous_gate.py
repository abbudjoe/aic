from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

from aic_signal_harness import (
    AIC_POLICY_TRAINING_OUTPUT_CHECKPOINT,
    AIC_POLICY_TRAINING_SOURCE_DATASET_REPORT,
    AUTONOMOUS_GATE_DECISION_KIND,
    ArtifactRef,
    AutonomousGateConfig,
    AutonomousGateFailureKind,
    AutonomousGateVerdict,
    BackendKind,
    CandidateCheckpointExpectation,
    CandidateEvalMode,
    CandidateEvalSpec,
    CandidatePolicyRecord,
    CandidatePolicyRegistry,
    CandidateSafetyContract,
    CandidatePolicyTemplate,
    HarnessIOError,
    LeakageClass,
    PolicyTrainingCommandResult,
    SimulatorKind,
    TrainerInvocation,
    TrainingSignal,
    TrainingSignalKind,
    TrainingSignalReport,
    TrainingSourceKind,
    derive_training_dataset_report,
    evaluate_autonomous_training_gate,
    read_autonomous_gate_decision_artifact,
    run_policy_training_command,
    sha256_file,
    write_autonomous_gate_decision,
    write_candidate_policy_registry,
    write_json,
)


def test_autonomous_gate_passes_bound_training_evidence(tmp_path: Path) -> None:
    source_dataset = _training_dataset_artifact(tmp_path / "candidate")
    trainer = _trainer()
    template = _candidate_policy_template()
    registry_artifact, selected_candidate_id = _candidate_registry_artifact(
        tmp_path,
        source_dataset=source_dataset,
        trainer=trainer,
        template=template,
    )
    training = _run_training(
        tmp_path / "training",
        source_dataset=source_dataset,
        trainer=trainer,
        template=template,
        metrics={"train.loss": 0.05, "validation.loss": 0.06},
    )
    report_artifact = _policy_training_report_artifact(training)

    decision = evaluate_autonomous_training_gate(
        candidate_registry=registry_artifact,
        policy_training_report=report_artifact,
        generated_at_utc="2026-04-25T01:00:00Z",
        config=AutonomousGateConfig(
            gate_id="trained_policy_pre_eval",
            min_training_examples=1,
            max_train_validation_loss_gap=0.1,
        ),
    )
    artifact = write_autonomous_gate_decision(tmp_path / "autonomous_gate.json", decision)
    reread = read_autonomous_gate_decision_artifact(artifact)

    assert decision.verdict is AutonomousGateVerdict.passed
    assert decision.eval_preconditions_met is True
    assert decision.autonomous_launch_allowed is False
    assert decision.failures == ()
    assert decision.selected_candidate_id == selected_candidate_id
    assert decision.policy_artifact == training.report.policy_artifact
    assert artifact.kind == AUTONOMOUS_GATE_DECISION_KIND
    assert reread == decision


def test_autonomous_gate_blocks_invalid_training_report_without_launch(tmp_path: Path) -> None:
    source_dataset = _training_dataset_artifact(tmp_path / "candidate")
    registry_artifact, _ = _candidate_registry_artifact(
        tmp_path,
        source_dataset=source_dataset,
        trainer=_trainer(),
        template=_candidate_policy_template(),
    )
    bad_report_path = tmp_path / "bad_policy_training_report.json"
    write_json(bad_report_path, {"schema_version": 1})
    bad_report_artifact = ArtifactRef(
        kind="policy_training_report",
        path=str(bad_report_path),
        sha256=sha256_file(bad_report_path),
        provenance={"producer": "pytest", "derivation": "malformed", "run_id": "bad"},
    )

    decision = evaluate_autonomous_training_gate(
        candidate_registry=registry_artifact,
        policy_training_report=bad_report_artifact,
        generated_at_utc="2026-04-25T01:00:00Z",
        config=AutonomousGateConfig(gate_id="trained_policy_pre_eval"),
    )

    assert decision.verdict is AutonomousGateVerdict.blocked
    assert decision.eval_preconditions_met is False
    assert decision.autonomous_launch_allowed is False
    assert tuple(failure.kind for failure in decision.failures) == (
        AutonomousGateFailureKind.training_report_invalid,
    )
    assert decision.train_run_id is None
    assert decision.policy_artifact is None


def test_autonomous_gate_blocks_candidate_dataset_and_leakage_mismatch(tmp_path: Path) -> None:
    candidate_dataset = _training_dataset_artifact(tmp_path / "candidate")
    report_dataset = _training_dataset_artifact(
        tmp_path / "privileged",
        leakage_class=LeakageClass.privileged_training_signal,
    )
    trainer = _trainer()
    template = _candidate_policy_template()
    registry_artifact, _ = _candidate_registry_artifact(
        tmp_path,
        source_dataset=candidate_dataset,
        trainer=trainer,
        template=template,
    )
    training = _run_training(
        tmp_path / "training",
        source_dataset=report_dataset,
        trainer=trainer,
        template=template,
        allowed_training_leakage_classes=(
            LeakageClass.legal_policy_action_output,
            LeakageClass.privileged_training_signal,
        ),
    )

    decision = evaluate_autonomous_training_gate(
        candidate_registry=registry_artifact,
        policy_training_report=_policy_training_report_artifact(training),
        generated_at_utc="2026-04-25T01:00:00Z",
        config=AutonomousGateConfig(gate_id="trained_policy_pre_eval"),
    )

    failure_kinds = {failure.kind for failure in decision.failures}
    assert decision.verdict is AutonomousGateVerdict.blocked
    assert AutonomousGateFailureKind.candidate_binding_mismatch in failure_kinds
    assert AutonomousGateFailureKind.leakage_violation in failure_kinds


def test_autonomous_gate_blocks_uncontrolled_overfit_and_stale_checkpoint(
    tmp_path: Path,
) -> None:
    source_dataset = _training_dataset_artifact(tmp_path / "candidate")
    trainer = _trainer()
    template = _candidate_policy_template()
    registry_artifact, _ = _candidate_registry_artifact(
        tmp_path,
        source_dataset=source_dataset,
        trainer=trainer,
        template=template,
        required_sha256="0" * 64,
    )
    training = _run_training(
        tmp_path / "training",
        source_dataset=source_dataset,
        trainer=trainer,
        template=template,
        metrics={"train.loss": 0.01, "validation.loss": 0.5},
    )
    uncontrolled_payload = training.report.to_dict()
    del uncontrolled_payload["policy_training_execution_report"]
    _remove_execution_sha(uncontrolled_payload["policy_artifact"])
    _remove_execution_sha(uncontrolled_payload["candidate_policy"])
    _remove_execution_sha(
        uncontrolled_payload["candidate_policy"]["runtime_boundary"]["policy_artifact"]
    )
    report_artifact = _write_policy_training_report_artifact(
        tmp_path,
        uncontrolled_payload,
        "uncontrolled_policy_training_report.json",
    )

    decision = evaluate_autonomous_training_gate(
        candidate_registry=registry_artifact,
        policy_training_report=report_artifact,
        generated_at_utc="2026-04-25T01:00:00Z",
        config=AutonomousGateConfig(
            gate_id="trained_policy_pre_eval",
            max_train_validation_loss_gap=0.1,
        ),
    )

    failure_kinds = {failure.kind for failure in decision.failures}
    assert decision.verdict is AutonomousGateVerdict.blocked
    assert AutonomousGateFailureKind.checkpoint_stale in failure_kinds
    assert AutonomousGateFailureKind.execution_not_controlled in failure_kinds
    assert AutonomousGateFailureKind.overfitting_risk in failure_kinds


def test_autonomous_gate_config_requires_controlled_execution() -> None:
    with pytest.raises(HarnessIOError, match="require_controlled_execution must be true"):
        AutonomousGateConfig(
            gate_id="trained_policy_pre_eval",
            require_controlled_execution=False,
        )


def test_autonomous_gate_blocks_candidate_gate_mismatch_and_dataset_quality(
    tmp_path: Path,
) -> None:
    source_dataset = _training_dataset_artifact(tmp_path / "candidate")
    trainer = _trainer()
    template = _candidate_policy_template()
    registry_artifact, _ = _candidate_registry_artifact(
        tmp_path,
        source_dataset=source_dataset,
        trainer=trainer,
        template=template,
        gate_id="other_gate",
    )
    training = _run_training(
        tmp_path / "training",
        source_dataset=source_dataset,
        trainer=trainer,
        template=template,
    )

    decision = evaluate_autonomous_training_gate(
        candidate_registry=registry_artifact,
        policy_training_report=_policy_training_report_artifact(training),
        generated_at_utc="2026-04-25T01:00:00Z",
        config=AutonomousGateConfig(
            gate_id="trained_policy_pre_eval",
            min_training_examples=2,
        ),
    )

    failure_kinds = {failure.kind for failure in decision.failures}
    assert decision.verdict is AutonomousGateVerdict.blocked
    assert AutonomousGateFailureKind.candidate_binding_mismatch in failure_kinds
    assert AutonomousGateFailureKind.dataset_quality in failure_kinds


def test_autonomous_gate_decision_read_revalidates_candidate_gate_binding(
    tmp_path: Path,
) -> None:
    source_dataset = _training_dataset_artifact(tmp_path / "candidate")
    trainer = _trainer()
    template = _candidate_policy_template()
    good_registry_artifact, _ = _candidate_registry_artifact(
        tmp_path,
        source_dataset=source_dataset,
        trainer=trainer,
        template=template,
    )
    training = _run_training(
        tmp_path / "training",
        source_dataset=source_dataset,
        trainer=trainer,
        template=template,
    )
    decision = evaluate_autonomous_training_gate(
        candidate_registry=good_registry_artifact,
        policy_training_report=_policy_training_report_artifact(training),
        generated_at_utc="2026-04-25T01:00:00Z",
        config=AutonomousGateConfig(gate_id="trained_policy_pre_eval"),
    )
    mismatched_registry_artifact, _ = _candidate_registry_artifact(
        tmp_path,
        source_dataset=source_dataset,
        trainer=trainer,
        template=template,
        gate_id="other_gate",
        registry_name="candidate_registry_other_gate.json",
    )
    stale_payload = decision.to_dict()
    stale_payload["candidate_registry"] = mismatched_registry_artifact.to_dict()

    with pytest.raises(HarnessIOError, match="candidate eval.gate_id"):
        write_autonomous_gate_decision(
            tmp_path / "stale_autonomous_gate.json",
            stale_payload,
        )


def test_autonomous_gate_blocks_timeout_risk_and_missing_overfit_metrics(
    tmp_path: Path,
) -> None:
    source_dataset = _training_dataset_artifact(tmp_path / "candidate")
    trainer = _trainer()
    template = _candidate_policy_template()
    registry_artifact, _ = _candidate_registry_artifact(
        tmp_path,
        source_dataset=source_dataset,
        trainer=trainer,
        template=template,
        safety=CandidateSafetyContract(max_training_runtime_seconds=1.0),
    )
    training = _run_training(
        tmp_path / "training",
        source_dataset=source_dataset,
        trainer=trainer,
        template=template,
    )

    decision = evaluate_autonomous_training_gate(
        candidate_registry=registry_artifact,
        policy_training_report=_policy_training_report_artifact(training),
        generated_at_utc="2026-04-25T01:00:00Z",
        config=AutonomousGateConfig(
            gate_id="trained_policy_pre_eval",
            max_train_validation_loss_gap=0.1,
        ),
    )

    failure_kinds = {failure.kind for failure in decision.failures}
    assert decision.verdict is AutonomousGateVerdict.blocked
    assert AutonomousGateFailureKind.timeout_risk in failure_kinds
    assert AutonomousGateFailureKind.overfitting_risk in failure_kinds


def test_autonomous_gate_decision_reader_fails_closed_on_provenance(tmp_path: Path) -> None:
    source_dataset = _training_dataset_artifact(tmp_path / "candidate")
    trainer = _trainer()
    template = _candidate_policy_template()
    registry_artifact, _ = _candidate_registry_artifact(
        tmp_path,
        source_dataset=source_dataset,
        trainer=trainer,
        template=template,
    )
    training = _run_training(
        tmp_path / "training",
        source_dataset=source_dataset,
        trainer=trainer,
        template=template,
    )
    decision = evaluate_autonomous_training_gate(
        candidate_registry=registry_artifact,
        policy_training_report=_policy_training_report_artifact(training),
        generated_at_utc="2026-04-25T01:00:00Z",
        config=AutonomousGateConfig(gate_id="trained_policy_pre_eval"),
    )
    artifact = write_autonomous_gate_decision(tmp_path / "autonomous_gate.json", decision)

    with pytest.raises(HarnessIOError, match="provenance producer must be"):
        read_autonomous_gate_decision_artifact(
            ArtifactRef(
                kind=artifact.kind,
                path=artifact.path,
                sha256=artifact.sha256,
                provenance={
                    "producer": "pytest",
                    "derivation": "write_autonomous_gate_decision",
                    "gate_id": decision.gate_id,
                    "verdict": decision.verdict.value,
                    "selected_candidate_id": decision.selected_candidate_id,
                },
            )
        )
    with pytest.raises(HarnessIOError, match="provenance verdict must match"):
        read_autonomous_gate_decision_artifact(
            ArtifactRef(
                kind=artifact.kind,
                path=artifact.path,
                sha256=artifact.sha256,
                provenance={
                    "producer": "aic_signal_harness.autonomous_gate",
                    "derivation": "write_autonomous_gate_decision",
                    "gate_id": decision.gate_id,
                    "verdict": "blocked",
                    "selected_candidate_id": decision.selected_candidate_id,
                },
            )
        )


def _candidate_registry_artifact(
    tmp_path: Path,
    *,
    source_dataset: ArtifactRef,
    trainer: TrainerInvocation,
    template: CandidatePolicyTemplate,
    required_sha256: str | None = None,
    safety: CandidateSafetyContract | None = None,
    gate_id: str = "trained_policy_pre_eval",
    registry_name: str = "candidate_registry.json",
) -> tuple[ArtifactRef, str]:
    candidate = CandidatePolicyRecord(
        generated_at_utc="2026-04-25T00:00:00Z",
        source_training_dataset_report=source_dataset,
        trainer=trainer,
        policy_template=template,
        checkpoint=CandidateCheckpointExpectation(
            output_name="policy.ckpt",
            required_sha256=required_sha256,
        ),
        eval=CandidateEvalSpec(
            mode=CandidateEvalMode.local_live_eval,
            gate_id=gate_id,
            planner_mode="lewm_mpc",
            min_improvement=1.0,
        ),
        safety=CandidateSafetyContract() if safety is None else safety,
    )
    registry = CandidatePolicyRegistry(
        registry_id="registry-gate-a",
        generated_at_utc="2026-04-25T00:00:00Z",
        candidates=(candidate,),
        selected_candidate_id=candidate.candidate_id,
    )
    artifact = write_candidate_policy_registry(tmp_path / registry_name, registry)
    assert candidate.candidate_id is not None
    return artifact, candidate.candidate_id


def _run_training(
    root: Path,
    *,
    source_dataset: ArtifactRef,
    trainer: TrainerInvocation,
    template: CandidatePolicyTemplate,
    metrics: Mapping[str, float] | None = None,
    allowed_training_leakage_classes: tuple[LeakageClass, ...] = (
        LeakageClass.legal_policy_input,
        LeakageClass.legal_policy_action_output,
    ),
) -> PolicyTrainingCommandResult:
    root.mkdir(parents=True, exist_ok=True)
    return run_policy_training_command(
        run_id="train-run-a",
        generated_at_utc="2026-04-25T00:30:00Z",
        source_training_dataset_report=source_dataset,
        trainer=trainer,
        candidate_policy_template=template,
        policy_checkpoint_path=root / "policy.ckpt",
        stdout_path=root / "trainer.stdout.log",
        stderr_path=root / "trainer.stderr.log",
        execution_report_path=root / "policy_training_execution_report.json",
        report_path=root / "policy_training_report.json",
        timeout_seconds=5.0,
        metrics={} if metrics is None else metrics,
        allowed_training_leakage_classes=allowed_training_leakage_classes,
    )


def _trainer() -> TrainerInvocation:
    return TrainerInvocation(
        trainer_id="lewm-runner-trainer-v1",
        trainer_name="LEWM controlled runner trainer",
        backend_kind=BackendKind.lewm_world_model,
        command=(sys.executable, "-c", _write_checkpoint_script()),
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
        deterministic=True,
    )


def _training_dataset_artifact(
    root: Path,
    *,
    leakage_class: LeakageClass = LeakageClass.legal_policy_action_output,
) -> ArtifactRef:
    root.mkdir(parents=True, exist_ok=True)
    signal_report = TrainingSignalReport(
        run_id=f"source-{root.name}",
        generated_at_utc="2026-04-25T00:00:00Z",
        source_trace=ArtifactRef(
            kind="episode_trace",
            uri=f"memory://pytest/{root.name}/episode_trace.json",
            sha256="a" * 64,
            provenance={
                "producer": "pytest",
                "run_id": f"source-{root.name}",
                "derivation": "derive_episode_trace",
            },
        ),
        signals=(
            TrainingSignal(
                signal_id=f"tsig_{root.name}_action_1",
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
    signal_path = root / "training_signal_report.json"
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
    dataset_path = root / "training_dataset_report.json"
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


def _policy_training_report_artifact(training: PolicyTrainingCommandResult) -> ArtifactRef:
    assert training.report_path is not None
    return ArtifactRef(
        kind="policy_training_report",
        path=training.report_path,
        sha256=sha256_file(training.report_path),
        provenance={
            "producer": "pytest",
            "derivation": "run_policy_training_command",
            "run_id": training.report.run_id,
        },
    )


def _write_policy_training_report_artifact(
    tmp_path: Path,
    payload: Mapping[str, Any],
    name: str,
) -> ArtifactRef:
    path = tmp_path / name
    write_json(path, payload, overwrite=True)
    return ArtifactRef(
        kind="policy_training_report",
        path=str(path),
        sha256=sha256_file(path),
        provenance={
            "producer": "pytest",
            "derivation": "mutated_policy_training_report",
            "run_id": payload.get("run_id"),
        },
    )


def _remove_execution_sha(artifact_payload: dict[str, Any]) -> None:
    artifact_payload["provenance"].pop("policy_training_execution_report_sha256")


def _write_checkpoint_script() -> str:
    return (
        "import os, pathlib; "
        f"source = pathlib.Path(os.environ[{AIC_POLICY_TRAINING_SOURCE_DATASET_REPORT!r}]); "
        f"checkpoint = pathlib.Path(os.environ[{AIC_POLICY_TRAINING_OUTPUT_CHECKPOINT!r}]); "
        "checkpoint.write_bytes(b'runner checkpoint:' + source.read_bytes()[:16]); "
        "print('checkpoint ready')"
    )
