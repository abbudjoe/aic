from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pytest

from aic_signal_harness import (
    ArtifactRef,
    BackendKind,
    CANDIDATE_POLICY_REGISTRY_KIND as EXPORTED_CANDIDATE_POLICY_REGISTRY_KIND,
    CandidateCheckpointExpectation as ExportedCandidateCheckpointExpectation,
    CandidateEvalMode as ExportedCandidateEvalMode,
    CandidateEvalSpec as ExportedCandidateEvalSpec,
    CandidatePolicyRecord as ExportedCandidatePolicyRecord,
    CandidatePolicyRegistry as ExportedCandidatePolicyRegistry,
    CandidateSafetyContract as ExportedCandidateSafetyContract,
    CandidatePolicyTemplate,
    DEFAULT_RUNTIME_CHECKPOINT_CONTAINER_PATH as EXPORTED_DEFAULT_RUNTIME_CHECKPOINT_CONTAINER_PATH,
    HarnessIOError,
    LeakageClass,
    NextExperimentAction,
    NextExperimentCandidate,
    NextExperimentCandidateKind,
    NextExperimentDecision,
    NextExperimentExecutionMode,
    NextExperimentPlan,
    NextExperimentPlanCandidate,
    NextExperimentPlanStatus,
    NextExperimentPriority,
    NextExperimentReport,
    PolicyBackendSpec,
    RunManifest,
    RuntimeRole,
    SimulatorKind,
    TrainerInvocation,
    TrainingSignal,
    TrainingSignalKind,
    TrainingSignalReport,
    TrainingSourceKind,
    candidate_policy_fingerprint_sha256 as exported_candidate_policy_fingerprint_sha256,
    candidate_policy_record_id_from_fingerprint as exported_candidate_policy_record_id_from_fingerprint,
    derive_training_dataset_report,
    read_candidate_policy_registry_artifact as exported_read_candidate_policy_registry_artifact,
    sha256_file,
    write_candidate_policy_registry as exported_write_candidate_policy_registry,
    write_json,
)
from aic_signal_harness.candidate_registry import (
    CandidateCheckpointExpectation,
    CandidateEvalMode,
    CandidateEvalSpec,
    CandidatePolicyRecord,
    CandidatePolicyRegistry,
    CandidateSafetyContract,
    candidate_policy_fingerprint_sha256,
    read_candidate_policy_registry_artifact,
    write_candidate_policy_registry,
)
from aic_signal_harness.train_eval_promote import (
    DEFAULT_RUNTIME_CHECKPOINT_CONTAINER_PATH as TRAIN_EVAL_PROMOTE_RUNTIME_CHECKPOINT_CONTAINER_PATH,
)


def test_candidate_policy_registry_round_trips_and_writes_artifact(tmp_path: Path) -> None:
    candidate = _candidate_record(tmp_path)
    registry = CandidatePolicyRegistry(
        registry_id="registry-gate2-a",
        generated_at_utc="2026-04-25T00:00:00Z",
        candidates=(candidate,),
        selected_candidate_id=candidate.candidate_id,
    )

    artifact = write_candidate_policy_registry(tmp_path / "candidate_registry.json", registry)
    decoded = read_candidate_policy_registry_artifact(artifact)

    assert decoded == registry
    assert artifact.kind == "candidate_policy_registry"
    assert artifact.sha256 == sha256_file(tmp_path / "candidate_registry.json")
    assert decoded.selected_candidate == candidate
    assert candidate.candidate_id is not None
    assert candidate.candidate_id.startswith("cand_")
    assert candidate.fingerprint_sha256 == candidate_policy_fingerprint_sha256(candidate)
    assert EXPORTED_CANDIDATE_POLICY_REGISTRY_KIND == "candidate_policy_registry"
    assert (
        EXPORTED_DEFAULT_RUNTIME_CHECKPOINT_CONTAINER_PATH
        == TRAIN_EVAL_PROMOTE_RUNTIME_CHECKPOINT_CONTAINER_PATH
    )
    assert ExportedCandidateCheckpointExpectation is CandidateCheckpointExpectation
    assert ExportedCandidateEvalMode is CandidateEvalMode
    assert ExportedCandidateEvalSpec is CandidateEvalSpec
    assert ExportedCandidatePolicyRecord is CandidatePolicyRecord
    assert ExportedCandidatePolicyRegistry is CandidatePolicyRegistry
    assert ExportedCandidateSafetyContract is CandidateSafetyContract
    assert exported_candidate_policy_fingerprint_sha256 is candidate_policy_fingerprint_sha256
    assert callable(exported_candidate_policy_record_id_from_fingerprint)
    assert exported_read_candidate_policy_registry_artifact is read_candidate_policy_registry_artifact
    assert exported_write_candidate_policy_registry is write_candidate_policy_registry


def test_candidate_policy_registry_artifact_requires_provenance(tmp_path: Path) -> None:
    candidate = _candidate_record(tmp_path)
    registry = CandidatePolicyRegistry(
        registry_id="registry-gate2-a",
        generated_at_utc="2026-04-25T00:00:00Z",
        candidates=(candidate,),
        selected_candidate_id=candidate.candidate_id,
    )
    artifact = write_candidate_policy_registry(tmp_path / "candidate_registry.json", registry)

    with pytest.raises(HarnessIOError, match="selected_candidate_id"):
        write_candidate_policy_registry(
            tmp_path / "unselected_registry.json",
            CandidatePolicyRegistry(
                registry_id="registry-gate2-unselected",
                generated_at_utc="2026-04-25T00:00:00Z",
                candidates=(candidate,),
            ),
        )

    with pytest.raises(HarnessIOError, match="provenance"):
        read_candidate_policy_registry_artifact(
            ArtifactRef(
                kind=artifact.kind,
                path=artifact.path,
                sha256=artifact.sha256,
            )
        )
    with pytest.raises(HarnessIOError, match="provenance.producer"):
        read_candidate_policy_registry_artifact(
            ArtifactRef(
                kind=artifact.kind,
                path=artifact.path,
                sha256=artifact.sha256,
                provenance={
                    "producer": "",
                    "derivation": "write_candidate_policy_registry",
                    "registry_id": registry.registry_id,
                    "selected_candidate_id": registry.selected_candidate_id,
                },
            )
        )


def test_candidate_policy_record_fingerprint_ignores_timestamp_and_notes(tmp_path: Path) -> None:
    first = _candidate_record(tmp_path, generated_at_utc="2026-04-25T00:00:00Z")
    second = _candidate_record(
        tmp_path,
        generated_at_utc="2026-04-25T00:10:00Z",
        notes=("operator annotation",),
    )

    assert second.candidate_id == first.candidate_id
    assert second.fingerprint_sha256 == first.fingerprint_sha256


def test_candidate_policy_registry_rejects_duplicate_candidate_ids(tmp_path: Path) -> None:
    candidate = _candidate_record(tmp_path)

    with pytest.raises(HarnessIOError, match="duplicate candidate_id"):
        CandidatePolicyRegistry(
            registry_id="registry-gate2-a",
            generated_at_utc="2026-04-25T00:00:00Z",
            candidates=(candidate, candidate),
        )


def test_candidate_policy_registry_selected_candidate_must_exist(tmp_path: Path) -> None:
    with pytest.raises(HarnessIOError, match="selected_candidate_id"):
        CandidatePolicyRegistry(
            registry_id="registry-gate2-a",
            generated_at_utc="2026-04-25T00:00:00Z",
            candidates=(_candidate_record(tmp_path),),
            selected_candidate_id="cand_missing",
        )


def test_candidate_policy_record_rejects_illegal_safety_and_runtime_flags(
    tmp_path: Path,
) -> None:
    with pytest.raises(HarnessIOError, match="allowed_training_leakage_classes"):
        _candidate_record(
            tmp_path,
            safety=CandidateSafetyContract(
                allowed_training_leakage_classes=(LeakageClass.privileged_eval_signal,),
            ),
        )

    with pytest.raises(HarnessIOError, match="autonomous_launch_allowed"):
        _candidate_record(
            tmp_path,
            safety=CandidateSafetyContract(autonomous_launch_allowed=True),
        )

    trainer_payload = _trainer().to_dict()
    trainer_payload["runtime_allowed"] = True
    with pytest.raises(HarnessIOError, match="runtime_allowed"):
        CandidatePolicyRecord(
            generated_at_utc="2026-04-25T00:00:00Z",
            source_training_dataset_report=_training_dataset_artifact(tmp_path),
            trainer=cast(Any, trainer_payload),
            policy_template=_policy_template(),
            eval=_eval_spec(),
        )

    with pytest.raises(HarnessIOError, match="require_policy_trace"):
        _candidate_record(
            tmp_path,
            safety=CandidateSafetyContract(require_policy_trace=False),
        )


def test_candidate_policy_record_rejects_dataset_leakage_outside_allowlist(
    tmp_path: Path,
) -> None:
    with pytest.raises(HarnessIOError, match="outside candidate safety"):
        CandidatePolicyRecord(
            generated_at_utc="2026-04-25T00:00:00Z",
            source_training_dataset_report=_training_dataset_artifact(
                tmp_path,
                include_privileged_signal=True,
            ),
            trainer=_trainer(),
            policy_template=_policy_template(),
            eval=_eval_spec(),
            safety=CandidateSafetyContract(
                allowed_training_leakage_classes=(LeakageClass.legal_policy_action_output,),
            ),
        )


def test_candidate_policy_record_validates_source_next_experiment_plan_artifact(
    tmp_path: Path,
) -> None:
    dataset_artifact = _training_dataset_artifact(tmp_path)
    plan_artifact = _next_experiment_plan_artifact(tmp_path)

    candidate = CandidatePolicyRecord(
        generated_at_utc="2026-04-25T00:00:00Z",
        source_training_dataset_report=dataset_artifact,
        trainer=_trainer(),
        policy_template=_policy_template(),
        eval=_eval_spec(),
        source_next_experiment_plan=plan_artifact,
        source_plan_candidate_id="next-candidate-a",
    )
    assert candidate.source_next_experiment_plan == plan_artifact

    with pytest.raises(HarnessIOError, match="source_next_experiment_plan"):
        CandidatePolicyRecord(
            generated_at_utc="2026-04-25T00:00:00Z",
            source_training_dataset_report=dataset_artifact,
            trainer=_trainer(),
            policy_template=_policy_template(),
            eval=_eval_spec(),
            source_next_experiment_plan=ArtifactRef(
                kind="run_manifest",
                path=dataset_artifact.path,
                sha256=dataset_artifact.sha256,
                provenance={},
            ),
            source_plan_candidate_id="next-candidate-a",
        )

    with pytest.raises(HarnessIOError, match="sha256 must match"):
        CandidatePolicyRecord(
            generated_at_utc="2026-04-25T00:00:00Z",
            source_training_dataset_report=dataset_artifact,
            trainer=_trainer(),
            policy_template=_policy_template(),
            eval=_eval_spec(),
            source_next_experiment_plan=ArtifactRef(
                kind="next_experiment_plan",
                path=plan_artifact.path,
                sha256="0" * 64,
                provenance=plan_artifact.provenance,
            ),
            source_plan_candidate_id="next-candidate-a",
        )

    with pytest.raises(HarnessIOError, match="must match a next experiment plan candidate"):
        CandidatePolicyRecord(
            generated_at_utc="2026-04-25T00:00:00Z",
            source_training_dataset_report=dataset_artifact,
            trainer=_trainer(),
            policy_template=_policy_template(),
            eval=_eval_spec(),
            source_next_experiment_plan=plan_artifact,
            source_plan_candidate_id="missing-candidate",
        )


def test_candidate_policy_record_rejects_trainer_backend_mismatch(tmp_path: Path) -> None:
    with pytest.raises(HarnessIOError, match="trainer.backend_kind"):
        _candidate_record(
            tmp_path,
            trainer=TrainerInvocation(
                trainer_id="replay-trainer-v1",
                trainer_name="Replay trainer",
                backend_kind=BackendKind.replay_servo,
                command=(sys.executable, "-c", "print('train')"),
            ),
        )


def test_candidate_policy_record_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = _candidate_record(tmp_path).to_dict()
    payload["mystery"] = True

    with pytest.raises(HarnessIOError, match="unknown fields"):
        CandidatePolicyRecord.from_dict(payload)


def test_candidate_policy_record_rejects_non_deterministic_id(tmp_path: Path) -> None:
    payload = _candidate_record(tmp_path).to_dict()
    payload["candidate_id"] = "cand_wrong"

    with pytest.raises(HarnessIOError, match="deterministic"):
        CandidatePolicyRecord.from_dict(payload)


def _candidate_record(
    tmp_path: Path,
    *,
    generated_at_utc: str = "2026-04-25T00:00:00Z",
    trainer: TrainerInvocation | None = None,
    safety: CandidateSafetyContract | None = None,
    notes: tuple[str, ...] = (),
) -> CandidatePolicyRecord:
    return CandidatePolicyRecord(
        generated_at_utc=generated_at_utc,
        source_training_dataset_report=_training_dataset_artifact(tmp_path),
        trainer=_trainer() if trainer is None else trainer,
        policy_template=_policy_template(),
        checkpoint=CandidateCheckpointExpectation(output_name="policy.ckpt"),
        eval=_eval_spec(),
        safety=CandidateSafetyContract() if safety is None else safety,
        notes=notes,
    )


def _eval_spec() -> CandidateEvalSpec:
    return CandidateEvalSpec(
        mode=CandidateEvalMode.local_live_eval,
        gate_id="trained_policy_live_eval",
        planner_mode="lewm_mpc",
        min_improvement=1.0,
    )


def _trainer() -> TrainerInvocation:
    return TrainerInvocation(
        trainer_id="lewm-trainer-v1",
        trainer_name="LEWM trainer",
        backend_kind=BackendKind.lewm_world_model,
        command=(sys.executable, "-c", "print('train')"),
        config={"learning_rate": 0.0001, "max_epochs": 1},
        seed=11,
        offline_only=True,
        runtime_allowed=False,
        uses_online_language_model_control=False,
    )


def _policy_template() -> CandidatePolicyTemplate:
    return CandidatePolicyTemplate(
        backend_kind=BackendKind.lewm_world_model,
        name="lewm-candidate",
        training_sources=(TrainingSourceKind.official_demo,),
        simulator_sources=(SimulatorKind.offline_replay,),
        legal_observation_contract="official aic_model observations only",
        description="Candidate produced by the outer-loop contract.",
        config={"planner_mode": "lewm_mpc"},
        deterministic=True,
    )


def _training_dataset_artifact(
    tmp_path: Path,
    *,
    include_privileged_signal: bool = False,
) -> ArtifactRef:
    signals = [
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
        )
    ]
    if include_privileged_signal:
        signals.append(
            TrainingSignal(
                signal_id="signal-safety-1",
                kind=TrainingSignalKind.safety_guard_avoidance,
                target="safety.guard",
                weight=0.5,
                source="episode_trace",
                source_event_indices=(2,),
                extraction_method="episode_trace.v1.safety_guard",
                leakage_class=LeakageClass.privileged_training_signal,
                evidence={"payload": {"limit": 18}},
            )
        )
    signal_report = TrainingSignalReport(
        run_id="source-run-a",
        generated_at_utc="2026-04-25T00:00:00Z",
        source_trace=ArtifactRef(
            kind="episode_trace",
            uri="memory://pytest/episode_trace.json",
            sha256="a" * 64,
            provenance={"producer": "pytest", "derivation": "fixture", "run_id": "source-run-a"},
        ),
        signals=tuple(signals),
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


def _next_experiment_plan_artifact(tmp_path: Path) -> ArtifactRef:
    manifest = RunManifest(
        run_id="source-run-a",
        status="planned",
        backend=PolicyBackendSpec(
            backend_kind=BackendKind.replay_servo,
            name="offline plan backend",
            runtime_role=RuntimeRole.offline_planner,
            training_sources=(TrainingSourceKind.none,),
            simulator_sources=(SimulatorKind.offline_replay,),
            runtime_allowed=False,
            leakage_class=LeakageClass.post_hoc_label,
        ),
        created_at_utc="2026-04-25T00:00:00Z",
        updated_at_utc="2026-04-25T00:00:00Z",
    )
    next_experiment = NextExperimentReport(
        run_id=manifest.run_id,
        gate_id="trained_policy_live_eval",
        generated_at_utc="2026-04-25T00:00:01Z",
        decision=NextExperimentDecision.iterate,
        candidates=(
            NextExperimentCandidate(
                candidate_id="next-candidate-a",
                kind=NextExperimentCandidateKind.controller_development,
                priority=NextExperimentPriority.high,
                action=NextExperimentAction.build,
                title="Build LEWM candidate",
                rationale="Create an executable trained-policy candidate.",
                evidence_terms=("official.score.total",),
                acceptance_checks=("Produce a policy checkpoint.",),
            ),
        ),
    )
    manifest_path = tmp_path / "source_run_manifest.json"
    next_experiment_path = tmp_path / "source_next_experiment.json"
    write_json(manifest_path, manifest.to_dict(), overwrite=True)
    write_json(next_experiment_path, next_experiment.to_dict(), overwrite=True)
    source_manifest = ArtifactRef(
        kind="run_manifest",
        path=str(manifest_path),
        sha256=sha256_file(manifest_path),
        provenance={"producer": "pytest", "run_id": manifest.run_id},
    )
    source_next_experiment = ArtifactRef(
        kind="next_experiment_report",
        path=str(next_experiment_path),
        sha256=sha256_file(next_experiment_path),
        provenance={
            "producer": "pytest",
            "derivation": "fixture",
            "run_id": next_experiment.run_id,
            "gate_id": next_experiment.gate_id,
        },
    )
    plan = NextExperimentPlan(
        run_id=manifest.run_id,
        gate_id=next_experiment.gate_id,
        generated_at_utc="2026-04-25T00:00:02Z",
        source_next_experiment=source_next_experiment,
        source_manifest=source_manifest,
        candidates=(
            NextExperimentPlanCandidate(
                plan_id="nxplan-test-a",
                candidate_id="next-candidate-a",
                candidate_kind=NextExperimentCandidateKind.controller_development,
                candidate_action=NextExperimentAction.build,
                priority=NextExperimentPriority.high,
                status=NextExperimentPlanStatus.requires_implementation,
                execution_mode=NextExperimentExecutionMode.policy_development,
                title="Build LEWM candidate",
                rationale="Create an executable trained-policy candidate.",
                evidence_terms=("official.score.total",),
                acceptance_checks=("Produce a policy checkpoint.",),
            ),
        ),
    )
    plan_path = tmp_path / "source_next_experiment_plan.json"
    write_json(plan_path, plan.to_dict(), overwrite=True)
    return ArtifactRef(
        kind="next_experiment_plan",
        path=str(plan_path),
        sha256=sha256_file(plan_path),
        provenance={"producer": "pytest", "derivation": "fixture"},
    )
