from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aic_signal_harness import (
    ArtifactRef,
    BackendKind,
    CandidateEvalMode,
    CandidateGenerationConfig,
    CandidatePolicyTemplate,
    HarnessIOError,
    LeakageClass,
    NEXT_EXPERIMENT_PLAN_KIND,
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
    RunStatus,
    RuntimeRole,
    SimulatorKind,
    TrainerInvocation,
    TrainingSignal,
    TrainingSignalKind,
    TrainingSignalReport,
    TrainingSourceKind,
    derive_candidate_policy_registry_from_plan,
    derive_training_dataset_report,
    read_candidate_policy_registry_artifact,
    read_next_experiment_plan_artifact,
    sha256_file,
    write_candidate_policy_registry,
    write_json,
)


def test_candidate_generation_derives_selected_registry_from_plan(tmp_path: Path) -> None:
    plan_artifact = _next_experiment_plan_artifact(tmp_path)
    dataset_artifact = _training_dataset_artifact(tmp_path)

    registry = derive_candidate_policy_registry_from_plan(
        source_next_experiment_plan=plan_artifact,
        source_training_dataset_report=dataset_artifact,
        trainer=_trainer(),
        policy_template=_policy_template(),
        config=CandidateGenerationConfig(
            registry_id="registry-generated-a",
            generated_at_utc="2026-04-25T02:00:00Z",
            selected_plan_candidate_id="build_lewm_candidate",
            eval_mode=CandidateEvalMode.local_live_eval,
            min_improvement=1.0,
        ),
    )
    artifact = write_candidate_policy_registry(tmp_path / "candidate_registry.json", registry)
    reread = read_candidate_policy_registry_artifact(artifact)

    assert reread == registry
    assert read_next_experiment_plan_artifact(plan_artifact).gate_id == "trained_policy_pre_eval"
    assert plan_artifact.kind == NEXT_EXPERIMENT_PLAN_KIND
    assert registry.selected_candidate is not None
    assert registry.selected_candidate.source_next_experiment_plan == plan_artifact
    assert registry.selected_candidate.source_plan_candidate_id == "build_lewm_candidate"
    assert registry.selected_candidate.source_training_dataset_report == dataset_artifact
    assert registry.selected_candidate.eval.gate_id == "trained_policy_pre_eval"
    assert registry.selected_candidate.eval.planner_mode == "lewm_mpc"
    assert registry.selected_candidate.safety.autonomous_launch_allowed is False


def test_candidate_generation_rejects_non_policy_development_plan_candidates(
    tmp_path: Path,
) -> None:
    dataset_artifact = _training_dataset_artifact(tmp_path)
    no_launch_artifact = _next_experiment_plan_artifact(
        tmp_path,
        status=NextExperimentPlanStatus.no_launch,
    )

    with pytest.raises(HarnessIOError, match="requires_implementation"):
        derive_candidate_policy_registry_from_plan(
            source_next_experiment_plan=no_launch_artifact,
            source_training_dataset_report=dataset_artifact,
            trainer=_trainer(),
            policy_template=_policy_template(),
            config=_generation_config(),
        )

    rejection_record_artifact = _next_experiment_plan_artifact(
        tmp_path,
        execution_mode=NextExperimentExecutionMode.rejection_record,
    )

    with pytest.raises(HarnessIOError, match="policy_development"):
        derive_candidate_policy_registry_from_plan(
            source_next_experiment_plan=rejection_record_artifact,
            source_training_dataset_report=dataset_artifact,
            trainer=_trainer(),
            policy_template=_policy_template(),
            config=_generation_config(),
        )


def test_candidate_generation_rejects_missing_selected_plan_candidate(tmp_path: Path) -> None:
    with pytest.raises(HarnessIOError, match="selected_plan_candidate_id"):
        derive_candidate_policy_registry_from_plan(
            source_next_experiment_plan=_next_experiment_plan_artifact(tmp_path),
            source_training_dataset_report=_training_dataset_artifact(tmp_path),
            trainer=_trainer(),
            policy_template=_policy_template(),
            config=CandidateGenerationConfig(
                registry_id="registry-generated-a",
                generated_at_utc="2026-04-25T02:00:00Z",
                selected_plan_candidate_id="missing-candidate",
            ),
        )


def test_candidate_generation_rejects_unverifiable_plan_artifact(tmp_path: Path) -> None:
    plan_artifact = _next_experiment_plan_artifact(tmp_path)

    with pytest.raises(HarnessIOError, match="sha256 must match"):
        read_next_experiment_plan_artifact(
            ArtifactRef(
                kind=plan_artifact.kind,
                path=plan_artifact.path,
                sha256="0" * 64,
                provenance=plan_artifact.provenance,
            )
        )

    with pytest.raises(HarnessIOError, match="local byte-verifiable"):
        derive_candidate_policy_registry_from_plan(
            source_next_experiment_plan=ArtifactRef(
                kind=NEXT_EXPERIMENT_PLAN_KIND,
                uri="gs://aic-bucket/next_experiment_plan.json",
                sha256=plan_artifact.sha256,
                provenance=plan_artifact.provenance,
            ),
            source_training_dataset_report=_training_dataset_artifact(tmp_path),
            trainer=_trainer(),
            policy_template=_policy_template(),
            config=_generation_config(),
        )

    with pytest.raises(HarnessIOError, match="provenance must set producer"):
        read_next_experiment_plan_artifact(
            ArtifactRef(
                kind=plan_artifact.kind,
                path=plan_artifact.path,
                sha256=plan_artifact.sha256,
                provenance={"derivation": "derive_next_experiment_plan"},
            )
        )

    with pytest.raises(HarnessIOError, match="provenance must set derivation"):
        read_next_experiment_plan_artifact(
            ArtifactRef(
                kind=plan_artifact.kind,
                path=plan_artifact.path,
                sha256=plan_artifact.sha256,
                provenance={"producer": "pytest"},
            )
        )


def test_candidate_generation_rejects_policy_template_without_planner_mode(
    tmp_path: Path,
) -> None:
    template = CandidatePolicyTemplate(
        backend_kind=BackendKind.lewm_world_model,
        name="lewm-candidate",
        training_sources=(TrainingSourceKind.official_demo,),
        simulator_sources=(SimulatorKind.offline_replay,),
        legal_observation_contract="official aic_model observations only",
        config={},
    )

    with pytest.raises(HarnessIOError, match="planner_mode"):
        derive_candidate_policy_registry_from_plan(
            source_next_experiment_plan=_next_experiment_plan_artifact(tmp_path),
            source_training_dataset_report=_training_dataset_artifact(tmp_path),
            trainer=_trainer(),
            policy_template=template,
            config=_generation_config(),
        )


def _generation_config() -> CandidateGenerationConfig:
    return CandidateGenerationConfig(
        registry_id="registry-generated-a",
        generated_at_utc="2026-04-25T02:00:00Z",
        selected_plan_candidate_id="build_lewm_candidate",
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
        description="Generated LEWM candidate.",
        config={"planner_mode": "lewm_mpc"},
        deterministic=True,
    )


def _next_experiment_plan_artifact(
    tmp_path: Path,
    *,
    status: NextExperimentPlanStatus = NextExperimentPlanStatus.requires_implementation,
    execution_mode: NextExperimentExecutionMode = NextExperimentExecutionMode.policy_development,
) -> ArtifactRef:
    manifest = RunManifest(
        run_id="source-run-a",
        status=RunStatus.planned,
        backend=PolicyBackendSpec(
            backend_kind=BackendKind.replay_servo,
            name="offline source backend",
            runtime_role=RuntimeRole.offline_planner,
            training_sources=(TrainingSourceKind.none,),
            simulator_sources=(SimulatorKind.offline_replay,),
            runtime_allowed=False,
            leakage_class=LeakageClass.post_hoc_label,
        ),
        created_at_utc="2026-04-25T00:00:00Z",
        updated_at_utc="2026-04-25T00:00:00Z",
    )
    report = NextExperimentReport(
        run_id=manifest.run_id,
        gate_id="trained_policy_pre_eval",
        generated_at_utc="2026-04-25T00:00:01Z",
        decision=NextExperimentDecision.iterate,
        candidates=(
            NextExperimentCandidate(
                candidate_id="build_lewm_candidate",
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
    report_path = tmp_path / "next_experiment.json"
    write_json(manifest_path, manifest.to_dict(), overwrite=True)
    write_json(report_path, report.to_dict(), overwrite=True)
    source_manifest = ArtifactRef(
        kind="run_manifest",
        path=str(manifest_path),
        sha256=sha256_file(manifest_path),
        provenance={"producer": "pytest", "run_id": manifest.run_id},
    )
    source_next_experiment = ArtifactRef(
        kind="next_experiment_report",
        path=str(report_path),
        sha256=sha256_file(report_path),
        provenance={
            "producer": "pytest",
            "derivation": "derive_next_experiment_report",
            "run_id": report.run_id,
            "gate_id": report.gate_id,
        },
    )
    plan = NextExperimentPlan(
        run_id=manifest.run_id,
        gate_id=report.gate_id,
        generated_at_utc="2026-04-25T00:00:02Z",
        source_next_experiment=source_next_experiment,
        source_manifest=source_manifest,
        candidates=(
            NextExperimentPlanCandidate(
                plan_id="nxplan-build-lewm",
                candidate_id="build_lewm_candidate",
                candidate_kind=NextExperimentCandidateKind.controller_development,
                candidate_action=NextExperimentAction.build,
                priority=NextExperimentPriority.high,
                status=status,
                execution_mode=execution_mode,
                title="Build LEWM candidate",
                rationale="Create an executable trained-policy candidate.",
                evidence_terms=("official.score.total",),
                acceptance_checks=("Produce a policy checkpoint.",),
                launchable=False,
            ),
        ),
    )
    plan_path = tmp_path / f"next_experiment_plan_{status.value}.json"
    write_json(plan_path, plan.to_dict(), overwrite=True)
    return ArtifactRef(
        kind=NEXT_EXPERIMENT_PLAN_KIND,
        path=str(plan_path),
        sha256=sha256_file(plan_path),
        provenance={
            "producer": "pytest",
            "derivation": "derive_next_experiment_plan",
            "run_id": plan.run_id,
            "gate_id": plan.gate_id,
        },
    )


def _training_dataset_artifact(tmp_path: Path) -> ArtifactRef:
    signal_report = TrainingSignalReport(
        run_id="source-run-a",
        generated_at_utc="2026-04-25T00:00:00Z",
        source_trace=ArtifactRef(
            kind="episode_trace",
            uri="memory://pytest/episode_trace.json",
            sha256="a" * 64,
            provenance={"producer": "pytest", "derivation": "fixture", "run_id": "source-run-a"},
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
