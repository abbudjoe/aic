"""Derive executable offline plans from typed next-experiment reports."""

from __future__ import annotations

from typing import Any, Mapping

from aic_signal_harness.artifacts import HarnessIOError, read_json
from aic_signal_harness.manifest import RunManifest
from aic_signal_harness.next_experiment import (
    NextExperimentAction,
    NextExperimentCandidate,
    NextExperimentCandidateKind,
    NextExperimentReport,
)
from aic_signal_harness.next_experiment_plan import (
    NextExperimentExecutionMode,
    NextExperimentPlan,
    NextExperimentPlanCandidate,
    NextExperimentPlanStatus,
    next_experiment_plan_id,
)
from aic_signal_harness.schemas import (
    ArtifactRef,
    BackendKind,
    ExperimentSpec,
    LeakageClass,
    PolicyBackendSpec,
    RuntimeRole,
    SchemaValidationError,
    SimulatorKind,
    TrainingSourceKind,
)


def derive_next_experiment_plan(
    next_experiment: NextExperimentReport | Mapping[str, Any],
    *,
    manifest: RunManifest | Mapping[str, Any],
    source_next_experiment: ArtifactRef,
    source_manifest: ArtifactRef,
    generated_at_utc: str | None = None,
) -> NextExperimentPlan:
    """Convert ``next_experiment.json`` into deterministic offline candidate plans."""

    typed_report = (
        next_experiment
        if isinstance(next_experiment, NextExperimentReport)
        else NextExperimentReport.from_dict(next_experiment)
    )
    typed_manifest = manifest if isinstance(manifest, RunManifest) else RunManifest.from_dict(manifest)
    _validate_inputs(
        next_experiment=typed_report,
        manifest=typed_manifest,
        source_next_experiment=source_next_experiment,
        source_manifest=source_manifest,
    )
    generated_at = typed_report.generated_at_utc if generated_at_utc is None else generated_at_utc
    candidates = tuple(
        _plan_candidate(
            candidate,
            next_experiment=typed_report,
            manifest=typed_manifest,
            source_next_experiment=source_next_experiment,
            source_manifest=source_manifest,
            generated_at_utc=generated_at,
        )
        for candidate in typed_report.candidates
    )
    return NextExperimentPlan(
        run_id=typed_report.run_id,
        gate_id=typed_report.gate_id,
        generated_at_utc=generated_at,
        source_next_experiment=source_next_experiment,
        source_manifest=source_manifest,
        candidates=candidates,
        notes=(
            "Derived plans are offline records for a future runner; this reducer does not launch jobs.",
        ),
    )


def _validate_inputs(
    *,
    next_experiment: NextExperimentReport,
    manifest: RunManifest,
    source_next_experiment: ArtifactRef,
    source_manifest: ArtifactRef,
) -> None:
    errors: list[str] = []
    if next_experiment.run_id != manifest.run_id:
        errors.append("next_experiment.run_id does not match manifest.run_id")
    try:
        source_payload = read_json(_required_local_artifact_path(source_next_experiment, "source_next_experiment"))
        source_report = NextExperimentReport.from_dict(source_payload)
        if source_report != next_experiment:
            errors.append("source_next_experiment does not match supplied next_experiment")
    except HarnessIOError as exc:
        errors.append(str(exc))
    try:
        manifest_payload = read_json(_required_local_artifact_path(source_manifest, "source_manifest"))
        source_run_manifest = RunManifest.from_dict(manifest_payload)
        if source_run_manifest != manifest:
            errors.append("source_manifest does not match supplied manifest")
    except (HarnessIOError, SchemaValidationError) as exc:
        errors.append(str(exc))
    if errors:
        raise HarnessIOError("; ".join(errors))


def _required_local_artifact_path(artifact: ArtifactRef, field_name: str):
    from aic_signal_harness.artifacts import local_artifact_path, sha256_file

    artifact_path = local_artifact_path(path=artifact.path, uri=artifact.uri, field_name=field_name)
    if artifact_path is None:
        raise HarnessIOError(f"{field_name} must be local byte-verifiable")
    if not artifact_path.exists():
        raise HarnessIOError(f"{field_name}.path must exist")
    if not artifact_path.is_file():
        raise HarnessIOError(f"{field_name}.path must be a file")
    if artifact.sha256 is None:
        raise HarnessIOError(f"{field_name}.sha256 must be set")
    if sha256_file(artifact_path) != artifact.sha256:
        raise HarnessIOError(f"{field_name}.sha256 must match path")
    return artifact_path


def _plan_candidate(
    candidate: NextExperimentCandidate,
    *,
    next_experiment: NextExperimentReport,
    manifest: RunManifest,
    source_next_experiment: ArtifactRef,
    source_manifest: ArtifactRef,
    generated_at_utc: str,
) -> NextExperimentPlanCandidate:
    status = _candidate_status(candidate)
    execution_mode = _candidate_execution_mode(candidate)
    plan_id = next_experiment_plan_id(
        next_experiment.run_id,
        next_experiment.gate_id,
        candidate.candidate_id,
        source_next_experiment_sha256=source_next_experiment.sha256,
        source_manifest_sha256=source_manifest.sha256,
    )
    experiment = (
        _experiment_spec(
            candidate,
            next_experiment=next_experiment,
            manifest=manifest,
            source_next_experiment=source_next_experiment,
            source_manifest=source_manifest,
            generated_at_utc=generated_at_utc,
            plan_id=plan_id,
            execution_mode=execution_mode,
        )
        if status is NextExperimentPlanStatus.launchable
        else None
    )
    return NextExperimentPlanCandidate(
        plan_id=plan_id,
        candidate_id=candidate.candidate_id,
        candidate_kind=candidate.kind,
        candidate_action=candidate.action,
        priority=candidate.priority,
        status=status,
        execution_mode=execution_mode,
        title=candidate.title,
        rationale=candidate.rationale,
        evidence_labels=candidate.evidence_labels,
        evidence_terms=candidate.evidence_terms,
        evidence_manifest_fields=candidate.evidence_manifest_fields,
        acceptance_checks=candidate.acceptance_checks,
        blocked_by_labels=candidate.blocked_by_labels,
        experiment=experiment,
        launchable=status is NextExperimentPlanStatus.launchable,
        notes=(_candidate_note(status),),
    )


def _candidate_status(candidate: NextExperimentCandidate) -> NextExperimentPlanStatus:
    if candidate.action is NextExperimentAction.block:
        return NextExperimentPlanStatus.blocked
    if candidate.action in (NextExperimentAction.reject, NextExperimentAction.keep):
        return NextExperimentPlanStatus.no_launch
    if candidate.action is NextExperimentAction.build:
        return NextExperimentPlanStatus.requires_implementation
    return NextExperimentPlanStatus.launchable


def _candidate_execution_mode(candidate: NextExperimentCandidate) -> NextExperimentExecutionMode:
    if candidate.action is NextExperimentAction.block:
        return NextExperimentExecutionMode.evidence_repair
    if candidate.action is NextExperimentAction.reject:
        return NextExperimentExecutionMode.rejection_record
    if candidate.action is NextExperimentAction.keep:
        return NextExperimentExecutionMode.safety_contract
    if candidate.action is NextExperimentAction.build:
        return NextExperimentExecutionMode.policy_development
    if candidate.kind is NextExperimentCandidateKind.offline_acceptance_gate:
        return NextExperimentExecutionMode.offline_gate
    if candidate.kind is NextExperimentCandidateKind.failure_analysis:
        return NextExperimentExecutionMode.failure_analysis
    return NextExperimentExecutionMode.promotion_review


def _experiment_spec(
    candidate: NextExperimentCandidate,
    *,
    next_experiment: NextExperimentReport,
    manifest: RunManifest,
    source_next_experiment: ArtifactRef,
    source_manifest: ArtifactRef,
    generated_at_utc: str,
    plan_id: str,
    execution_mode: NextExperimentExecutionMode,
) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=f"{next_experiment.gate_id}.{candidate.candidate_id}",
        hypothesis=candidate.rationale,
        backend=PolicyBackendSpec(
            backend_kind=BackendKind.custom,
            name=f"next-experiment:{candidate.candidate_id}",
            runtime_role=RuntimeRole.offline_planner,
            training_sources=(TrainingSourceKind.none,),
            simulator_sources=(SimulatorKind.offline_replay,),
            runtime_allowed=False,
            leakage_class=LeakageClass.post_hoc_label,
            description=candidate.title,
            config={
                "candidate_id": candidate.candidate_id,
                "candidate_kind": candidate.kind.value,
                "candidate_action": candidate.action.value,
                "execution_mode": execution_mode.value,
                "gate_id": next_experiment.gate_id,
                "source_run_id": manifest.run_id,
            },
            provenance={
                "producer": "aic_signal_harness.reducers.next_experiment_plan",
                "plan_id": plan_id,
                "source_next_experiment_sha256": source_next_experiment.sha256,
                "source_manifest_sha256": source_manifest.sha256,
            },
        ),
        expected_artifacts=(
            ArtifactRef(
                kind=_expected_artifact_kind(execution_mode),
                uri=f"memory://aic_signal_harness/next_experiment_plan/{plan_id}/expected_output",
                provenance={
                    "producer": "aic_signal_harness.reducers.next_experiment_plan",
                    "run_id": next_experiment.run_id,
                    "candidate_id": candidate.candidate_id,
                },
            ),
        ),
        created_at_utc=generated_at_utc,
        tags=(
            next_experiment.gate_id,
            candidate.kind.value,
            candidate.action.value,
            execution_mode.value,
        ),
    )


def _expected_artifact_kind(execution_mode: NextExperimentExecutionMode) -> str:
    if execution_mode is NextExperimentExecutionMode.offline_gate:
        return "offline_acceptance_gate_report"
    if execution_mode is NextExperimentExecutionMode.failure_analysis:
        return "failure_analysis_report"
    return "next_experiment_review_report"


def _candidate_note(status: NextExperimentPlanStatus) -> str:
    if status is NextExperimentPlanStatus.launchable:
        return "Launchable by a future offline runner only; autonomous launch is disabled."
    if status is NextExperimentPlanStatus.requires_implementation:
        return "Requires code or policy implementation before a runner may launch it."
    if status is NextExperimentPlanStatus.blocked:
        return "Blocked until required evidence is repaired."
    return "Recorded as a no-launch decision."
