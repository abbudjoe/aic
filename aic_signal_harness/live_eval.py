"""Post-eval finalization for live AIC runs.

This module runs after the official evaluator exits. It observes official
artifacts, reduces them into neutral harness contracts, and writes decision
evidence. It does not participate in the live policy control path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from aic_signal_harness.artifacts import HarnessIOError, sha256_file, write_json
from aic_signal_harness.ledger import append_ledger_entry, read_ledger_entries
from aic_signal_harness.manifest import RunManifest, RunStatus
from aic_signal_harness.promotion import (
    PromotionDecision,
    read_promotion_decision,
    write_baseline_decision,
    write_promotion_decision,
)
from aic_signal_harness.reducers import (
    PolicyTraceReduction,
    attach_scoring_yaml_reduction,
    build_ledger_entry,
    derive_episode_trace,
    derive_next_experiment_plan,
    derive_next_experiment_report,
    derive_reward_failure_reports,
    derive_training_signal_report,
    promote_ledger_entry,
    reduce_policy_trace_jsonl,
    reduce_scoring_yaml,
)
from aic_signal_harness.schemas import (
    ArtifactRef,
    BackendKind,
    LeakageClass,
    PolicyBackendSpec,
    RuntimeBoundaryProof,
    RuntimeRole,
    SimulatorKind,
    TrainingSourceKind,
    utc_now_iso,
)


_DEFAULT_LEGAL_OBSERVATION_CONTRACT = (
    "official aic_model task, observation, and action callbacks only"
)
_RUNTIME_ENV_KEYS = (
    "AIC_DOCKER_GPUS",
    "AIC_EVAL_IMAGE",
    "AIC_EVAL_TIMEOUT_SEC",
    "AIC_EVAL_USE_LOCAL_LAUNCH",
    "AIC_GZ_VERBOSITY_LEVEL",
    "AIC_LEWM_ANGULAR_VEL_LIMIT",
    "AIC_LEWM_COMMAND_FRAME",
    "AIC_LEWM_CONTROL_HZ",
    "AIC_LEWM_CHECKPOINT",
    "AIC_LEWM_DEVICE",
    "AIC_LEWM_FINAL_SERVO_DURATION_SEC",
    "AIC_LEWM_FINAL_SERVO_ENABLED",
    "AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE",
    "AIC_LEWM_FINAL_SERVO_FORCE_GUARD_N",
    "AIC_LEWM_LINEAR_VEL_LIMIT",
    "AIC_LEWM_MAX_RUNTIME_SEC",
    "AIC_LEWM_NUM_ACTION_CANDIDATES",
    "AIC_LEWM_PLANNER_MODE",
    "AIC_LEWM_PLANNING_HORIZON",
    "AIC_LEWM_POLICY_TRACE_TRUST_TASK_OFFICIAL_TRIAL_ID",
    "AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_ID_MAP",
    "AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_IDS",
    "AIC_LEWM_GOAL_DATASET",
    "AIC_LEWM_REPLAY_ACTION_GAIN",
    "AIC_LEWM_REPLAY_DATASET",
    "AIC_LEWM_REPLAY_HZ",
    "AIC_LEWM_REPLAY_TIME_SCALE",
    "AIC_LEWM_REQUIRE_CHECKPOINT",
    "AIC_LEWM_SC_FINAL_SERVO_ANGULAR",
    "AIC_LEWM_SC_FINAL_SERVO_LINEAR",
    "AIC_LEWM_SC_REPLAY_STOP_SEC",
    "AIC_LEWM_SFP_FINAL_SERVO_ANGULAR",
    "AIC_LEWM_SFP_FINAL_SERVO_LINEAR",
)
_CONFIG_ENV_KEYS = (
    "AIC_LEWM_ANGULAR_VEL_LIMIT",
    "AIC_LEWM_COMMAND_FRAME",
    "AIC_LEWM_CONTROL_HZ",
    "AIC_LEWM_DEVICE",
    "AIC_LEWM_FINAL_SERVO_DURATION_SEC",
    "AIC_LEWM_FINAL_SERVO_ENABLED",
    "AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE",
    "AIC_LEWM_FINAL_SERVO_FORCE_GUARD_N",
    "AIC_LEWM_LINEAR_VEL_LIMIT",
    "AIC_LEWM_MAX_RUNTIME_SEC",
    "AIC_LEWM_NUM_ACTION_CANDIDATES",
    "AIC_LEWM_PLANNER_MODE",
    "AIC_LEWM_PLANNING_HORIZON",
    "AIC_LEWM_REPLAY_ACTION_GAIN",
    "AIC_LEWM_REPLAY_HZ",
    "AIC_LEWM_REPLAY_TIME_SCALE",
    "AIC_LEWM_SC_FINAL_SERVO_ANGULAR",
    "AIC_LEWM_SC_FINAL_SERVO_LINEAR",
    "AIC_LEWM_SC_REPLAY_STOP_SEC",
    "AIC_LEWM_SFP_FINAL_SERVO_ANGULAR",
    "AIC_LEWM_SFP_FINAL_SERVO_LINEAR",
)


@dataclass(frozen=True)
class LiveEvalFinalization:
    """Paths and typed outputs produced by one live eval finalization."""

    manifest: RunManifest
    manifest_artifact: ArtifactRef
    ledger_entry_path: Path
    manifest_path: Path
    score_report_path: Path
    scoring_artifact_path: Path
    reward_report_path: Path
    failure_report_path: Path
    summary_path: Path
    policy_trace_report_path: Path | None = None
    policy_trace_artifact_path: Path | None = None
    episode_trace_path: Path | None = None
    training_signal_report_path: Path | None = None
    promotion_path: Path | None = None
    next_experiment_path: Path | None = None
    next_experiment_plan_path: Path | None = None
    ledger_path: Path | None = None

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.manifest.run_id,
            "score_total": None if self.manifest.score is None else self.manifest.score.total,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_artifact.sha256,
            "score_report_path": str(self.score_report_path),
            "scoring_artifact_path": str(self.scoring_artifact_path),
            "policy_trace_report_path": _optional_path(self.policy_trace_report_path),
            "policy_trace_artifact_path": _optional_path(self.policy_trace_artifact_path),
            "episode_trace_path": _optional_path(self.episode_trace_path),
            "training_signal_report_path": _optional_path(self.training_signal_report_path),
            "ledger_entry_path": str(self.ledger_entry_path),
            "ledger_path": _optional_path(self.ledger_path),
            "promotion_path": _optional_path(self.promotion_path),
            "reward_report_path": str(self.reward_report_path),
            "failure_report_path": str(self.failure_report_path),
            "next_experiment_path": _optional_path(self.next_experiment_path),
            "next_experiment_plan_path": _optional_path(self.next_experiment_plan_path),
        }


def finalize_live_eval_run(
    *,
    run_id: str,
    result_root: str | Path,
    harness_root: str | Path,
    scoring_yaml: str | Path,
    policy_trace: str | Path | None = None,
    ledger_path: str | Path | None = None,
    baseline_path: str | Path | None = None,
    update_baseline_path: str | Path | None = None,
    bootstrap_promotion: bool = False,
    min_improvement: float = 0.0,
    eligible_for_submission: bool = False,
    gate_id: str = "live_eval",
    experiment_id: str | None = None,
    hypothesis: str | None = None,
    model_image: str | None = None,
    model_image_id: str | None = None,
    policy_checkpoint: str | Path | None = None,
    backend: PolicyBackendSpec | Mapping[str, Any] | None = None,
    extra_manifest_artifacts: tuple[ArtifactRef | Mapping[str, Any], ...] = (),
    backend_kind: str | BackendKind | None = None,
    planner_mode: str | None = None,
    runtime_env: Mapping[str, str] | None = None,
    overwrite: bool = False,
    append_ledger: bool = True,
    write_next_experiment: bool = True,
    generated_at_utc: str | None = None,
) -> LiveEvalFinalization:
    """Reduce official live eval artifacts into the neutral harness record."""

    run_id = _require_text(run_id, "run_id")
    result_root = _resolve_existing_dir(result_root, "result_root")
    scoring_yaml = _resolve_bound_scoring_yaml(scoring_yaml, result_root)
    harness_root = Path(harness_root).expanduser().resolve(strict=False)
    generated_at = utc_now_iso() if generated_at_utc is None else generated_at_utc
    env = _runtime_env_from_mapping(runtime_env)
    baseline = _load_baseline_decision(baseline_path=baseline_path)
    _validate_declared_policy_trace(policy_trace)

    scoring_reduction = reduce_scoring_yaml(
        scoring_yaml,
        parsed_at_utc=generated_at,
        provenance={"producer": "aic_signal_harness.live_eval", "run_id": run_id},
    )
    policy_trace_reduction = _reduce_optional_policy_trace(
        policy_trace=policy_trace,
        run_id=run_id,
        reduced_at_utc=generated_at,
    )
    backend_spec = (
        _typed_backend_override(backend)
        if backend is not None
        else _backend_spec(
            run_id=run_id,
            model_image=model_image,
            model_image_id=model_image_id,
            policy_checkpoint=policy_checkpoint,
            backend_kind=backend_kind,
            planner_mode=planner_mode,
            runtime_env=env,
        )
    )
    extra_artifacts = _typed_extra_manifest_artifacts(extra_manifest_artifacts)
    score_report_path = harness_root / "score_report.json"
    scoring_artifact_path = harness_root / "scoring_yaml_artifact.json"
    policy_trace_report_path = (
        None if policy_trace_reduction is None else harness_root / "policy_trace_report.json"
    )
    policy_trace_artifact_path = (
        None if policy_trace_reduction is None else harness_root / "policy_trace_artifact.json"
    )
    episode_trace_path = (
        None if policy_trace_reduction is None else harness_root / "episode_trace.json"
    )
    training_signal_report_path = (
        None if policy_trace_reduction is None else harness_root / "training_signal_report.json"
    )
    manifest_path = harness_root / "run_manifest.json"
    promotion_path = (
        harness_root / "promotion_report.json"
        if baseline is not None or bootstrap_promotion
        else None
    )
    reward_report_path = harness_root / "reward_report.json"
    failure_report_path = harness_root / "failure_report.json"
    next_experiment_path = (
        harness_root / "next_experiment.json" if write_next_experiment else None
    )
    next_experiment_plan_path = (
        harness_root / "next_experiment_plan.json" if write_next_experiment else None
    )
    ledger_entry_path = harness_root / "ledger_entry.json"
    summary_path = harness_root / "live_eval_summary.json"
    typed_ledger_path = None if ledger_path is None else Path(ledger_path).expanduser()
    base_artifacts = (
        (scoring_reduction.artifact, *extra_artifacts)
        + (
            ()
            if policy_trace_reduction is None
            else (policy_trace_reduction.artifact,)
        )
    )
    base_manifest = RunManifest(
        run_id=run_id,
        status=RunStatus.completed,
        backend=backend_spec,
        created_at_utc=generated_at,
        updated_at_utc=generated_at,
        artifacts=base_artifacts,
        score=scoring_reduction.score,
        experiment_id=experiment_id,
        hypothesis=hypothesis,
        notes=(
            "Generated after official AIC evaluator completion; harness outputs are post-hoc evidence.",
            f"Official eval result root: {result_root}",
        ),
    )
    base_manifest = attach_scoring_yaml_reduction(
        base_manifest,
        scoring_reduction,
        updated_at_utc=generated_at,
    )
    promotion_input_manifest_artifact = _snapshot_artifact(
        kind="run_manifest",
        payload=base_manifest.to_dict(),
        run_id=run_id,
        uri=f"memory://aic_signal_harness/live_eval/promotion_input_manifest/{run_id}",
        provenance={
            "producer": "aic_signal_harness.live_eval",
            "run_id": run_id,
            "snapshot": "pre_reward_failure_manifest",
        },
    )

    pending_ledger_entry = build_ledger_entry(
        base_manifest,
        manifest_artifact=promotion_input_manifest_artifact,
        recorded_at_utc=generated_at,
    )
    promotion_snapshot = _promotion_decision(
        pending_ledger_entry,
        baseline=baseline,
        bootstrap_promotion=bootstrap_promotion,
        min_improvement=min_improvement,
        eligible_for_submission=eligible_for_submission,
        generated_at_utc=generated_at,
    )
    promotion_snapshot_artifact = (
        None
        if promotion_snapshot is None
        else _promotion_snapshot_artifact(promotion_snapshot)
    )
    accepted_baseline_update = None
    if (
        promotion_snapshot is not None
        and promotion_snapshot.accepted
        and update_baseline_path is not None
    ):
        accepted_baseline_update = Path(update_baseline_path).expanduser()
        if not append_ledger or typed_ledger_path is None:
            raise HarnessIOError("accepted baseline update requires successful ledger append")
    if promotion_snapshot is not None and (not append_ledger or typed_ledger_path is None):
        raise HarnessIOError("promotion decisions require successful ledger append")
    _preflight_live_eval_writes(
        run_id=run_id,
        output_paths=(
            score_report_path,
            scoring_artifact_path,
            policy_trace_report_path,
            policy_trace_artifact_path,
            episode_trace_path,
            training_signal_report_path,
            manifest_path,
            promotion_path,
            reward_report_path,
            failure_report_path,
            next_experiment_path,
            next_experiment_plan_path,
            ledger_entry_path,
            summary_path,
        ),
        ledger_path=typed_ledger_path,
        append_ledger=append_ledger,
        overwrite=overwrite,
    )
    if accepted_baseline_update is not None:
        _preflight_baseline_update_path_collisions(
            accepted_baseline_update,
            output_paths=(
                score_report_path,
                scoring_artifact_path,
                policy_trace_report_path,
                policy_trace_artifact_path,
                episode_trace_path,
                training_signal_report_path,
                manifest_path,
                promotion_path,
                reward_report_path,
                failure_report_path,
                next_experiment_path,
                next_experiment_plan_path,
                ledger_entry_path,
                summary_path,
            ),
            ledger_path=typed_ledger_path,
        )
    if promotion_snapshot is not None and promotion_path is not None:
        _preflight_publish_target(
            promotion_path,
            overwrite=overwrite,
            field_name="promotion decision path",
        )
    if accepted_baseline_update is not None:
        _preflight_publish_target(
            accepted_baseline_update,
            overwrite=True,
            field_name="baseline decision path",
        )
    harness_root.mkdir(parents=True, exist_ok=True)
    write_json(score_report_path, scoring_reduction.score.to_dict(), overwrite=overwrite)
    write_json(scoring_artifact_path, scoring_reduction.artifact.to_dict(), overwrite=overwrite)
    if policy_trace_reduction is not None:
        assert policy_trace_report_path is not None
        assert policy_trace_artifact_path is not None
        write_json(
            policy_trace_report_path,
            policy_trace_reduction.report.to_dict(),
            overwrite=overwrite,
        )
        write_json(
            policy_trace_artifact_path,
            policy_trace_reduction.artifact.to_dict(),
            overwrite=overwrite,
        )

    reward_failure = derive_reward_failure_reports(
        base_manifest,
        promotion=promotion_snapshot,
        generated_at_utc=generated_at,
    )
    write_json(
        reward_report_path,
        reward_failure.reward_report.to_dict(),
        overwrite=overwrite,
    )
    write_json(
        failure_report_path,
        reward_failure.failure_report.to_dict(),
        overwrite=overwrite,
    )
    reward_failure_source_artifact_refs = (scoring_reduction.artifact,) + (
        ()
        if promotion_snapshot_artifact is None
        or promotion_snapshot is None
        or promotion_snapshot.metric.baseline_value is None
        else (promotion_snapshot_artifact,)
    )
    reward_failure_source_artifacts = tuple(
        {
            "kind": artifact.kind,
            "sha256": artifact.sha256,
        }
        for artifact in reward_failure_source_artifact_refs
    )
    reward_failure_source_provenance: dict[str, Any] = {
        "source_artifacts": list(reward_failure_source_artifacts),
        "source_manifest_run_id": base_manifest.run_id,
    }
    reward_report_artifact = ArtifactRef(
        kind="reward_report",
        path=str(reward_report_path),
        sha256=sha256_file(reward_report_path),
        provenance={
            "producer": "aic_signal_harness.live_eval",
            "run_id": run_id,
            "derivation": "derive_reward_failure_reports",
            **reward_failure_source_provenance,
        },
    )
    failure_report_artifact = ArtifactRef(
        kind="failure_report",
        path=str(failure_report_path),
        sha256=sha256_file(failure_report_path),
        provenance={
            "producer": "aic_signal_harness.live_eval",
            "run_id": run_id,
            "derivation": "derive_reward_failure_reports",
            **reward_failure_source_provenance,
        },
    )
    report_artifacts = (reward_report_artifact, failure_report_artifact)
    derived_artifacts: tuple[ArtifactRef, ...] = ()
    if policy_trace_reduction is not None:
        assert episode_trace_path is not None
        assert training_signal_report_path is not None
        episode_trace = derive_episode_trace(
            run_id=run_id,
            policy_events=policy_trace_reduction.events,
            source_artifacts=(
                scoring_reduction.artifact,
                policy_trace_reduction.artifact,
                *reward_failure_source_artifact_refs[1:],
                reward_report_artifact,
                failure_report_artifact,
            ),
            score_report=scoring_reduction.score,
            reward_report=reward_failure.reward_report,
            failure_report=reward_failure.failure_report,
            generated_at_utc=generated_at,
        )
        write_json(episode_trace_path, episode_trace.to_dict(), overwrite=overwrite)
        episode_trace_artifact = ArtifactRef(
            kind="episode_trace",
            path=str(episode_trace_path),
            sha256=sha256_file(episode_trace_path),
            provenance={
                "producer": "aic_signal_harness.live_eval",
                "run_id": run_id,
                "derivation": "derive_episode_trace",
                "source_artifacts": [
                    {"kind": artifact.kind, "sha256": artifact.sha256}
                    for artifact in (
                        scoring_reduction.artifact,
                        policy_trace_reduction.artifact,
                        *reward_failure_source_artifact_refs[1:],
                        reward_report_artifact,
                        failure_report_artifact,
                    )
                ],
            },
        )
        training_signal_report = derive_training_signal_report(
            episode_trace=episode_trace,
            source_trace=episode_trace_artifact,
            reward_report=reward_failure.reward_report,
            failure_report=reward_failure.failure_report,
            source_reward_report=reward_report_artifact,
            source_failure_report=failure_report_artifact,
            generated_at_utc=generated_at,
        )
        write_json(
            training_signal_report_path,
            training_signal_report.to_dict(),
            overwrite=overwrite,
        )
        training_signal_report_artifact = ArtifactRef(
            kind="training_signal_report",
            path=str(training_signal_report_path),
            sha256=sha256_file(training_signal_report_path),
            provenance={
                "producer": "aic_signal_harness.live_eval",
                "run_id": run_id,
                "derivation": "derive_training_signal_report",
                "source_trace_sha256": episode_trace_artifact.sha256,
            },
        )
        derived_artifacts = (episode_trace_artifact, training_signal_report_artifact)

    manifest = RunManifest(
        run_id=base_manifest.run_id,
        status=base_manifest.status,
        backend=base_manifest.backend,
        created_at_utc=base_manifest.created_at_utc,
        updated_at_utc=base_manifest.updated_at_utc,
        artifacts=base_manifest.artifacts + report_artifacts + derived_artifacts,
        score=base_manifest.score,
        experiment_id=base_manifest.experiment_id,
        hypothesis=base_manifest.hypothesis,
        notes=base_manifest.notes,
    )
    write_json(manifest_path, manifest.to_dict(), overwrite=overwrite)
    manifest_artifact = ArtifactRef(
        kind="run_manifest",
        path=str(manifest_path),
        sha256=sha256_file(manifest_path),
        provenance={"producer": "aic_signal_harness.live_eval", "run_id": run_id},
    )
    ledger_entry_without_promotion = build_ledger_entry(
        manifest,
        manifest_artifact=manifest_artifact,
        recorded_at_utc=generated_at,
    )
    promotion = _promotion_decision(
        ledger_entry_without_promotion,
        baseline=baseline,
        bootstrap_promotion=bootstrap_promotion,
        min_improvement=min_improvement,
        eligible_for_submission=eligible_for_submission,
        generated_at_utc=generated_at,
    )
    accepted_baseline_update = None
    if promotion is not None and promotion.accepted and update_baseline_path is not None:
        accepted_baseline_update = Path(update_baseline_path).expanduser()

    if write_next_experiment:
        assert next_experiment_path is not None
        assert next_experiment_plan_path is not None
        next_experiment = derive_next_experiment_report(
            manifest,
            reward_report=reward_failure.reward_report,
            failure_report=reward_failure.failure_report,
            gate_id=gate_id,
            promotion=promotion,
            objective="Improve official AIC qualification score without violating runtime boundaries.",
            generated_at_utc=generated_at,
        )
        write_json(next_experiment_path, next_experiment.to_dict(), overwrite=overwrite)
        next_experiment_artifact = ArtifactRef(
            kind="next_experiment_report",
            path=str(next_experiment_path),
            sha256=sha256_file(next_experiment_path),
            provenance={
                "producer": "aic_signal_harness.live_eval",
                "run_id": run_id,
                "gate_id": gate_id,
                "derivation": "derive_next_experiment_report",
            },
        )
        next_experiment_plan = derive_next_experiment_plan(
            next_experiment,
            manifest=manifest,
            source_next_experiment=next_experiment_artifact,
            source_manifest=manifest_artifact,
            generated_at_utc=generated_at,
        )
        write_json(next_experiment_plan_path, next_experiment_plan.to_dict(), overwrite=overwrite)

    ledger_entry = build_ledger_entry(
        manifest,
        manifest_artifact=manifest_artifact,
        promotion={} if promotion is None else promotion.to_dict(),
        recorded_at_utc=generated_at,
    )
    write_json(ledger_entry_path, ledger_entry.to_dict(), overwrite=overwrite)

    finalization = LiveEvalFinalization(
        manifest=manifest,
        manifest_artifact=manifest_artifact,
        manifest_path=manifest_path,
        score_report_path=score_report_path,
        scoring_artifact_path=scoring_artifact_path,
        policy_trace_report_path=policy_trace_report_path,
        policy_trace_artifact_path=policy_trace_artifact_path,
        episode_trace_path=episode_trace_path,
        training_signal_report_path=training_signal_report_path,
        ledger_entry_path=ledger_entry_path,
        ledger_path=typed_ledger_path,
        promotion_path=promotion_path,
        reward_report_path=reward_report_path,
        failure_report_path=failure_report_path,
        next_experiment_path=next_experiment_path,
        next_experiment_plan_path=next_experiment_plan_path,
        summary_path=summary_path,
    )
    staged_promotion_path = None
    staged_baseline_path = None
    published_decisions: list[_PublishedDecision] = []
    try:
        if accepted_baseline_update is not None and promotion is not None:
            staged_baseline_path = _stage_baseline_decision(
                accepted_baseline_update,
                promotion,
            )
        if promotion is not None and promotion_path is not None:
            staged_promotion_path = _stage_promotion_decision(promotion_path, promotion)
        if staged_promotion_path is not None and promotion_path is not None:
            published_decisions.append(
                _publish_staged_decision(
                    staged_promotion_path,
                    promotion_path,
                    field_name="promotion decision",
                )
            )
            staged_promotion_path = None
        if staged_baseline_path is not None and accepted_baseline_update is not None:
            published_decisions.append(
                _publish_staged_decision(
                    staged_baseline_path,
                    accepted_baseline_update,
                    field_name="baseline decision",
                )
            )
            staged_baseline_path = None
        if append_ledger and typed_ledger_path is not None:
            append_ledger_entry(typed_ledger_path, ledger_entry, allow_duplicate=False)
    except BaseException:
        if staged_promotion_path is not None:
            staged_promotion_path.unlink(missing_ok=True)
        if staged_baseline_path is not None:
            staged_baseline_path.unlink(missing_ok=True)
        for published_decision in reversed(published_decisions):
            published_decision.rollback()
        raise
    for published_decision in published_decisions:
        published_decision.discard_backup()
    write_json(finalization.summary_path, finalization.to_summary(), overwrite=overwrite)
    return finalization


def _reduce_optional_policy_trace(
    *,
    policy_trace: str | Path | None,
    run_id: str,
    reduced_at_utc: str,
) -> PolicyTraceReduction | None:
    if policy_trace is None:
        return None
    trace_path = Path(policy_trace).expanduser()
    reduction = reduce_policy_trace_jsonl(
        trace_path,
        provenance={"producer": "aic_signal_harness.live_eval", "run_id": run_id},
        reduced_at_utc=reduced_at_utc,
    )
    if reduction.report.run_id != run_id:
        raise HarnessIOError(
            f"policy trace run_id {reduction.report.run_id!r} "
            f"does not match finalizer run_id {run_id!r}"
        )
    return reduction


def _resolve_bound_scoring_yaml(scoring_yaml: str | Path, result_root: Path) -> Path:
    scoring_path = Path(scoring_yaml).expanduser().resolve(strict=False)
    expected_path = (result_root / "eval" / "scoring.yaml").resolve(strict=False)
    if scoring_path != expected_path:
        raise HarnessIOError(
            "live eval scoring_yaml must be bound to result_root/eval/scoring.yaml: "
            f"{scoring_path} != {expected_path}"
        )
    if not scoring_path.is_file():
        raise HarnessIOError(f"scoring_yaml does not exist: {scoring_path}")
    return scoring_path


def _preflight_live_eval_writes(
    *,
    run_id: str,
    output_paths: tuple[Path | None, ...],
    ledger_path: Path | None,
    append_ledger: bool,
    overwrite: bool,
) -> None:
    if not overwrite:
        existing_paths = tuple(path for path in output_paths if path is not None and path.exists())
        if existing_paths:
            raise HarnessIOError(
                "live eval output already exists; pass overwrite=True to replace it: "
                + ", ".join(str(path) for path in existing_paths)
            )
    if append_ledger and ledger_path is not None:
        ledger_resolved = _resolved_contract_path(ledger_path, "ledger path")
        for output_path in output_paths:
            if output_path is None:
                continue
            if ledger_resolved == _resolved_contract_path(
                output_path,
                "live eval generated output path",
            ):
                raise HarnessIOError(
                    "append ledger path must not collide with live eval generated output: "
                    f"{ledger_path} == {output_path}"
                )
        if any(entry.run_id == run_id for entry in read_ledger_entries(ledger_path)):
            raise HarnessIOError(f"ledger already contains run_id={run_id}")


def _preflight_baseline_update_path_collisions(
    baseline_path: Path,
    *,
    output_paths: tuple[Path | None, ...],
    ledger_path: Path | None,
) -> None:
    baseline_resolved = _resolved_contract_path(
        baseline_path,
        "baseline decision path",
    )
    for output_path in output_paths:
        if output_path is None:
            continue
        if baseline_resolved == _resolved_contract_path(
            output_path,
            "live eval generated output path",
        ):
            raise HarnessIOError(
                "baseline decision path must not collide with live eval generated output: "
                f"{baseline_path} == {output_path}"
            )
    if ledger_path is not None and baseline_resolved == _resolved_contract_path(
        ledger_path,
        "ledger path",
    ):
        raise HarnessIOError(
            "baseline decision path must not collide with append ledger path: "
            f"{baseline_path} == {ledger_path}"
        )


def _resolved_contract_path(path: Path, field_name: str) -> Path:
    try:
        return path.expanduser().resolve(strict=False)
    except (OSError, ValueError) as exc:
        raise HarnessIOError(f"{field_name} cannot be resolved: {path}") from exc


def _preflight_publish_target(
    path: Path,
    *,
    overwrite: bool,
    field_name: str,
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HarnessIOError(f"{field_name} parent cannot be created: {path.parent}") from exc
    if path.exists():
        if path.is_dir():
            raise HarnessIOError(f"{field_name} must not be a directory: {path}")
        if not overwrite:
            raise HarnessIOError(
                f"{field_name} already exists; pass overwrite=True to replace it: {path}"
            )


def _load_baseline_decision(
    *,
    baseline_path: str | Path | None,
) -> PromotionDecision | None:
    if baseline_path is None:
        return None
    path = Path(baseline_path).expanduser()
    if path.exists():
        return read_promotion_decision(str(path))
    raise HarnessIOError(f"baseline does not exist: {path}")


def _validate_declared_policy_trace(policy_trace: str | Path | None) -> None:
    if policy_trace is None:
        return
    trace_path = Path(policy_trace).expanduser()
    if not trace_path.is_file():
        raise HarnessIOError(f"policy trace does not exist: {trace_path}")
    if trace_path.stat().st_size <= 0:
        raise HarnessIOError(f"policy trace is empty: {trace_path}")


def _promotion_decision(
    entry,
    *,
    baseline: PromotionDecision | None,
    bootstrap_promotion: bool,
    min_improvement: float,
    eligible_for_submission: bool,
    generated_at_utc: str,
) -> PromotionDecision | None:
    if baseline is None and not bootstrap_promotion:
        return None
    return promote_ledger_entry(
        entry,
        baseline=baseline,
        bootstrap=bootstrap_promotion and baseline is None,
        min_improvement=min_improvement,
        eligible_for_submission=eligible_for_submission,
        notes=("Generated by live eval harness finalizer.",),
        decided_at_utc=generated_at_utc,
    )


def _stage_promotion_decision(
    promotion_path: Path,
    promotion: PromotionDecision,
) -> Path:
    promotion_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=promotion_path.parent,
        prefix=f".{promotion_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        staged_path = Path(handle.name)
    staged_path.unlink(missing_ok=True)
    try:
        write_promotion_decision(str(staged_path), promotion, overwrite=False)
    except BaseException:
        staged_path.unlink(missing_ok=True)
        raise
    return staged_path


def _stage_baseline_decision(
    baseline_path: Path,
    promotion: PromotionDecision,
) -> Path:
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=baseline_path.parent,
        prefix=f".{baseline_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        staged_path = Path(handle.name)
    staged_path.unlink(missing_ok=True)
    try:
        write_baseline_decision(str(staged_path), promotion, overwrite=False)
    except BaseException:
        staged_path.unlink(missing_ok=True)
        raise
    return staged_path


@dataclass
class _PublishedDecision:
    target_path: Path
    backup_path: Path | None
    created_target: bool

    def rollback(self) -> None:
        if self.created_target:
            self.target_path.unlink(missing_ok=True)
            return
        if self.backup_path is not None and self.backup_path.exists():
            self.backup_path.replace(self.target_path)

    def discard_backup(self) -> None:
        if self.backup_path is not None:
            self.backup_path.unlink(missing_ok=True)


def _publish_staged_decision(
    staged_path: Path,
    target_path: Path,
    *,
    field_name: str,
) -> _PublishedDecision:
    backup_path = None
    created_target = not target_path.exists()
    try:
        if not created_target:
            backup_path = _temporary_sibling(target_path, suffix=".bak")
            target_path.replace(backup_path)
        staged_path.replace(target_path)
    except OSError as exc:
        staged_path.unlink(missing_ok=True)
        if backup_path is not None and backup_path.exists():
            try:
                backup_path.replace(target_path)
            except OSError as restore_exc:
                raise HarnessIOError(
                    f"failed to publish {field_name}: {exc}; "
                    f"failed to restore previous target: {restore_exc}"
                ) from exc
        raise HarnessIOError(f"failed to publish {field_name}: {exc}") from exc
    return _PublishedDecision(
        target_path=target_path,
        backup_path=backup_path,
        created_target=created_target,
    )


def _temporary_sibling(path: Path, *, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=suffix,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    temporary_path.unlink(missing_ok=True)
    return temporary_path


def _promotion_snapshot_artifact(promotion: PromotionDecision) -> ArtifactRef:
    return _snapshot_artifact(
        kind="promotion_decision_snapshot",
        payload=promotion.to_dict(),
        run_id=promotion.run_id,
        uri=(
            "memory://aic_signal_harness/live_eval/"
            f"promotion_decision_snapshot/{promotion.run_id}"
        ),
        provenance={
            "producer": "aic_signal_harness.live_eval",
            "run_id": promotion.run_id,
            "derivation": "promote_ledger_entry",
            "source_manifest_artifact": promotion.manifest.to_dict(),
        },
    )


def _snapshot_artifact(
    *,
    kind: str,
    payload: Mapping[str, Any],
    run_id: str,
    uri: str,
    provenance: Mapping[str, Any],
) -> ArtifactRef:
    return ArtifactRef(
        kind=kind,
        uri=uri,
        sha256=_json_mapping_sha256(payload),
        provenance={
            **dict(provenance),
            "snapshot_schema": "json_object_sha256",
            "snapshot_run_id": run_id,
        },
    )


def _json_mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded + b"\n").hexdigest()


def _backend_spec(
    *,
    run_id: str,
    model_image: str | None,
    model_image_id: str | None,
    policy_checkpoint: str | Path | None,
    backend_kind: str | BackendKind | None,
    planner_mode: str | None,
    runtime_env: Mapping[str, str],
) -> PolicyBackendSpec:
    mode = (planner_mode or runtime_env.get("AIC_LEWM_PLANNER_MODE") or "lewm_mpc").strip()
    kind = (
        BackendKind.parse(backend_kind)
        if backend_kind is not None
        else BackendKind.replay_servo
        if mode == "replay"
        else BackendKind.lewm_world_model
    )
    image = _optional_text(model_image, "model_image") or "aic-lewm-learned:latest"
    image_sha256 = _docker_image_sha256(model_image_id) if kind is BackendKind.lewm_world_model else None
    docker_artifact = ArtifactRef(
        kind="docker_image",
        uri=_docker_uri(image),
        sha256=image_sha256,
        provenance={"run_id": run_id},
    )
    policy_artifact = (
        _runtime_policy_checkpoint_artifact(
            policy_checkpoint,
            run_id=run_id,
            runtime_env=runtime_env,
        )
        if policy_checkpoint is not None
        else docker_artifact
    )
    runtime_boundary = RuntimeBoundaryProof(
        deterministic=True,
        uses_online_language_model_control=False,
        legal_observation_contract=_DEFAULT_LEGAL_OBSERVATION_CONTRACT,
        policy_artifact=policy_artifact if kind is BackendKind.lewm_world_model else None,
        notes="Harness finalizer records this after the official live eval; no online LLM control.",
    )
    return PolicyBackendSpec(
        backend_kind=kind,
        name=f"aic-lewm-{mode}",
        runtime_role=RuntimeRole.live_policy,
        training_sources=(TrainingSourceKind.official_demo,),
        simulator_sources=(SimulatorKind.gazebo,),
        runtime_allowed=True,
        leakage_class=LeakageClass.legal_policy_input,
        runtime_boundary=runtime_boundary,
        description="AIC LEWM policy evaluated through the official AIC runtime.",
        config={
            "planner_mode": mode,
            "runtime_env": {
                key: runtime_env[key]
                for key in _CONFIG_ENV_KEYS
                if key in runtime_env
            },
        },
        provenance={
            "producer": "aic_signal_harness.live_eval",
            "model_image": image,
            "model_image_id": _optional_text(model_image_id, "model_image_id"),
            "docker_artifact": docker_artifact.to_dict(),
            "runtime_env": dict(runtime_env),
        },
    )


def _runtime_policy_checkpoint_artifact(
    policy_checkpoint: str | Path,
    *,
    run_id: str,
    runtime_env: Mapping[str, str],
) -> ArtifactRef:
    checkpoint_path = Path(policy_checkpoint).expanduser().resolve(strict=False)
    if not checkpoint_path.exists():
        raise HarnessIOError(f"policy_checkpoint does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise HarnessIOError(f"policy_checkpoint must be a file: {checkpoint_path}")
    container_path = _require_text(
        runtime_env.get("AIC_LEWM_CHECKPOINT"),
        "runtime_env.AIC_LEWM_CHECKPOINT",
    )
    return ArtifactRef(
        kind="policy_checkpoint",
        path=str(checkpoint_path),
        sha256=sha256_file(checkpoint_path),
        provenance={
            "producer": "aic_signal_harness.live_eval",
            "run_id": run_id,
            "derivation": "bind_runtime_policy_checkpoint",
            "container_path": container_path,
        },
    )


def _typed_backend_override(
    backend: PolicyBackendSpec | Mapping[str, Any],
) -> PolicyBackendSpec:
    if isinstance(backend, PolicyBackendSpec):
        return backend
    try:
        return PolicyBackendSpec.from_dict(backend)
    except Exception as exc:
        raise HarnessIOError(f"live eval backend override is invalid: {exc}") from exc


def _typed_extra_manifest_artifacts(
    artifacts: tuple[ArtifactRef | Mapping[str, Any], ...],
) -> tuple[ArtifactRef, ...]:
    if not isinstance(artifacts, tuple):
        raise HarnessIOError("extra_manifest_artifacts must be a tuple")
    typed_artifacts: list[ArtifactRef] = []
    for index, artifact in enumerate(artifacts):
        if isinstance(artifact, ArtifactRef):
            typed_artifacts.append(artifact)
            continue
        try:
            typed_artifacts.append(ArtifactRef.from_dict(artifact))
        except Exception as exc:
            raise HarnessIOError(
                f"extra_manifest_artifacts[{index}] is invalid: {exc}"
            ) from exc
    return tuple(typed_artifacts)


def _docker_uri(image: str) -> str:
    return "docker://" + image.lstrip("/")


def _docker_image_sha256(image_id: str | None) -> str:
    value = _require_text(image_id, "model_image_id")
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
        raise HarnessIOError("model_image_id must be a sha256 Docker image id")
    return digest.lower()


def _runtime_env_from_mapping(runtime_env: Mapping[str, str] | None) -> dict[str, str]:
    source = os.environ if runtime_env is None else runtime_env
    return {
        key: str(source[key])
        for key in _RUNTIME_ENV_KEYS
        if key in source and str(source[key]).strip()
    }


def runtime_env_from_mapping(runtime_env: Mapping[str, str] | None) -> dict[str, str]:
    """Return the live-eval runtime environment fields recorded by the harness."""

    return _runtime_env_from_mapping(runtime_env)


def _resolve_existing_dir(path: str | Path, field_name: str) -> Path:
    directory = Path(path).expanduser().resolve(strict=False)
    if not directory.is_dir():
        raise HarnessIOError(f"{field_name} must be an existing directory: {directory}")
    return directory


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessIOError(f"{field_name} must be a nonempty string")
    return value.strip()


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name)


def _optional_path(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _cmd_finalize(args: argparse.Namespace) -> int:
    finalization = finalize_live_eval_run(
        run_id=args.run_id,
        result_root=args.result_root,
        harness_root=args.harness_root,
        scoring_yaml=args.scoring_yaml,
        policy_trace=args.policy_trace,
        ledger_path=args.ledger,
        baseline_path=args.baseline,
        update_baseline_path=args.update_baseline,
        bootstrap_promotion=args.bootstrap_promotion,
        min_improvement=args.min_improvement,
        eligible_for_submission=args.eligible_for_submission,
        gate_id=args.gate_id,
        experiment_id=args.experiment_id,
        hypothesis=args.hypothesis,
        model_image=args.model_image,
        model_image_id=args.model_image_id,
        policy_checkpoint=args.policy_checkpoint,
        backend_kind=args.backend_kind,
        planner_mode=args.planner_mode,
        overwrite=args.overwrite,
        append_ledger=not args.no_append_ledger,
        write_next_experiment=not args.no_next_experiment,
    )
    print(f"AIC_HARNESS_MANIFEST_PATH={finalization.manifest_path}")
    print(f"AIC_HARNESS_LEDGER_ENTRY_PATH={finalization.ledger_entry_path}")
    if finalization.ledger_path is not None:
        print(f"AIC_HARNESS_LEDGER_PATH={finalization.ledger_path}")
    if finalization.promotion_path is not None:
        print(f"AIC_HARNESS_PROMOTION_PATH={finalization.promotion_path}")
    if finalization.next_experiment_path is not None:
        print(f"AIC_HARNESS_NEXT_EXPERIMENT_PATH={finalization.next_experiment_path}")
    print(f"AIC_HARNESS_SUMMARY_PATH={finalization.summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize a live AIC eval into harness evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    finalize = subparsers.add_parser("finalize", help="Reduce one completed live eval")
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--result-root", required=True)
    finalize.add_argument("--harness-root", required=True)
    finalize.add_argument("--scoring-yaml", required=True)
    finalize.add_argument("--policy-trace")
    finalize.add_argument("--ledger")
    finalize.add_argument("--baseline")
    finalize.add_argument("--update-baseline")
    finalize.add_argument("--bootstrap-promotion", action="store_true")
    finalize.add_argument("--min-improvement", type=float, default=0.0)
    finalize.add_argument("--eligible-for-submission", action="store_true")
    finalize.add_argument("--gate-id", default="live_eval")
    finalize.add_argument("--experiment-id")
    finalize.add_argument("--hypothesis")
    finalize.add_argument("--model-image")
    finalize.add_argument("--model-image-id")
    finalize.add_argument("--policy-checkpoint")
    finalize.add_argument("--backend-kind")
    finalize.add_argument("--planner-mode")
    finalize.add_argument("--overwrite", action="store_true", default=_bool_env("AIC_HARNESS_OVERWRITE"))
    finalize.add_argument("--no-append-ledger", action="store_true")
    finalize.add_argument("--no-next-experiment", action="store_true")
    finalize.set_defaults(func=_cmd_finalize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except HarnessIOError as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
