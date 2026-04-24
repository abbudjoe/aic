from __future__ import annotations

from pathlib import Path

import pytest

from aic_signal_harness import (
    ArtifactRef,
    BackendKind,
    FailureKind,
    FailureLabel,
    FailureReport,
    FailureSeverity,
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
    PromotionDecision,
    PromotionDecisionKind,
    PromotionGoal,
    PromotionMetric,
    RewardReport,
    RewardSignalKind,
    RewardTerm,
    RunManifest,
    RunStatus,
    RuntimeBoundaryProof,
    RuntimeRole,
    ScoreReport,
    SimulatorKind,
    TrainingSourceKind,
    TrialScore,
    backfill_legacy_replay_policy_eval,
    derive_next_experiment_plan,
    derive_next_experiment_report,
    derive_reward_failure_reports,
    sha256_file,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "aic_lewm_policy" / "experiments" / "runs"
_ABS_SCORE_SOURCE = "/tmp/eval/scoring.yaml"


def _live_backend() -> PolicyBackendSpec:
    return PolicyBackendSpec(
        backend_kind=BackendKind.replay_servo,
        name="replay-servo-baseline",
        runtime_role=RuntimeRole.live_policy,
        training_sources=(TrainingSourceKind.official_demo,),
        simulator_sources=(SimulatorKind.offline_replay,),
        runtime_allowed=True,
        leakage_class=LeakageClass.legal_policy_input,
        runtime_boundary=RuntimeBoundaryProof(
            deterministic=True,
            uses_online_language_model_control=False,
            legal_observation_contract="official aic_model observations only",
        ),
        description="Replay a screened demonstration through the official runtime path.",
        config={"control_hz": 10},
        provenance={"producer": "pytest"},
    )


def _live_backend_with_runtime_env(runtime_env: dict[str, str]) -> PolicyBackendSpec:
    base = _live_backend()
    return PolicyBackendSpec(
        backend_kind=base.backend_kind,
        name=base.name,
        runtime_role=base.runtime_role,
        training_sources=base.training_sources,
        simulator_sources=base.simulator_sources,
        runtime_allowed=base.runtime_allowed,
        leakage_class=base.leakage_class,
        runtime_boundary=base.runtime_boundary,
        description=base.description,
        config=base.config,
        provenance={**base.provenance, "legacy_runtime_env": runtime_env},
    )


def _score_report(*, total_offset: float = 0.0) -> ScoreReport:
    trials = {
        "trial_1": TrialScore(total=48.0 + total_offset, tier_1=1.0, tier_2=22.0, tier_3=25.0),
        "trial_2": TrialScore(total=48.0, tier_1=1.0, tier_2=22.0, tier_3=25.0),
        "trial_3": TrialScore(total=29.0, tier_1=1.0, tier_2=22.0, tier_3=6.0),
    }
    return ScoreReport(
        source=_ABS_SCORE_SOURCE,
        parsed_at_utc="2026-04-23T00:00:00Z",
        total=sum(trial.total for trial in trials.values()),
        trials=trials,
    )


def _manifest(score: ScoreReport | None = None) -> RunManifest:
    return RunManifest(
        run_id="gate2-candidate",
        status=RunStatus.completed if score is not None else RunStatus.failed,
        backend=_live_backend(),
        created_at_utc="2026-04-23T00:00:00Z",
        updated_at_utc="2026-04-23T00:00:00Z",
        artifacts=(
            ArtifactRef(
                kind="scoring_yaml",
                path=_ABS_SCORE_SOURCE,
                sha256="a" * 64,
            ),
        )
        if score is not None
        else (),
        score=score,
        experiment_id="gate2",
    )


def _manifest_with_runtime_env(
    score: ScoreReport,
    runtime_env: dict[str, str],
) -> RunManifest:
    return RunManifest(
        run_id="gate2-candidate",
        status=RunStatus.completed,
        backend=_live_backend_with_runtime_env(runtime_env),
        created_at_utc="2026-04-23T00:00:00Z",
        updated_at_utc="2026-04-23T00:00:00Z",
        artifacts=(
            ArtifactRef(
                kind="scoring_yaml",
                path=_ABS_SCORE_SOURCE,
                sha256="a" * 64,
            ),
        ),
        score=score,
        experiment_id="gate2",
    )


def _reward_report(
    run_id: str = "gate2-candidate",
    *,
    total: float = 125.0,
    promotion_improvement: float = 0.2,
    promotion_margin_gap: float = -0.8,
) -> RewardReport:
    return RewardReport(
        run_id=run_id,
        generated_at_utc="2026-04-23T00:00:00Z",
        terms=(
            RewardTerm(
                name="official.score.total",
                value=total,
                signal_kind=RewardSignalKind.official_score,
                leakage_class=LeakageClass.privileged_eval_signal,
                source=_ABS_SCORE_SOURCE,
            ),
            RewardTerm(
                name="promotion.metric.improvement",
                value=promotion_improvement,
                signal_kind=RewardSignalKind.diagnostic,
                leakage_class=LeakageClass.privileged_eval_signal,
                source="promotion_decision",
            ),
            RewardTerm(
                name="promotion.metric.margin_gap",
                value=promotion_margin_gap,
                signal_kind=RewardSignalKind.diagnostic,
                leakage_class=LeakageClass.privileged_eval_signal,
                source="promotion_decision",
            ),
        ),
    )


def _failure_report(run_id: str = "gate2-candidate") -> FailureReport:
    return FailureReport(
        run_id=run_id,
        generated_at_utc="2026-04-23T00:00:00Z",
        labels=(
            FailureLabel(
                kind=FailureKind.no_partial_or_full_insertion,
                severity=FailureSeverity.blocker,
                summary="Tier 3 evidence is proximity-only.",
                source=_ABS_SCORE_SOURCE,
                leakage_class=LeakageClass.privileged_eval_signal,
                evidence={"max_tier_3": 25.0},
            ),
            FailureLabel(
                kind=FailureKind.below_promotion_margin,
                severity=FailureSeverity.blocker,
                summary="Candidate did not clear the margin.",
                source="promotion_decision",
                leakage_class=LeakageClass.privileged_eval_signal,
                evidence={
                    "decision": "rejected",
                    "accepted": False,
                    "metric_name": "evaluation.score.total",
                    "goal": "max",
                    "value": 125.0,
                    "baseline_value": 124.8,
                    "min_improvement": 1.0,
                    "improvement": 0.2,
                    "margin_gap": -0.8,
                },
            ),
        ),
    )


def _promotion_decision(
    *,
    value: float = 125.0,
    baseline_value: float = 124.8,
    run_id: str = "gate2-candidate",
) -> PromotionDecision:
    return PromotionDecision(
        decided_at_utc="2026-04-23T00:52:15Z",
        decision=PromotionDecisionKind.rejected,
        accepted=False,
        eligible_for_submission=False,
        reason=f"{value} < {baseline_value} + 1.0",
        metric=PromotionMetric(
            name="evaluation.score.total",
            goal=PromotionGoal.max,
            value=value,
            baseline_value=baseline_value,
            min_improvement=1.0,
        ),
        run_id=run_id,
        manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-candidate/manifest.json",
            sha256="b" * 64,
        ),
        baseline_run_id="gate2-baseline",
        baseline_manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-baseline/manifest.json",
            sha256="c" * 64,
        ),
    )


def _accepted_promotion_decision(
    *,
    value: float,
    baseline_value: float,
) -> PromotionDecision:
    return PromotionDecision(
        decided_at_utc="2026-04-23T00:52:15Z",
        decision=PromotionDecisionKind.accepted,
        accepted=True,
        eligible_for_submission=True,
        reason=f"{value} >= {baseline_value} + 1.0",
        metric=PromotionMetric(
            name="evaluation.score.total",
            goal=PromotionGoal.max,
            value=value,
            baseline_value=baseline_value,
            min_improvement=1.0,
        ),
        run_id="gate2-candidate",
        manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-candidate/manifest.json",
            sha256="b" * 64,
        ),
        baseline_run_id="gate2-baseline",
        baseline_manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-baseline/manifest.json",
            sha256="c" * 64,
        ),
    )


def _run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def _write_source_artifacts(
    tmp_path: Path,
    *,
    manifest: RunManifest,
    next_experiment: NextExperimentReport,
) -> tuple[ArtifactRef, ArtifactRef]:
    manifest_path = tmp_path / "run_manifest.json"
    next_experiment_path = tmp_path / "next_experiment.json"
    write_json(manifest_path, manifest.to_dict())
    write_json(next_experiment_path, next_experiment.to_dict())
    return (
        ArtifactRef(
            kind="run_manifest",
            path=str(manifest_path),
            sha256=sha256_file(manifest_path),
            provenance={"producer": "pytest", "run_id": manifest.run_id},
        ),
        ArtifactRef(
            kind="next_experiment_report",
            path=str(next_experiment_path),
            sha256=sha256_file(next_experiment_path),
            provenance={
                "producer": "pytest",
                "run_id": next_experiment.run_id,
                "gate_id": next_experiment.gate_id,
                "derivation": "derive_next_experiment_report",
            },
        ),
    )


def _historical_baseline_seed(
    *,
    run_id: str,
    value: float,
    baseline_value: float,
) -> PromotionDecision:
    baseline_manifest_path = _run_dir(run_id) / "manifest.json"
    return PromotionDecision(
        decided_at_utc="2026-04-22T17:53:13Z",
        decision=PromotionDecisionKind.accepted,
        accepted=True,
        eligible_for_submission=True,
        reason="seeded historical baseline",
        metric=PromotionMetric(
            name="evaluation.score.total",
            goal=PromotionGoal.max,
            value=value,
            baseline_value=baseline_value,
            min_improvement=1.0,
        ),
        run_id=run_id,
        manifest=ArtifactRef(
            kind="run_manifest",
            path=str(baseline_manifest_path),
            sha256=sha256_file(baseline_manifest_path),
        ),
        baseline_run_id="gate2-20260422T135752Z-replay6p5",
        baseline_manifest=ArtifactRef(
            kind="run_manifest",
            path=str(_run_dir("gate2-20260422T135752Z-replay6p5") / "manifest.json"),
            sha256=sha256_file(_run_dir("gate2-20260422T135752Z-replay6p5") / "manifest.json"),
        ),
    )


def test_next_experiment_report_round_trips_strictly() -> None:
    report = NextExperimentReport(
        run_id="gate2-candidate",
        gate_id="gate_2_perception_servo",
        generated_at_utc="2026-04-23T00:00:00Z",
        decision=NextExperimentDecision.iterate,
        objective="Improve final-centimeter insertion.",
        candidates=(
            NextExperimentCandidate(
                candidate_id="build_adaptive_final_centimeter_primitive",
                kind=NextExperimentCandidateKind.controller_development,
                priority=NextExperimentPriority.high,
                action=NextExperimentAction.build,
                title="Build adaptive final-centimeter insertion primitive",
                rationale="No insertion credit was earned.",
                evidence_labels=("no_partial_or_full_insertion",),
                evidence_terms=("official.score.total",),
                acceptance_checks=("Uses legal observations only.",),
                runtime_notes="Must run through aic_model.",
                leakage_notes="Labels stay offline.",
            ),
        ),
        notes=("report only",),
    )

    assert NextExperimentReport.from_dict(report.to_dict()) == report


def test_next_experiment_plan_derives_executable_offline_candidates(tmp_path: Path) -> None:
    score = _score_report()
    manifest = _manifest_with_runtime_env(
        score,
        {
            "AIC_LEWM_FINAL_SERVO_ENABLED": "1",
            "AIC_LEWM_SFP_FINAL_SERVO_LINEAR": "0.02,0,0",
            "AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE": "delta",
            "AIC_LEWM_FINAL_SERVO_FORCE_GUARD_N": "21",
        },
    )
    next_experiment = derive_next_experiment_report(
        manifest,
        reward_report=_reward_report(total=score.total),
        failure_report=_failure_report(),
        promotion=_promotion_decision(value=score.total, baseline_value=score.total - 0.2),
        gate_id="gate_2_perception_servo",
        generated_at_utc="2026-04-23T00:00:00Z",
    )
    manifest_artifact, next_experiment_artifact = _write_source_artifacts(
        tmp_path,
        manifest=manifest,
        next_experiment=next_experiment,
    )

    plan = derive_next_experiment_plan(
        next_experiment,
        manifest=manifest,
        source_next_experiment=next_experiment_artifact,
        source_manifest=manifest_artifact,
        generated_at_utc="2026-04-23T00:00:01Z",
    )

    assert NextExperimentPlan.from_dict(plan.to_dict()) == plan
    assert plan.generated_at_utc == "2026-04-23T00:00:01Z"
    candidates_by_id = {candidate.candidate_id: candidate for candidate in plan.candidates}
    assert candidates_by_id["reject_fixed_lateral_bias_push"].status is NextExperimentPlanStatus.no_launch
    assert candidates_by_id["reject_fixed_lateral_bias_push"].experiment is None
    assert candidates_by_id["build_adaptive_final_centimeter_primitive"].status is (
        NextExperimentPlanStatus.requires_implementation
    )
    assert candidates_by_id["build_adaptive_final_centimeter_primitive"].experiment is None
    gate_plan = candidates_by_id["add_offline_final_centimeter_acceptance_gate"]
    assert gate_plan.status is NextExperimentPlanStatus.launchable
    assert gate_plan.execution_mode is NextExperimentExecutionMode.offline_gate
    assert gate_plan.launchable is True
    assert gate_plan.autonomous_launch_allowed is False
    assert gate_plan.experiment is not None
    assert gate_plan.experiment.backend.runtime_allowed is False
    assert gate_plan.experiment.backend.config["candidate_id"] == gate_plan.candidate_id
    assert gate_plan.experiment.expected_artifacts[0].kind == "offline_acceptance_gate_report"


def test_next_experiment_plan_rejects_unbound_sources_and_duplicates(tmp_path: Path) -> None:
    score = _score_report()
    manifest = _manifest(score)
    next_experiment = derive_next_experiment_report(
        manifest,
        reward_report=RewardReport(
            run_id="gate2-candidate",
            generated_at_utc="2026-04-23T00:00:00Z",
            terms=(
                RewardTerm(
                    name="official.score.total",
                    value=score.total,
                    signal_kind=RewardSignalKind.official_score,
                    leakage_class=LeakageClass.privileged_eval_signal,
                    source=score.source,
                ),
            ),
        ),
        failure_report=FailureReport(
            run_id="gate2-candidate",
            generated_at_utc="2026-04-23T00:00:00Z",
            labels=(
                FailureLabel(
                    kind=FailureKind.no_partial_or_full_insertion,
                    severity=FailureSeverity.blocker,
                    summary="Tier 3 evidence is proximity-only.",
                    source=score.source,
                    leakage_class=LeakageClass.privileged_eval_signal,
                    evidence={"max_tier_3": 25.0},
                ),
            ),
        ),
        gate_id="gate_2_perception_servo",
    )
    manifest_artifact, next_experiment_artifact = _write_source_artifacts(
        tmp_path,
        manifest=manifest,
        next_experiment=next_experiment,
    )
    uri_only_next_experiment = ArtifactRef(
        kind="next_experiment_report",
        uri="memory://pytest/next_experiment.json",
        sha256=next_experiment_artifact.sha256,
        provenance={
            "producer": "pytest",
            "run_id": next_experiment.run_id,
            "gate_id": next_experiment.gate_id,
            "derivation": "derive_next_experiment_report",
        },
    )
    with pytest.raises(HarnessIOError, match="local byte-verifiable"):
        derive_next_experiment_plan(
            next_experiment,
            manifest=manifest,
            source_next_experiment=uri_only_next_experiment,
            source_manifest=manifest_artifact,
        )

    plan = derive_next_experiment_plan(
        next_experiment,
        manifest=manifest,
        source_next_experiment=next_experiment_artifact,
        source_manifest=manifest_artifact,
    )
    payload = plan.to_dict()
    payload["candidates"].append(dict(payload["candidates"][0]))
    with pytest.raises(HarnessIOError, match="duplicate plan_id"):
        NextExperimentPlan.from_dict(payload)

    launch_payload = plan.to_dict()
    launch_payload["candidates"][0]["autonomous_launch_allowed"] = True
    with pytest.raises(HarnessIOError, match="autonomous_launch_allowed"):
        NextExperimentPlan.from_dict(launch_payload)

    blocked_payload = plan.to_dict()
    launchable_experiment = next(
        candidate.experiment for candidate in plan.candidates if candidate.experiment is not None
    )
    blocked_payload["candidates"][0] = {
        **blocked_payload["candidates"][0],
        "status": "blocked",
        "blocked_by_labels": ["inconclusive_missing_evidence"],
        "experiment": launchable_experiment.to_dict(),
        "launchable": False,
    }
    with pytest.raises(HarnessIOError, match="blocked.*must not include experiment"):
        NextExperimentPlan.from_dict(blocked_payload)


def test_next_experiment_plan_rejects_source_content_mismatch(tmp_path: Path) -> None:
    score = _score_report()
    manifest = _manifest(score)
    next_experiment = derive_next_experiment_report(
        manifest,
        reward_report=RewardReport(
            run_id="gate2-candidate",
            generated_at_utc="2026-04-23T00:00:00Z",
            terms=(
                RewardTerm(
                    name="official.score.total",
                    value=score.total,
                    signal_kind=RewardSignalKind.official_score,
                    leakage_class=LeakageClass.privileged_eval_signal,
                    source=score.source,
                ),
            ),
        ),
        failure_report=FailureReport(
            run_id="gate2-candidate",
            generated_at_utc="2026-04-23T00:00:00Z",
            labels=(
                FailureLabel(
                    kind=FailureKind.no_partial_or_full_insertion,
                    severity=FailureSeverity.blocker,
                    summary="Tier 3 evidence is proximity-only.",
                    source=score.source,
                    leakage_class=LeakageClass.privileged_eval_signal,
                    evidence={"max_tier_3": 25.0},
                ),
            ),
        ),
        gate_id="gate_2_perception_servo",
    )
    manifest_artifact, next_experiment_artifact = _write_source_artifacts(
        tmp_path,
        manifest=manifest,
        next_experiment=next_experiment,
    )
    contradictory_report = NextExperimentReport(
        run_id=next_experiment.run_id,
        gate_id=next_experiment.gate_id,
        generated_at_utc=next_experiment.generated_at_utc,
        decision=next_experiment.decision,
        objective=next_experiment.objective,
        candidates=tuple(reversed(next_experiment.candidates)),
        notes=next_experiment.notes,
    )
    with pytest.raises(HarnessIOError, match="does not match supplied next_experiment"):
        derive_next_experiment_plan(
            contradictory_report,
            manifest=manifest,
            source_next_experiment=next_experiment_artifact,
            source_manifest=manifest_artifact,
        )

    original_plan = derive_next_experiment_plan(
        next_experiment,
        manifest=manifest,
        source_next_experiment=next_experiment_artifact,
        source_manifest=manifest_artifact,
    )
    first_candidate = next_experiment.candidates[0]
    retitled_candidate = NextExperimentCandidate(
        candidate_id=first_candidate.candidate_id,
        kind=first_candidate.kind,
        priority=first_candidate.priority,
        action=first_candidate.action,
        title=first_candidate.title + " v2",
        rationale=first_candidate.rationale,
        evidence_labels=first_candidate.evidence_labels,
        evidence_terms=first_candidate.evidence_terms,
        evidence_manifest_fields=first_candidate.evidence_manifest_fields,
        acceptance_checks=first_candidate.acceptance_checks,
        blocked_by_labels=first_candidate.blocked_by_labels,
        runtime_notes=first_candidate.runtime_notes,
        leakage_notes=first_candidate.leakage_notes,
    )
    retitled_report = NextExperimentReport(
        run_id=next_experiment.run_id,
        gate_id=next_experiment.gate_id,
        generated_at_utc=next_experiment.generated_at_utc,
        decision=next_experiment.decision,
        objective=next_experiment.objective,
        candidates=(retitled_candidate, *next_experiment.candidates[1:]),
        notes=next_experiment.notes,
    )
    retitled_manifest_artifact, retitled_next_artifact = _write_source_artifacts(
        tmp_path / "retitled",
        manifest=manifest,
        next_experiment=retitled_report,
    )
    retitled_plan = derive_next_experiment_plan(
        retitled_report,
        manifest=manifest,
        source_next_experiment=retitled_next_artifact,
        source_manifest=retitled_manifest_artifact,
    )
    assert original_plan.candidates[0].candidate_id == retitled_plan.candidates[0].candidate_id
    assert original_plan.candidates[0].plan_id != retitled_plan.candidates[0].plan_id

    manifest_variant = RunManifest(
        run_id=manifest.run_id,
        status=manifest.status,
        backend=manifest.backend,
        created_at_utc=manifest.created_at_utc,
        updated_at_utc=manifest.updated_at_utc,
        artifacts=manifest.artifacts,
        score=manifest.score,
        experiment_id=manifest.experiment_id,
        hypothesis="Same recommendation against a changed manifest.",
        notes=manifest.notes,
    )
    variant_manifest_artifact, variant_next_artifact = _write_source_artifacts(
        tmp_path / "manifest_variant",
        manifest=manifest_variant,
        next_experiment=next_experiment,
    )
    variant_plan = derive_next_experiment_plan(
        next_experiment,
        manifest=manifest_variant,
        source_next_experiment=variant_next_artifact,
        source_manifest=variant_manifest_artifact,
    )
    assert original_plan.candidates[0].candidate_id == variant_plan.candidates[0].candidate_id
    assert original_plan.candidates[0].plan_id != variant_plan.candidates[0].plan_id


def test_next_experiment_report_rejects_unknown_nested_fields() -> None:
    with pytest.raises(HarnessIOError, match="next experiment candidate has unknown fields"):
        NextExperimentReport.from_dict(
            {
                "schema_version": 1,
                "run_id": "gate2-candidate",
                "gate_id": "gate_2_perception_servo",
                "generated_at_utc": "2026-04-23T00:00:00Z",
                "decision": "iterate",
                "objective": None,
                "candidates": [
                    {
                        "candidate_id": "candidate-a",
                        "kind": "controller_development",
                        "priority": "high",
                        "action": "build",
                        "title": "Build candidate",
                        "rationale": "Need a better primitive.",
                        "unexpected": True,
                    }
                ],
                "notes": [],
            }
        )


def test_next_experiment_candidate_requires_machine_readable_evidence() -> None:
    with pytest.raises(HarnessIOError, match="must cite evidence_labels"):
        NextExperimentCandidate(
            candidate_id="candidate-a",
            kind=NextExperimentCandidateKind.failure_analysis,
            priority=NextExperimentPriority.medium,
            action=NextExperimentAction.analyze,
            title="Inspect failure",
            rationale="The report should not be prose-only.",
        )


def test_next_experiment_report_rejects_duplicate_candidate_ids() -> None:
    candidate = NextExperimentCandidate(
        candidate_id="candidate-a",
        kind=NextExperimentCandidateKind.failure_analysis,
        priority=NextExperimentPriority.medium,
        action=NextExperimentAction.analyze,
        title="Inspect failure",
        rationale="A classified failure needs inspection.",
        evidence_labels=("no_partial_or_full_insertion",),
    )

    with pytest.raises(HarnessIOError, match="duplicate candidate_id"):
        NextExperimentReport(
            run_id="gate2-candidate",
            gate_id="gate_2_perception_servo",
            generated_at_utc="2026-04-23T00:00:00Z",
            decision=NextExperimentDecision.iterate,
            candidates=(candidate, candidate),
        )


def test_derive_next_experiment_report_rejects_run_id_mismatches() -> None:
    score = _score_report()
    with pytest.raises(HarnessIOError, match="reward_report.run_id does not match"):
        derive_next_experiment_report(
            _manifest(score),
            reward_report=_reward_report(run_id="other-run"),
            failure_report=_failure_report(),
            promotion=_promotion_decision(value=score.total, baseline_value=score.total - 0.2),
            gate_id="gate_2_perception_servo",
        )
    with pytest.raises(HarnessIOError, match="missing promotion-derived labels"):
        derive_next_experiment_report(
            _manifest(score),
            reward_report=_reward_report(),
            failure_report=FailureReport(
                run_id="gate2-candidate",
                generated_at_utc="2026-04-23T00:00:00Z",
                labels=(),
            ),
            promotion=_promotion_decision(value=score.total, baseline_value=score.total - 0.2),
            gate_id="gate_2_perception_servo",
        )
    with pytest.raises(HarnessIOError, match="failure_report.run_id does not match"):
        derive_next_experiment_report(
            _manifest(score),
            reward_report=_reward_report(),
            failure_report=_failure_report(run_id="other-run"),
            promotion=_promotion_decision(value=score.total, baseline_value=score.total - 0.2),
            gate_id="gate_2_perception_servo",
        )
    with pytest.raises(HarnessIOError, match="promotion.run_id does not match"):
        derive_next_experiment_report(
            _manifest(score),
            reward_report=_reward_report(),
            failure_report=_failure_report(),
            promotion=_promotion_decision(value=score.total, baseline_value=score.total - 0.2, run_id="other-run"),
            gate_id="gate_2_perception_servo",
        )


def test_derive_next_experiment_report_rejects_missing_required_evidence() -> None:
    score = _score_report()
    with pytest.raises(HarnessIOError, match="requires reward terms"):
        derive_next_experiment_report(
            _manifest(score),
            reward_report=RewardReport(
                run_id="gate2-candidate",
                generated_at_utc="2026-04-23T00:00:00Z",
                terms=(),
            ),
            failure_report=_failure_report(),
            promotion=_promotion_decision(value=score.total, baseline_value=score.total - 0.2),
            gate_id="gate_2_perception_servo",
        )


def test_derive_next_experiment_report_defaults_to_reward_report_timestamp() -> None:
    score = _score_report()
    report = derive_next_experiment_report(
        _manifest(score),
        reward_report=RewardReport(
            run_id="gate2-candidate",
            generated_at_utc="2026-04-23T01:02:03Z",
            terms=(
                RewardTerm(
                    name="official.score.total",
                    value=score.total,
                    signal_kind=RewardSignalKind.official_score,
                    leakage_class=LeakageClass.privileged_eval_signal,
                    source=score.source,
                ),
            ),
        ),
        failure_report=FailureReport(
            run_id="gate2-candidate",
            generated_at_utc="2026-04-23T04:05:06Z",
            labels=(
                FailureLabel(
                    kind=FailureKind.no_partial_or_full_insertion,
                    severity=FailureSeverity.blocker,
                    summary="Tier 3 evidence is proximity-only.",
                    source=score.source,
                    leakage_class=LeakageClass.privileged_eval_signal,
                    evidence={"max_tier_3": 25.0},
                ),
            ),
        ),
        gate_id="gate_2_perception_servo",
    )

    assert report.generated_at_utc == "2026-04-23T01:02:03Z"


def test_derive_next_experiment_report_rejects_missing_score_without_missing_evidence_label() -> None:
    with pytest.raises(HarnessIOError, match="requires manifest.score"):
        derive_next_experiment_report(
            _manifest(None),
            reward_report=_reward_report(),
            failure_report=_failure_report(),
            gate_id="gate_2_perception_servo",
        )


def test_derive_next_experiment_report_requires_official_total_reward_term() -> None:
    score = _score_report()
    with pytest.raises(HarnessIOError, match="requires exactly one official.score.total"):
        derive_next_experiment_report(
            _manifest(score),
            reward_report=RewardReport(
                run_id="gate2-candidate",
                generated_at_utc="2026-04-23T00:00:00Z",
                terms=(
                    RewardTerm(
                        name="promotion.metric.margin_gap",
                        value=2.0,
                        signal_kind=RewardSignalKind.diagnostic,
                        leakage_class=LeakageClass.privileged_eval_signal,
                        source="promotion_decision",
                    ),
                ),
            ),
            failure_report=_failure_report(),
            promotion=_promotion_decision(value=score.total, baseline_value=score.total - 0.2),
            gate_id="gate_2_perception_servo",
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("value", "value does not match"),
        ("source", "source does not match"),
        ("signal_kind", "signal_kind official_score"),
        ("leakage_class", "leakage_class privileged_eval_signal"),
    ),
)
def test_derive_next_experiment_report_rejects_contradictory_official_total_reward_term(
    mutation: str,
    match: str,
) -> None:
    score = _score_report()
    value = score.total
    source = score.source
    signal_kind = RewardSignalKind.official_score
    leakage_class = LeakageClass.privileged_eval_signal
    if mutation == "value":
        value = score.total - 1.0
    elif mutation == "source":
        source = "/tmp/eval/other-scoring.yaml"
    elif mutation == "signal_kind":
        signal_kind = RewardSignalKind.diagnostic
    elif mutation == "leakage_class":
        leakage_class = LeakageClass.post_hoc_label

    with pytest.raises(HarnessIOError, match=match):
        derive_next_experiment_report(
            _manifest(score),
            reward_report=RewardReport(
                run_id="gate2-candidate",
                generated_at_utc="2026-04-23T00:00:00Z",
                terms=(
                    RewardTerm(
                        name="official.score.total",
                        value=value,
                        signal_kind=signal_kind,
                        leakage_class=leakage_class,
                        source=source,
                    ),
                ),
            ),
            failure_report=FailureReport(
                run_id="gate2-candidate",
                generated_at_utc="2026-04-23T00:00:00Z",
                labels=(),
            ),
            promotion=_accepted_promotion_decision(value=score.total, baseline_value=score.total - 2.0),
            gate_id="gate_2_perception_servo",
        )


def test_derive_next_experiment_report_blocks_on_missing_score_evidence() -> None:
    report = derive_next_experiment_report(
        _manifest(None),
        reward_report=RewardReport(
            run_id="gate2-candidate",
            generated_at_utc="2026-04-23T00:00:00Z",
            terms=(),
        ),
        failure_report=FailureReport(
            run_id="gate2-candidate",
            generated_at_utc="2026-04-23T00:00:00Z",
            labels=(
                FailureLabel(
                    kind=FailureKind.inconclusive_missing_evidence,
                    severity=FailureSeverity.blocker,
                    summary="Missing score evidence.",
                    source="run_manifest",
                    leakage_class=LeakageClass.post_hoc_label,
                    evidence={"missing": "score"},
                ),
            ),
        ),
        gate_id="gate_2_perception_servo",
        generated_at_utc="2026-04-23T00:00:00Z",
    )

    assert report.decision is NextExperimentDecision.block
    assert tuple(candidate.candidate_id for candidate in report.candidates) == (
        "repair_missing_evidence",
    )


def test_derive_next_experiment_report_rejects_mismatched_promotion_score() -> None:
    score = _score_report()
    with pytest.raises(HarnessIOError, match="promotion metric.value does not match"):
        derive_next_experiment_report(
            _manifest(score),
            reward_report=_reward_report(),
            failure_report=_failure_report(),
            promotion=_promotion_decision(value=score.total - 10.0, baseline_value=score.total),
            gate_id="gate_2_perception_servo",
        )


def test_derive_next_experiment_report_rejects_promotion_evidence_without_promotion() -> None:
    score = _score_report()
    with pytest.raises(HarnessIOError, match="PromotionDecision"):
        derive_next_experiment_report(
            _manifest(score),
            reward_report=_reward_report(total=score.total),
            failure_report=_failure_report(),
            gate_id="gate_2_perception_servo",
        )


def test_derive_next_experiment_report_rejects_score_regression_without_promotion() -> None:
    score = _score_report()
    with pytest.raises(HarnessIOError, match="promotion failure labels require"):
        derive_next_experiment_report(
            _manifest(score),
            reward_report=RewardReport(
                run_id="gate2-candidate",
                generated_at_utc="2026-04-23T00:00:00Z",
                terms=(
                    RewardTerm(
                        name="official.score.total",
                        value=score.total,
                        signal_kind=RewardSignalKind.official_score,
                        leakage_class=LeakageClass.privileged_eval_signal,
                        source=score.source,
                    ),
                ),
            ),
            failure_report=FailureReport(
                run_id="gate2-candidate",
                generated_at_utc="2026-04-23T00:00:00Z",
                labels=(
                    FailureLabel(
                        kind=FailureKind.score_regression,
                        severity=FailureSeverity.blocker,
                        summary="Candidate official score regressed.",
                        source="promotion_decision",
                        leakage_class=LeakageClass.privileged_eval_signal,
                        evidence={"margin_gap": -1.0},
                    ),
                ),
            ),
            gate_id="gate_2_perception_servo",
        )


def test_derive_next_experiment_report_rejects_promotion_failure_label_contradictions() -> None:
    score = _score_report()
    with pytest.raises(HarnessIOError, match="contradict promotion decision"):
        derive_next_experiment_report(
            _manifest(score),
            reward_report=_reward_report(
                total=score.total,
                promotion_improvement=2.0,
                promotion_margin_gap=1.0,
            ),
            failure_report=_failure_report(),
            promotion=_accepted_promotion_decision(value=score.total, baseline_value=score.total - 2.0),
            gate_id="gate_2_perception_servo",
        )


def test_derive_next_experiment_report_rejects_malformed_promotion_failure_evidence() -> None:
    score = _score_report()
    failure = _failure_report()
    bad_label = FailureLabel(
        kind=FailureKind.below_promotion_margin,
        severity=FailureSeverity.blocker,
        summary="Candidate did not clear the margin.",
        source="promotion_decision",
        leakage_class=LeakageClass.privileged_eval_signal,
        evidence={**failure.labels[1].evidence, "margin_gap": -0.7},
    )

    with pytest.raises(HarnessIOError, match="evidence.margin_gap"):
        derive_next_experiment_report(
            _manifest(score),
            reward_report=_reward_report(),
            failure_report=FailureReport(
                run_id="gate2-candidate",
                generated_at_utc="2026-04-23T00:00:00Z",
                labels=(failure.labels[0], bad_label),
            ),
            promotion=_promotion_decision(value=score.total, baseline_value=score.total - 0.2),
            gate_id="gate_2_perception_servo",
        )


def test_derive_next_experiment_report_rejects_duplicate_promotion_failure_labels() -> None:
    score = _score_report()
    failure = _failure_report()

    with pytest.raises(HarnessIOError, match="duplicate promotion-derived"):
        derive_next_experiment_report(
            _manifest(score),
            reward_report=_reward_report(),
            failure_report=FailureReport(
                run_id="gate2-candidate",
                generated_at_utc="2026-04-23T00:00:00Z",
                labels=(failure.labels[0], failure.labels[1], failure.labels[1]),
            ),
            promotion=_promotion_decision(value=score.total, baseline_value=score.total - 0.2),
            gate_id="gate_2_perception_servo",
        )


def test_derive_next_experiment_report_rejects_contradictory_promotion_reward_terms() -> None:
    score = _score_report()
    with pytest.raises(HarnessIOError, match="promotion.metric.margin_gap"):
        derive_next_experiment_report(
            _manifest(score),
            reward_report=_reward_report(
                total=score.total,
                promotion_improvement=2.0,
                promotion_margin_gap=-0.8,
            ),
            failure_report=FailureReport(
                run_id="gate2-candidate",
                generated_at_utc="2026-04-23T00:00:00Z",
                labels=(),
            ),
            promotion=_accepted_promotion_decision(value=score.total, baseline_value=score.total - 2.0),
            gate_id="gate_2_perception_servo",
        )


def test_derive_next_experiment_report_requires_promotion_reward_terms() -> None:
    score = _score_report()
    with pytest.raises(HarnessIOError, match="promotion.metric.improvement"):
        derive_next_experiment_report(
            _manifest(score),
            reward_report=RewardReport(
                run_id="gate2-candidate",
                generated_at_utc="2026-04-23T00:00:00Z",
                terms=(
                    RewardTerm(
                        name="official.score.total",
                        value=score.total,
                        signal_kind=RewardSignalKind.official_score,
                        leakage_class=LeakageClass.privileged_eval_signal,
                        source=score.source,
                    ),
                ),
            ),
            failure_report=FailureReport(
                run_id="gate2-candidate",
                generated_at_utc="2026-04-23T00:00:00Z",
                labels=(),
            ),
            promotion=_accepted_promotion_decision(value=score.total, baseline_value=score.total - 2.0),
            gate_id="gate_2_perception_servo",
        )


def test_derive_next_experiment_report_rejects_nonofficial_promotion_metric() -> None:
    score = _score_report()
    promotion = PromotionDecision(
        decided_at_utc="2026-04-23T00:52:15Z",
        decision=PromotionDecisionKind.accepted,
        accepted=True,
        eligible_for_submission=True,
        reason="diagnostic metric improved",
        metric=PromotionMetric(
            name="diagnostic.reward.total",
            goal=PromotionGoal.max,
            value=10.0,
            baseline_value=8.0,
            min_improvement=1.0,
        ),
        run_id="gate2-candidate",
        manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-candidate/manifest.json",
            sha256="b" * 64,
        ),
        baseline_run_id="gate2-baseline",
        baseline_manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-baseline/manifest.json",
            sha256="c" * 64,
        ),
    )

    with pytest.raises(HarnessIOError, match="must use evaluation.score.total"):
        derive_next_experiment_report(
            _manifest(score),
            reward_report=_reward_report(),
            failure_report=_failure_report(),
            promotion=promotion,
            gate_id="gate_2_perception_servo",
        )


def test_derive_next_experiment_report_rejects_min_goal_official_score_promotion() -> None:
    score = _score_report()
    promotion = PromotionDecision(
        decided_at_utc="2026-04-23T00:52:15Z",
        decision=PromotionDecisionKind.accepted,
        accepted=True,
        eligible_for_submission=True,
        reason="lower official score should not be promotable for AIC",
        metric=PromotionMetric(
            name="evaluation.score.total",
            goal=PromotionGoal.min,
            value=score.total,
            baseline_value=score.total + 2.0,
            min_improvement=1.0,
        ),
        run_id="gate2-candidate",
        manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-candidate/manifest.json",
            sha256="b" * 64,
        ),
        baseline_run_id="gate2-baseline",
        baseline_manifest=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-baseline/manifest.json",
            sha256="c" * 64,
        ),
    )

    with pytest.raises(HarnessIOError, match="must use goal=max"):
        derive_next_experiment_report(
            _manifest(score),
            reward_report=_reward_report(
                total=score.total,
                promotion_improvement=2.0,
                promotion_margin_gap=1.0,
            ),
            failure_report=FailureReport(
                run_id="gate2-candidate",
                generated_at_utc="2026-04-23T00:00:00Z",
                labels=(),
            ),
            promotion=promotion,
            gate_id="gate_2_perception_servo",
        )


def test_derive_next_experiment_report_promotes_clean_accepted_candidate() -> None:
    score = _score_report()

    report = derive_next_experiment_report(
        _manifest(score),
        reward_report=_reward_report(
            total=score.total,
            promotion_improvement=2.0,
            promotion_margin_gap=1.0,
        ),
        failure_report=FailureReport(
            run_id="gate2-candidate",
            generated_at_utc="2026-04-23T00:00:00Z",
            labels=(),
        ),
        promotion=_accepted_promotion_decision(value=score.total, baseline_value=score.total - 2.0),
        gate_id="gate_2_perception_servo",
        generated_at_utc="2026-04-23T00:00:00Z",
    )

    assert report.decision is NextExperimentDecision.promote
    assert tuple(candidate.candidate_id for candidate in report.candidates) == (
        "prepare_promoted_candidate_review",
    )
    assert report.candidates[0].evidence_terms == (
        "official.score.total",
        "promotion.metric.improvement",
        "promotion.metric.margin_gap",
    )


def test_derive_next_experiment_report_keeps_accepted_candidate_with_failures_in_iteration() -> None:
    score = _score_report()
    report = derive_next_experiment_report(
        _manifest(score),
        reward_report=_reward_report(
            total=score.total,
            promotion_improvement=2.0,
            promotion_margin_gap=1.0,
        ),
        failure_report=FailureReport(
            run_id="gate2-candidate",
            generated_at_utc="2026-04-23T00:00:00Z",
            labels=(
                FailureLabel(
                    kind=FailureKind.off_limit_contact,
                    severity=FailureSeverity.blocker,
                    summary="Off-limit contact still needs investigation.",
                    source="/tmp/eval/bag_trial_1_0.mcap",
                    leakage_class=LeakageClass.privileged_eval_signal,
                    evidence={"off_limit_contact_count": 1},
                ),
            ),
        ),
        promotion=_accepted_promotion_decision(value=score.total, baseline_value=score.total - 2.0),
        gate_id="gate_2_perception_servo",
        generated_at_utc="2026-04-23T00:00:00Z",
    )

    assert report.decision is NextExperimentDecision.iterate
    assert tuple(candidate.candidate_id for candidate in report.candidates) == (
        "mine_contact_failure_window",
    )


def test_derive_next_experiment_report_does_not_prepare_promotion_with_unmatched_failure() -> None:
    score = _score_report()
    report = derive_next_experiment_report(
        _manifest(score),
        reward_report=_reward_report(
            total=score.total,
            promotion_improvement=2.0,
            promotion_margin_gap=1.0,
        ),
        failure_report=FailureReport(
            run_id="gate2-candidate",
            generated_at_utc="2026-04-23T00:00:00Z",
            labels=(
                FailureLabel(
                    kind=FailureKind.contact_guard_stop_recommended,
                    severity=FailureSeverity.warning,
                    summary="Stop recommendation needs classification before promotion review.",
                    source="/tmp/eval/bag_trial_1_0.mcap",
                    leakage_class=LeakageClass.privileged_eval_signal,
                    evidence={"recommended_stop_sec": 0.15},
                ),
            ),
        ),
        promotion=_accepted_promotion_decision(value=score.total, baseline_value=score.total - 2.0),
        gate_id="gate_2_perception_servo",
        generated_at_utc="2026-04-23T00:00:00Z",
    )

    assert report.decision is NextExperimentDecision.iterate
    assert tuple(candidate.candidate_id for candidate in report.candidates) == (
        "inspect_unclassified_failure_state",
    )


def test_derive_next_experiment_report_recommends_gate2_iteration() -> None:
    score = _score_report()
    report = derive_next_experiment_report(
        _manifest(score),
        reward_report=_reward_report(),
        failure_report=_failure_report(),
        promotion=_promotion_decision(value=score.total, baseline_value=score.total - 0.2),
        gate_id="gate_2_perception_servo",
        objective="Replace brittle replay drift with legal visual alignment and force-aware insertion control.",
        generated_at_utc="2026-04-23T00:53:04Z",
    )

    assert report.decision is NextExperimentDecision.iterate
    candidate_ids = {candidate.candidate_id for candidate in report.candidates}
    assert "reject_fixed_lateral_bias_push" not in candidate_ids
    assert "keep_baseline_relative_force_guard" not in candidate_ids
    assert "build_adaptive_final_centimeter_primitive" in candidate_ids
    assert "add_offline_final_centimeter_acceptance_gate" in candidate_ids
    adaptive = next(
        candidate
        for candidate in report.candidates
        if candidate.candidate_id == "build_adaptive_final_centimeter_primitive"
    )
    assert "no_partial_or_full_insertion" in adaptive.evidence_labels
    assert "promotion.metric.margin_gap" in adaptive.evidence_terms
    assert "legal observations" in " ".join(adaptive.acceptance_checks)


def test_derive_next_experiment_report_requires_enabled_final_servo_for_runtime_primitive_claims() -> None:
    score = _score_report()
    report = derive_next_experiment_report(
        _manifest_with_runtime_env(
            score,
            {
                "AIC_LEWM_FINAL_SERVO_ENABLED": "0",
                "AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE": "delta",
                "AIC_LEWM_FINAL_SERVO_FORCE_GUARD_N": "4",
                "AIC_LEWM_SFP_FINAL_SERVO_LINEAR": "-0.01,0.02,-0.012",
                "AIC_LEWM_SC_FINAL_SERVO_LINEAR": "0,0,0",
            },
        ),
        reward_report=_reward_report(),
        failure_report=_failure_report(),
        promotion=_promotion_decision(value=score.total, baseline_value=score.total - 0.2),
        gate_id="gate_2_perception_servo",
        generated_at_utc="2026-04-23T00:53:04Z",
    )

    candidate_ids = {candidate.candidate_id for candidate in report.candidates}
    assert "reject_fixed_lateral_bias_push" not in candidate_ids
    assert "keep_baseline_relative_force_guard" not in candidate_ids
    assert "build_adaptive_final_centimeter_primitive" in candidate_ids


def test_real_latbias_backfill_derives_next_experiment_report(tmp_path: Path) -> None:
    backfill = backfill_legacy_replay_policy_eval(
        _run_dir("gate2-20260422T230909Z-latbias"),
        manifest_output_path=tmp_path / "gate2-20260422T230909Z-latbias.run_manifest.json",
        baseline=_historical_baseline_seed(
            run_id="gate2-20260422T172605Z-scstop2p75",
            value=127.78018667287431,
            baseline_value=122.39001511101944,
        ),
        repo_root=REPO_ROOT,
    )
    assert backfill.promotion_decision is not None
    reward_failure = derive_reward_failure_reports(
        backfill.manifest,
        promotion=backfill.promotion_decision,
        generated_at_utc="2026-04-23T00:53:04Z",
    )

    report = derive_next_experiment_report(
        backfill.manifest,
        reward_report=reward_failure.reward_report,
        failure_report=reward_failure.failure_report,
        promotion=backfill.promotion_decision,
        gate_id="gate_2_perception_servo",
        objective="Replace brittle replay drift with legal visual alignment and force-aware insertion control.",
        generated_at_utc="2026-04-23T00:53:04Z",
    )

    assert report.run_id == "gate2-20260422T230909Z-latbias"
    assert report.decision is NextExperimentDecision.iterate
    candidate_ids = {candidate.candidate_id for candidate in report.candidates}
    assert "reject_fixed_lateral_bias_push" in candidate_ids
    assert "keep_baseline_relative_force_guard" in candidate_ids
    assert "build_adaptive_final_centimeter_primitive" in candidate_ids
    assert "add_offline_final_centimeter_acceptance_gate" in candidate_ids
    fixed_rejection = next(
        candidate
        for candidate in report.candidates
        if candidate.candidate_id == "reject_fixed_lateral_bias_push"
    )
    assert (
        "backend.provenance.legacy_runtime_env.AIC_LEWM_FINAL_SERVO_ENABLED"
        in fixed_rejection.evidence_manifest_fields
    )
    assert (
        "backend.provenance.legacy_runtime_env.AIC_LEWM_SFP_FINAL_SERVO_LINEAR"
        in fixed_rejection.evidence_manifest_fields
    )
    force_guard = next(
        candidate
        for candidate in report.candidates
        if candidate.candidate_id == "keep_baseline_relative_force_guard"
    )
    assert force_guard.action is NextExperimentAction.keep
    assert force_guard.evidence_manifest_fields == (
        "backend.provenance.legacy_runtime_env.AIC_LEWM_FINAL_SERVO_ENABLED",
        "backend.provenance.legacy_runtime_env.AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE",
        "backend.provenance.legacy_runtime_env.AIC_LEWM_FINAL_SERVO_FORCE_GUARD_N",
    )
    assert "live guard may only consume legal runtime observations" in force_guard.leakage_notes
