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
    McapEvalBundleReport,
    McapEvalFirstContact,
    McapEvalTaskHints,
    McapEvalTrialReport,
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
    derive_reward_failure_reports,
    sha256_file,
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


def _score_report(*, tier_3: tuple[float, float, float] = (25.0, 25.0, 6.0)) -> ScoreReport:
    trials = {
        f"trial_{index}": TrialScore(
            total=1.0 + 22.0 + tier,
            tier_1=1.0,
            tier_2=22.0,
            tier_3=tier,
        )
        for index, tier in enumerate(tier_3, start=1)
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


def _promotion_decision(*, value: float = 128.0, baseline_value: float = 127.8) -> PromotionDecision:
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


def _latbias_mcap_bundle() -> McapEvalBundleReport:
    return McapEvalBundleReport(
        source=str(
            REPO_ROOT
            / "artifacts"
            / "cloud-logs"
            / "gate2-20260422T230909Z-latbias"
            / "eval"
        ),
        analyzed_at_utc="2026-04-23T00:53:04Z",
        contact_margin_sec=0.25,
        stop_step_sec=0.05,
        recommended_env={},
        trials=(
            McapEvalTrialReport(
                trial_id="trial_1",
                source=str(
                    REPO_ROOT
                    / "artifacts"
                    / "cloud-logs"
                    / "gate2-20260422T230909Z-latbias"
                    / "eval"
                    / "bag_trial_1_20260422_233452_230"
                    / "bag_trial_1_20260422_233452_230_0.mcap"
                ),
                size_bytes=1,
                sha256="1" * 64,
                controller_state_count=4090,
                pose_command_count=37617,
                off_limit_contact_count=0,
                controller_stamp_start_sec=2.65,
                controller_stamp_end_sec=10.848,
                controller_duration_sec=8.198,
                final_tcp_position=(-0.39314625574579115, 0.23708652367391764, 0.18003685383572243),
                final_tcp_error=(-0.0002884982328384966, -0.0038959342653075713, -0.028664984921960596),
            ),
            McapEvalTrialReport(
                trial_id="trial_2",
                source=str(
                    REPO_ROOT
                    / "artifacts"
                    / "cloud-logs"
                    / "gate2-20260422T230909Z-latbias"
                    / "eval"
                    / "bag_trial_2_20260423_000051_086"
                    / "bag_trial_2_20260423_000051_086_0.mcap"
                ),
                size_bytes=1,
                sha256="2" * 64,
                controller_state_count=8201,
                pose_command_count=77388,
                off_limit_contact_count=0,
                controller_stamp_start_sec=13.696,
                controller_stamp_end_sec=21.898,
                controller_duration_sec=8.202,
                final_tcp_position=(-0.4000253796751864, 0.2886959168656743, 0.19707371527145434),
                final_tcp_error=(0.0010774320272363136, -0.00012170513430415086, 0.00866723340376696),
            ),
            McapEvalTrialReport(
                trial_id="trial_3",
                source=str(
                    REPO_ROOT
                    / "artifacts"
                    / "cloud-logs"
                    / "gate2-20260422T230909Z-latbias"
                    / "eval"
                    / "bag_trial_3_20260423_002740_872"
                    / "bag_trial_3_20260423_002740_872_0.mcap"
                ),
                size_bytes=1,
                sha256="3" * 64,
                controller_state_count=12303,
                pose_command_count=108332,
                off_limit_contact_count=0,
                controller_stamp_start_sec=24.796,
                controller_stamp_end_sec=32.998,
                controller_duration_sec=8.201999999999998,
                final_tcp_position=(-0.5036676280310033, 0.30092804306844867, 0.16741701249328972),
                final_tcp_error=(-0.00020671279699979728, 7.916016580544749e-05, 0.008228445481841595),
            ),
        ),
    )


def _trial_source(index: int) -> str:
    return f"/tmp/eval/bag_trial_{index}_20260423_120000_000/bag_trial_{index}_0.mcap"


def _empty_trial(index: int) -> McapEvalTrialReport:
    return McapEvalTrialReport(
        trial_id=f"trial_{index}",
        source=_trial_source(index),
        size_bytes=100 + index,
        sha256=(hex(index)[2:] * 64)[:64],
        controller_state_count=0,
        pose_command_count=0,
        off_limit_contact_count=0,
    )


def _contact_trial() -> McapEvalTrialReport:
    return McapEvalTrialReport(
        trial_id="trial_1",
        source=_trial_source(1),
        size_bytes=123,
        sha256="d" * 64,
        controller_state_count=2,
        pose_command_count=1,
        off_limit_contact_count=2,
        controller_stamp_start_sec=10.0,
        controller_stamp_end_sec=10.4,
        controller_duration_sec=0.4,
        final_tcp_position=(1.0, 2.0, 3.0),
        final_tcp_error=(0.1, 0.2, 0.3),
        task_hints=McapEvalTaskHints(port_type="sc", task_id="task_1"),
        first_off_limit_contact=McapEvalFirstContact(
            log_time_ns=1_300_000_000,
            collision1="plug",
            collision2="enclosure",
            log_elapsed_sec=0.3,
            nearest_controller_elapsed_sec=0.4,
            nearest_command_elapsed_sec=0.15,
            nearest_tcp_position=(1.0, 2.0, 3.0),
            nearest_tcp_error=(0.1, 0.2, 0.3),
            nearest_command_linear=(0.01, 0.02, 0.03),
            nearest_command_angular=(0.04, 0.05, 0.06),
            recommended_stop_sec=0.15,
        ),
    )


def _mcap_bundle() -> McapEvalBundleReport:
    return McapEvalBundleReport(
        source="/tmp/eval",
        analyzed_at_utc="2026-04-23T12:00:00Z",
        contact_margin_sec=0.25,
        stop_step_sec=0.05,
        recommended_env={"AIC_LEWM_SC_REPLAY_STOP_SEC": "0.15"},
        trials=(_contact_trial(), _empty_trial(2), _empty_trial(3)),
    )


def _run_dir(run_id: str) -> Path:
    return RUNS_ROOT / run_id


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


def test_reward_report_round_trips_strictly() -> None:
    report = RewardReport(
        run_id="gate2-candidate",
        generated_at_utc="2026-04-23T00:00:00Z",
        terms=(
            RewardTerm(
                name="official.score.total",
                value=128.0,
                signal_kind=RewardSignalKind.official_score,
                leakage_class=LeakageClass.privileged_eval_signal,
                source="/tmp/eval/scoring.yaml",
                provenance={"trial_count": 3},
            ),
        ),
        notes=("offline report",),
    )

    assert RewardReport.from_dict(report.to_dict()) == report


def test_reward_report_rejects_unknown_nested_fields() -> None:
    with pytest.raises(HarnessIOError, match="reward term has unknown fields"):
        RewardReport.from_dict(
            {
                "schema_version": 1,
                "run_id": "gate2-candidate",
                "generated_at_utc": "2026-04-23T00:00:00Z",
                "terms": [
                    {
                        "name": "official.score.total",
                        "value": 128.0,
                        "signal_kind": "official_score",
                        "leakage_class": "privileged_eval_signal",
                        "source": "/tmp/eval/scoring.yaml",
                        "unexpected": True,
                    }
                ],
                "notes": [],
            }
        )


def test_failure_report_round_trips_strictly() -> None:
    report = FailureReport(
        run_id="gate2-candidate",
        generated_at_utc="2026-04-23T00:00:00Z",
        labels=(
            FailureLabel(
                kind=FailureKind.no_partial_or_full_insertion,
                severity=FailureSeverity.blocker,
                summary="No insertion credit.",
                source="/tmp/eval/scoring.yaml",
                leakage_class=LeakageClass.privileged_eval_signal,
                evidence={"max_tier_3": 25.0},
            ),
        ),
        notes=("offline labels",),
    )

    assert FailureReport.from_dict(report.to_dict()) == report


def test_derive_reward_failure_reports_labels_proximity_only_and_margin_failure() -> None:
    score = _score_report()
    reduction = derive_reward_failure_reports(
        _manifest(score),
        promotion=_promotion_decision(value=score.total, baseline_value=score.total - 0.2),
        generated_at_utc="2026-04-23T00:00:00Z",
    )

    reward_names = {term.name for term in reduction.reward_report.terms}
    assert "official.score.total" in reward_names
    assert "promotion.metric.margin_gap" in reward_names
    failure_kinds = {label.kind for label in reduction.failure_report.labels}
    assert FailureKind.no_partial_or_full_insertion in failure_kinds
    assert FailureKind.below_promotion_margin in failure_kinds
    margin_gap = next(
        term for term in reduction.reward_report.terms if term.name == "promotion.metric.margin_gap"
    )
    assert margin_gap.value == pytest.approx(-0.8)


def test_derive_reward_failure_reports_labels_mcap_contacts_and_guard() -> None:
    reduction = derive_reward_failure_reports(
        _manifest(_score_report(tier_3=(75.0, 75.0, 75.0))),
        mcap_eval=_mcap_bundle(),
        generated_at_utc="2026-04-23T00:00:00Z",
    )

    failure_kinds = {label.kind for label in reduction.failure_report.labels}
    assert FailureKind.off_limit_contact in failure_kinds
    assert FailureKind.contact_guard_stop_recommended in failure_kinds
    contact_total = next(
        term
        for term in reduction.reward_report.terms
        if term.name == "diagnostic.off_limit_contact_count.total"
    )
    assert contact_total.value == 2.0
    contact_label = next(
        label
        for label in reduction.failure_report.labels
        if label.kind is FailureKind.off_limit_contact
    )
    assert contact_label.trial_id == "trial_1"
    assert contact_label.evidence["off_limit_contact_count"] == 2
    guard_label = next(
        label
        for label in reduction.failure_report.labels
        if label.kind is FailureKind.contact_guard_stop_recommended
    )
    assert guard_label.leakage_class is LeakageClass.privileged_eval_signal


def test_derive_reward_failure_reports_rejects_wrong_promotion_run() -> None:
    promotion = _promotion_decision()
    promotion_payload = promotion.to_dict()
    promotion_payload["run_id"] = "other-run"

    with pytest.raises(HarnessIOError, match="promotion.run_id does not match"):
        derive_reward_failure_reports(
            _manifest(_score_report()),
            promotion=PromotionDecision.from_dict(promotion_payload),
            generated_at_utc="2026-04-23T00:00:00Z",
        )


def test_derive_reward_failure_reports_rejects_mismatched_promotion_score() -> None:
    with pytest.raises(HarnessIOError, match="promotion metric.value does not match"):
        derive_reward_failure_reports(
            _manifest(_score_report()),
            promotion=_promotion_decision(value=128.0, baseline_value=127.8),
            generated_at_utc="2026-04-23T00:00:00Z",
        )


def test_derive_reward_failure_reports_marks_missing_score_inconclusive() -> None:
    reduction = derive_reward_failure_reports(
        _manifest(None),
        generated_at_utc="2026-04-23T00:00:00Z",
    )

    assert reduction.reward_report.terms == ()
    assert tuple(label.kind for label in reduction.failure_report.labels) == (
        FailureKind.inconclusive_missing_evidence,
    )


def test_real_latbias_backfill_derives_gate2_failure_labels(tmp_path: Path) -> None:
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

    reduction = derive_reward_failure_reports(
        backfill.manifest,
        mcap_eval=_latbias_mcap_bundle(),
        promotion=backfill.promotion_decision,
        generated_at_utc="2026-04-23T00:53:04Z",
    )

    total = next(term for term in reduction.reward_report.terms if term.name == "official.score.total")
    assert total.value == pytest.approx(128.00872420948235)
    failure_kinds = {label.kind for label in reduction.failure_report.labels}
    assert FailureKind.no_partial_or_full_insertion in failure_kinds
    assert FailureKind.below_promotion_margin in failure_kinds
    assert FailureKind.off_limit_contact not in failure_kinds
