from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from aic_signal_harness import (
    ArtifactRef,
    EpisodeTrace,
    FailureKind,
    FailureLabel,
    FailureReport,
    FailureSeverity,
    HarnessIOError,
    LeakageClass,
    PolicyTraceEvent,
    PolicyTraceEventType,
    RewardReport,
    RewardSignalKind,
    RewardTerm,
    ScoreReport,
    SchemaValidationError,
    TimelineEvent,
    TimelineEventKind,
    TrainingSignal,
    TrainingSignalKind,
    TrainingSignalReport,
    TrialTrace,
    TrialScore,
    derive_episode_trace,
    derive_training_signal_report,
    sha256_file,
    write_json,
)

_SCORE_SOURCE = "memory://pytest/scoring.yaml"


def _policy_events(run_id: str = "run-a", official_trial_id: str | None = "trial_1"):
    return (
        PolicyTraceEvent(
            run_id=run_id,
            trial_id="task_1__policy_call_0001",
            event_index=0,
            event_type=PolicyTraceEventType.task_started,
            elapsed_sec=0.0,
            emitted_at_utc="2026-04-24T00:00:00Z",
            source="pytest",
            leakage_class=LeakageClass.legal_policy_input,
            payload={"task_id": "task_1"},
            official_trial_id=official_trial_id,
        ),
        PolicyTraceEvent(
            run_id=run_id,
            trial_id="task_1__policy_call_0001",
            event_index=1,
            event_type=PolicyTraceEventType.action_published,
            elapsed_sec=0.25,
            emitted_at_utc="2026-04-24T00:00:01Z",
            source="pytest",
            leakage_class=LeakageClass.legal_policy_action_output,
            official_trial_id=official_trial_id,
            payload={
                "linear": [0.1, 0.0, 0.0],
                "angular": [0.0, 0.0, 0.0],
                "frame_id": "base_link",
            },
        ),
        PolicyTraceEvent(
            run_id=run_id,
            trial_id="task_1__policy_call_0001",
            event_index=2,
            event_type=PolicyTraceEventType.safety_guard,
            elapsed_sec=0.5,
            emitted_at_utc="2026-04-24T00:00:02Z",
            source="pytest",
            leakage_class=LeakageClass.legal_policy_input,
            payload={"guard": "delta_force", "stopped": True},
            official_trial_id=official_trial_id,
        ),
    )


def _score_report() -> ScoreReport:
    return ScoreReport(
        source=_SCORE_SOURCE,
        parsed_at_utc="2026-04-24T00:00:03Z",
        total=7.5,
        trials={"trial_1": TrialScore(total=7.5, tier_1=1.0, tier_2=2.5, tier_3=4.0)},
    )


def _reward_report() -> RewardReport:
    return RewardReport(
        run_id="run-a",
        generated_at_utc="2026-04-24T00:00:04Z",
        terms=(
            RewardTerm(
                name="official.score.total",
                value=7.5,
                signal_kind=RewardSignalKind.official_score,
                leakage_class=LeakageClass.privileged_eval_signal,
                source=_SCORE_SOURCE,
            ),
            RewardTerm(
                name="diagnostic.margin",
                value=123.0,
                signal_kind=RewardSignalKind.diagnostic,
                leakage_class=LeakageClass.privileged_eval_signal,
                source=_SCORE_SOURCE,
                trial_id="task_1__policy_call_0001",
            ),
        ),
    )


def _failure_report() -> FailureReport:
    return FailureReport(
        run_id="run-a",
        generated_at_utc="2026-04-24T00:00:04Z",
        labels=(
            FailureLabel(
                kind=FailureKind.no_partial_or_full_insertion,
                severity=FailureSeverity.blocker,
                summary="No insertion credit.",
                source=_SCORE_SOURCE,
                leakage_class=LeakageClass.privileged_eval_signal,
                trial_id="task_1__policy_call_0001",
                evidence={"max_tier_3": 4.0},
            ),
        ),
    )


def _source_artifacts(*, include_reports: bool = False) -> tuple[ArtifactRef, ...]:
    policy_artifact = ArtifactRef(
        kind="policy_trace_jsonl",
        uri="memory://pytest/policy_trace.jsonl",
        sha256="a" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    scoring_artifact = ArtifactRef(
        kind="scoring_yaml",
        uri=_SCORE_SOURCE,
        sha256="c" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    artifacts: tuple[ArtifactRef, ...] = (scoring_artifact, policy_artifact)
    if include_reports:
        report_source_artifacts = [
            {"kind": scoring_artifact.kind, "sha256": scoring_artifact.sha256},
            {"kind": policy_artifact.kind, "sha256": policy_artifact.sha256},
        ]
        artifacts += (
            ArtifactRef(
                kind="reward_report",
                uri="memory://pytest/reward_report.json",
                sha256="d" * 64,
                provenance={
                    "producer": "pytest",
                    "run_id": "run-a",
                    "derivation": "derive_reward_failure_reports",
                    "source_artifacts": report_source_artifacts,
                },
            ),
            ArtifactRef(
                kind="failure_report",
                uri="memory://pytest/failure_report.json",
                sha256="e" * 64,
                provenance={
                    "producer": "pytest",
                    "run_id": "run-a",
                    "derivation": "derive_reward_failure_reports",
                    "source_artifacts": report_source_artifacts,
                },
            ),
        )
    return artifacts


def _episode_trace() -> EpisodeTrace:
    return derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=_source_artifacts(include_reports=True),
        score_report=_score_report(),
        reward_report=_reward_report(),
        failure_report=_failure_report(),
        generated_at_utc="2026-04-24T00:00:05Z",
    )


def _episode_trace_sha256(trace: EpisodeTrace) -> str:
    payload = (
        json.dumps(trace.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_trace_ref(
    trace: EpisodeTrace,
    *,
    uri: str = "memory://pytest/episode_trace.json",
) -> ArtifactRef:
    return ArtifactRef(
        kind="episode_trace",
        uri=uri,
        sha256=_episode_trace_sha256(trace),
        provenance={
            "producer": "pytest",
            "run_id": trace.run_id,
            "derivation": "derive_episode_trace",
        },
    )


def test_episode_trace_round_trip_and_training_signals() -> None:
    trace = derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=_source_artifacts(include_reports=True),
        score_report=_score_report(),
        reward_report=_reward_report(),
        failure_report=_failure_report(),
        generated_at_utc="2026-04-24T00:00:05Z",
    )

    assert EpisodeTrace.from_dict(trace.to_dict()) == trace
    assert trace.trials[0].action_event_count == 1
    assert trace.trials[0].nonzero_action_event_count == 1
    assert trace.trials[0].safety_guard_event_count == 1
    assert trace.trials[0].score is not None
    assert {event.event_kind for event in trace.run_events} == {
        TimelineEventKind.official_score,
        TimelineEventKind.reward_term,
        TimelineEventKind.failure_label,
    }

    report = derive_training_signal_report(
        episode_trace=trace,
        source_trace=_source_trace_ref(trace),
        generated_at_utc="2026-04-24T00:00:06Z",
    )

    assert TrainingSignalReport.from_dict(report.to_dict()) == report
    assert {signal.kind for signal in report.signals} == {
        TrainingSignalKind.behavior_clone_action,
        TrainingSignalKind.safety_guard_avoidance,
        TrainingSignalKind.official_score_term,
        TrainingSignalKind.reward_term,
        TrainingSignalKind.failure_label,
    }
    assert report.generated_at_utc == "2026-04-24T00:00:06Z"
    for signal in report.signals:
        assert signal.offline_only is True
        assert signal.runtime_allowed is False
        assert signal.consumable_by_policy_runtime is False
    reward_signal = next(signal for signal in report.signals if signal.kind is TrainingSignalKind.reward_term)
    assert reward_signal.weight == 1.0
    assert reward_signal.evidence["value"] == 123.0
    assert all(
        signal.kind is not TrainingSignalKind.reward_term
        or signal.evidence["signal_kind"] != "official_score"
        for signal in report.signals
    )
    assert all(event.elapsed_sec == 0.5 for event in trace.run_events)
    assert all(event.payload["event_scope"].startswith("post_hoc") for event in trace.run_events)
    trial_scoped = [event for event in trace.run_events if event.trial_id == "task_1__policy_call_0001"]
    assert trial_scoped
    assert all(event.payload["evidence_window"]["end_elapsed_sec"] == 0.5 for event in trial_scoped)


def test_training_signal_report_defaults_to_episode_trace_timestamp() -> None:
    trace = _episode_trace()

    first_report = derive_training_signal_report(
        episode_trace=trace,
        source_trace=_source_trace_ref(trace),
    )
    second_report = derive_training_signal_report(
        episode_trace=trace,
        source_trace=_source_trace_ref(trace),
    )

    assert first_report == second_report
    assert first_report.generated_at_utc == trace.generated_at_utc


def test_episode_trace_defaults_to_latest_source_evidence_timestamp() -> None:
    first_trace = derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=_source_artifacts(),
        score_report=_score_report(),
    )
    second_trace = derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=_source_artifacts(),
        score_report=_score_report(),
    )

    assert first_trace == second_trace
    assert first_trace.generated_at_utc == "2026-04-24T00:00:03Z"

    fused_trace = derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=_source_artifacts(include_reports=True),
        score_report=_score_report(),
        reward_report=_reward_report(),
        failure_report=_failure_report(),
    )
    assert fused_trace.generated_at_utc == "2026-04-24T00:00:04Z"


def test_episode_trace_rejects_policy_run_id_mismatch() -> None:
    with pytest.raises(HarnessIOError, match="policy trace events do not match"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(run_id="other-run"),
            source_artifacts=(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )


def test_episode_trace_rejects_non_policy_event_inputs() -> None:
    with pytest.raises(HarnessIOError, match="PolicyTraceEvent instances"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=cast(Any, ("not-a-policy-event",)),
            source_artifacts=_source_artifacts(),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )


def test_episode_trace_rejects_reward_report_run_id_mismatch() -> None:
    bad_reward = RewardReport(
        run_id="other-run",
        generated_at_utc="2026-04-24T00:00:04Z",
        terms=(),
    )

    with pytest.raises(HarnessIOError, match="reward_report.run_id"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(),
            reward_report=bad_reward,
            generated_at_utc="2026-04-24T00:00:05Z",
        )


def test_episode_trace_rejects_ambiguous_or_mismatched_score_trial_mapping() -> None:
    second_policy_call = PolicyTraceEvent(
        run_id="run-a",
        trial_id="task_1__policy_call_0002",
        event_index=3,
        event_type=PolicyTraceEventType.task_started,
        elapsed_sec=0.0,
        emitted_at_utc="2026-04-24T00:00:03Z",
        source="pytest",
        leakage_class=LeakageClass.legal_policy_input,
        payload={"task_id": "task_1"},
        official_trial_id="trial_1",
    )

    with pytest.raises(HarnessIOError, match="ambiguously"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events() + (second_policy_call,),
            source_artifacts=_source_artifacts(),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    two_trial_score = ScoreReport(
        source=_SCORE_SOURCE,
        parsed_at_utc="2026-04-24T00:00:03Z",
        total=8.0,
        trials={
            "trial_1": TrialScore(total=7.5, tier_1=1.0, tier_2=2.5, tier_3=4.0),
            "trial_2": TrialScore(total=0.5, tier_1=0.5, tier_2=0.0, tier_3=0.0),
        },
    )
    with pytest.raises(HarnessIOError, match="unmatched official scores"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=_source_artifacts(),
            score_report=two_trial_score,
            generated_at_utc="2026-04-24T00:00:05Z",
        )


def test_episode_trace_rejects_scored_policy_events_without_official_trial_id() -> None:
    with pytest.raises(HarnessIOError, match="official_trial_id"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(official_trial_id=None),
            source_artifacts=_source_artifacts(),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )


def test_episode_trace_revalidates_direct_policy_event_sequences() -> None:
    events = _policy_events()
    bad_index = PolicyTraceEvent(
        run_id="run-a",
        trial_id="task_1__policy_call_0001",
        event_index=4,
        event_type=PolicyTraceEventType.task_finished,
        elapsed_sec=0.75,
        emitted_at_utc="2026-04-24T00:00:03Z",
        source="pytest",
        leakage_class=LeakageClass.legal_policy_input,
        payload={},
    )
    with pytest.raises(HarnessIOError, match="contiguous"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=events + (bad_index,),
            source_artifacts=_source_artifacts(),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    nonmonotonic = PolicyTraceEvent(
        run_id="run-a",
        trial_id="task_1__policy_call_0001",
        event_index=3,
        event_type=PolicyTraceEventType.task_finished,
        elapsed_sec=0.1,
        emitted_at_utc="2026-04-24T00:00:03Z",
        source="pytest",
        leakage_class=LeakageClass.legal_policy_input,
        payload={},
    )
    with pytest.raises(HarnessIOError, match="nondecreasing"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=events + (nonmonotonic,),
            source_artifacts=_source_artifacts(),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )


def test_episode_trace_rejects_weak_source_artifacts() -> None:
    with pytest.raises(HarnessIOError, match="source_artifacts must not be empty"):
        EpisodeTrace(
            run_id="run-a",
            generated_at_utc="2026-04-24T00:00:05Z",
            trials=(_episode_trace().trials[0],),
        )
    with pytest.raises(HarnessIOError, match="must include exactly one scoring_yaml"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(_source_artifacts()[1],),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )
    with pytest.raises(HarnessIOError, match="must include exactly one reward_report"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=_source_artifacts(),
            score_report=_score_report(),
            reward_report=_reward_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )
    with pytest.raises(HarnessIOError, match="must include exactly one failure_report"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=_source_artifacts(),
            score_report=_score_report(),
            failure_report=_failure_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )
    scoring_artifact, policy_artifact = _source_artifacts()
    weak_reward_artifact = ArtifactRef(
        kind="reward_report",
        uri="memory://pytest/reward_report.json",
        sha256="d" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="provenance.derivation"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(scoring_artifact, policy_artifact, weak_reward_artifact),
            score_report=_score_report(),
            reward_report=_reward_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )
    malformed_reward_artifact = ArtifactRef(
        kind="reward_report",
        uri="memory://pytest/reward_report.json",
        sha256="d" * 64,
        provenance={
            "producer": "pytest",
            "run_id": "run-a",
            "derivation": "derive_reward_failure_reports",
            "source_artifacts": [{"kind": "scoring_yaml", "sha256": "not-a-digest"}],
        },
    )
    with pytest.raises(HarnessIOError, match=r"source_artifacts\[0\].sha256"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(scoring_artifact, policy_artifact, malformed_reward_artifact),
            score_report=_score_report(),
            reward_report=_reward_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )
    forged_reward_artifact = ArtifactRef(
        kind="reward_report",
        uri="memory://pytest/reward_report.json",
        sha256="d" * 64,
        provenance={
            "producer": "pytest",
            "run_id": "run-a",
            "derivation": "derive_reward_failure_reports",
            "source_artifacts": [{"kind": "scoring_yaml", "sha256": "f" * 64}],
        },
    )
    with pytest.raises(HarnessIOError, match="must match an episode trace source artifact"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(scoring_artifact, policy_artifact, forged_reward_artifact),
            score_report=_score_report(),
            reward_report=_reward_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )
    promotion_reward_report = RewardReport(
        run_id="run-a",
        generated_at_utc="2026-04-24T00:00:04Z",
        terms=(
            RewardTerm(
                name="promotion.metric.improvement",
                value=1.0,
                signal_kind=RewardSignalKind.diagnostic,
                leakage_class=LeakageClass.privileged_eval_signal,
                source="promotion_decision",
            ),
        ),
    )
    promotion_derived_reward_artifact = ArtifactRef(
        kind="reward_report",
        uri="memory://pytest/reward_report.json",
        sha256="d" * 64,
        provenance={
            "producer": "pytest",
            "run_id": "run-a",
            "derivation": "derive_reward_failure_reports",
            "source_artifacts": [
                {"kind": scoring_artifact.kind, "sha256": scoring_artifact.sha256},
            ],
        },
    )
    with pytest.raises(HarnessIOError, match="promotion_decision_snapshot"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(
                scoring_artifact,
                policy_artifact,
                promotion_derived_reward_artifact,
            ),
            score_report=_score_report(),
            reward_report=promotion_reward_report,
            generated_at_utc="2026-04-24T00:00:05Z",
        )


def test_episode_trace_rejects_foreign_or_unbound_source_artifacts(tmp_path: Path) -> None:
    scoring_artifact, policy_artifact = _source_artifacts()
    foreign_policy = ArtifactRef(
        kind=policy_artifact.kind,
        uri=policy_artifact.uri,
        sha256=policy_artifact.sha256,
        provenance={"producer": "pytest", "run_id": "other-run"},
    )
    with pytest.raises(HarnessIOError, match="provenance.run_id"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(scoring_artifact, foreign_policy),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    missing_policy = ArtifactRef(
        kind="policy_trace_jsonl",
        path=str(tmp_path / "missing_policy_trace.jsonl"),
        sha256="a" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="policy_trace_jsonl source artifact path must exist"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(scoring_artifact, missing_policy),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    missing_policy_file_uri = ArtifactRef(
        kind="policy_trace_jsonl",
        uri=(tmp_path / "missing_policy_trace_uri.jsonl").as_uri(),
        sha256="a" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="policy_trace_jsonl source artifact path must exist"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(scoring_artifact, missing_policy_file_uri),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    wrong_scoring = ArtifactRef(
        kind="scoring_yaml",
        path="/tmp/not_the_score.yaml",
        sha256="c" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="score_report.source"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(wrong_scoring, policy_artifact),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    scoring_path = tmp_path / "scoring.yaml"
    scoring_path.write_text("total: 7.5\n", encoding="utf-8")
    wrong_digest_scoring = ArtifactRef(
        kind="scoring_yaml",
        path=str(scoring_path),
        sha256="0" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="scoring_yaml source artifact sha256"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(wrong_digest_scoring, policy_artifact),
            score_report=ScoreReport(
                source=str(scoring_path.resolve()),
                parsed_at_utc="2026-04-24T00:00:03Z",
                total=7.5,
                trials={
                    "trial_1": TrialScore(total=7.5, tier_1=1.0, tier_2=2.5, tier_3=4.0)
                },
            ),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    localhost_scoring = ArtifactRef(
        kind="scoring_yaml",
        path=str(scoring_path),
        sha256=sha256_file(scoring_path),
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    localhost_trace = derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=(localhost_scoring, policy_artifact),
        score_report=ScoreReport(
            source=f"file://localhost{scoring_path.resolve().as_posix()}",
            parsed_at_utc="2026-04-24T00:00:03Z",
            total=7.5,
            trials={
                "trial_1": TrialScore(total=7.5, tier_1=1.0, tier_2=2.5, tier_3=4.0)
            },
        ),
        generated_at_utc="2026-04-24T00:00:05Z",
    )
    assert localhost_trace.trials[0].score is not None

    missing_scoring_without_report = ArtifactRef(
        kind="scoring_yaml",
        path=str(tmp_path / "missing_scoring.yaml"),
        sha256="c" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="scoring_yaml source artifact path must exist"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(missing_scoring_without_report, policy_artifact),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    missing_scoring_file_uri_without_report = ArtifactRef(
        kind="scoring_yaml",
        uri=(tmp_path / "missing_scoring_uri.yaml").as_uri(),
        sha256="c" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="scoring_yaml source artifact path must exist"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(missing_scoring_file_uri_without_report, policy_artifact),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    scoring_a_path = tmp_path / "scoring_a.yaml"
    scoring_b_path = tmp_path / "scoring_b.yaml"
    scoring_a_path.write_text("total: 7.5\n", encoding="utf-8")
    scoring_b_path.write_text("total: 7.5\n", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="path and file URI"):
        ArtifactRef(
            kind="scoring_yaml",
            path=str(scoring_a_path),
            uri=scoring_b_path.as_uri(),
            sha256=sha256_file(scoring_a_path),
            provenance={"producer": "pytest", "run_id": "run-a"},
        )

    policy_path = tmp_path / "policy_trace.jsonl"
    policy_path.write_text(
        json.dumps(
            {
                **_policy_events()[0].to_dict(),
                "payload": {"task_id": "different_task"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    bound_policy = ArtifactRef(
        kind="policy_trace_jsonl",
        path=str(policy_path),
        sha256=sha256_file(policy_path),
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="must match policy_events"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(scoring_artifact, bound_policy),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )


def test_episode_trace_rejects_inconsistent_summaries_and_indices() -> None:
    trace = _episode_trace()
    trial = trace.trials[0]
    with pytest.raises(HarnessIOError, match="action_event_count"):
        TrialTrace(
            trial_id=trial.trial_id,
            start_elapsed_sec=trial.start_elapsed_sec,
            end_elapsed_sec=trial.end_elapsed_sec,
            event_count=trial.event_count,
            action_event_count=0,
            nonzero_action_event_count=trial.nonzero_action_event_count,
            safety_guard_event_count=trial.safety_guard_event_count,
            error_event_count=trial.error_event_count,
            events=trial.events,
        )

    bad_run_event = TimelineEvent(
        event_index=99,
        event_kind=TimelineEventKind.official_score,
        elapsed_sec=0.5,
        source="/tmp/scoring.yaml",
        leakage_class=LeakageClass.privileged_eval_signal,
        payload={"event_scope": "post_hoc_run_summary", "evidence_window": {"start_elapsed_sec": 0.0, "end_elapsed_sec": 0.5}},
    )
    with pytest.raises(HarnessIOError, match="contiguous"):
        EpisodeTrace(
            run_id=trace.run_id,
            generated_at_utc=trace.generated_at_utc,
            source_artifacts=trace.source_artifacts,
            trials=trace.trials,
            run_events=(bad_run_event,),
        )


def test_training_signal_report_rejects_weak_source_trace_and_non_string_keys() -> None:
    trace = _episode_trace()
    with pytest.raises(HarnessIOError, match="source_trace.sha256"):
        derive_training_signal_report(
            episode_trace=trace,
            source_trace=ArtifactRef(
                kind="episode_trace",
                path="/tmp/episode_trace.json",
                provenance={
                    "producer": "pytest",
                    "run_id": "run-a",
                    "derivation": "derive_episode_trace",
                },
            ),
        )
    with pytest.raises(HarnessIOError, match="provenance run_id"):
        TrainingSignalReport(
            run_id="run-a",
            generated_at_utc="2026-04-24T00:00:06Z",
            source_trace=ArtifactRef(
                kind="episode_trace",
                uri="memory://pytest/episode_trace.json",
                sha256="b" * 64,
                provenance={
                    "producer": "pytest",
                    "run_id": "other-run",
                    "derivation": "derive_episode_trace",
                },
            ),
        )
    with pytest.raises(HarnessIOError, match="keys must be nonempty strings"):
        TimelineEvent(
            event_index=0,
            event_kind=TimelineEventKind.policy_event,
            elapsed_sec=0.0,
            source="pytest",
            leakage_class=LeakageClass.legal_policy_input,
            payload=cast(Any, {1: "not-json-object-contract"}),
        )
    with pytest.raises(HarnessIOError, match="keys must be nonempty strings"):
        TrainingSignal(
            kind=TrainingSignalKind.failure_label,
            target="failure",
            weight=1.0,
            source="pytest",
            leakage_class=LeakageClass.post_hoc_label,
            evidence=cast(Any, {1: "not-json-object-contract"}),
        )


def test_training_signal_report_binds_source_trace_digest_and_content(tmp_path: Path) -> None:
    trace = _episode_trace()
    bogus_trace = derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=_source_artifacts(include_reports=True),
        score_report=_score_report(),
        generated_at_utc="2026-04-24T00:00:05Z",
    )
    source_path = tmp_path / "episode_trace.json"
    write_json(source_path, bogus_trace.to_dict())

    with pytest.raises(HarnessIOError, match="does not match supplied episode trace"):
        derive_training_signal_report(
            episode_trace=trace,
            source_trace=ArtifactRef(
                kind="episode_trace",
                path=str(source_path),
                sha256=sha256_file(source_path),
                provenance={
                    "producer": "pytest",
                    "run_id": "run-a",
                    "derivation": "derive_episode_trace",
                },
            ),
        )

    with pytest.raises(HarnessIOError, match="source_trace.path must exist"):
        derive_training_signal_report(
            episode_trace=trace,
            source_trace=ArtifactRef(
                kind="episode_trace",
                path=str(tmp_path / "missing_episode_trace.json"),
                sha256=_episode_trace_sha256(trace),
                provenance={
                    "producer": "pytest",
                    "run_id": "run-a",
                    "derivation": "derive_episode_trace",
                },
            ),
        )

    with pytest.raises(HarnessIOError, match="source_trace.path must exist"):
        derive_training_signal_report(
            episode_trace=trace,
            source_trace=ArtifactRef(
                kind="episode_trace",
                uri=(tmp_path / "missing_episode_trace_uri.json").as_uri(),
                sha256=_episode_trace_sha256(trace),
                provenance={
                    "producer": "pytest",
                    "run_id": "run-a",
                    "derivation": "derive_episode_trace",
                },
            ),
        )

    copied_source_path = tmp_path / "episode_trace_copy.json"
    copied_uri_path = tmp_path / "episode_trace_uri_copy.json"
    write_json(copied_source_path, trace.to_dict())
    write_json(copied_uri_path, trace.to_dict())
    with pytest.raises(SchemaValidationError, match="path and file URI"):
        ArtifactRef(
            kind="episode_trace",
            path=str(copied_source_path),
            uri=copied_uri_path.as_uri(),
            sha256=sha256_file(copied_source_path),
            provenance={
                "producer": "pytest",
                "run_id": "run-a",
                "derivation": "derive_episode_trace",
            },
        )

    with pytest.raises(HarnessIOError, match="sha256 must match supplied"):
        derive_training_signal_report(
            episode_trace=trace,
            source_trace=ArtifactRef(
                kind="episode_trace",
                uri="memory://pytest/episode_trace.json",
                sha256="b" * 64,
                provenance={
                    "producer": "pytest",
                    "run_id": "run-a",
                    "derivation": "derive_episode_trace",
                },
            ),
        )


def test_training_signals_preserve_trace_trial_ids_for_mapped_labels() -> None:
    reward_report = RewardReport(
        run_id="run-a",
        generated_at_utc="2026-04-24T00:00:04Z",
        terms=(
            RewardTerm(
                name="diagnostic.margin",
                value=123.0,
                signal_kind=RewardSignalKind.diagnostic,
                leakage_class=LeakageClass.privileged_eval_signal,
                source=_SCORE_SOURCE,
                trial_id="trial_1",
            ),
        ),
    )
    failure_report = FailureReport(
        run_id="run-a",
        generated_at_utc="2026-04-24T00:00:04Z",
        labels=(
            FailureLabel(
                kind=FailureKind.no_partial_or_full_insertion,
                severity=FailureSeverity.blocker,
                summary="No insertion credit.",
                source=_SCORE_SOURCE,
                leakage_class=LeakageClass.privileged_eval_signal,
                trial_id="trial_1",
            ),
        ),
    )
    trace = derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=_source_artifacts(include_reports=True),
        score_report=_score_report(),
        reward_report=reward_report,
        failure_report=failure_report,
        generated_at_utc="2026-04-24T00:00:05Z",
    )

    report = derive_training_signal_report(
        episode_trace=trace,
        source_trace=_source_trace_ref(trace),
        generated_at_utc="2026-04-24T00:00:06Z",
    )

    label_signals = tuple(
        signal
        for signal in report.signals
        if signal.kind in {TrainingSignalKind.reward_term, TrainingSignalKind.failure_label}
    )
    assert {signal.trial_id for signal in label_signals} == {"task_1__policy_call_0001"}
    assert {signal.evidence["trial_id"] for signal in label_signals} == {"trial_1"}
    assert {signal.evidence["source_trial_id"] for signal in label_signals} == {"trial_1"}


def test_training_signal_from_dict_requires_explicit_runtime_boundary_flags() -> None:
    payload = TrainingSignal(
        kind=TrainingSignalKind.failure_label,
        target="failure",
        weight=1.0,
        source="pytest",
        leakage_class=LeakageClass.post_hoc_label,
        evidence={"kind": "failure"},
    ).to_dict()

    for field_name in (
        "offline_only",
        "runtime_allowed",
        "consumable_by_policy_runtime",
    ):
        missing = dict(payload)
        del missing[field_name]
        with pytest.raises(HarnessIOError, match=field_name):
            TrainingSignal.from_dict(missing)


def test_training_signal_report_rejects_privileged_action_signal_extraction() -> None:
    privileged_action = TimelineEvent(
        event_index=1,
        event_kind=TimelineEventKind.action_event,
        elapsed_sec=0.25,
        source="pytest",
        leakage_class=LeakageClass.privileged_eval_signal,
        trial_id="task_1__policy_call_0001",
        payload={
            "policy_event_type": "action_published",
            "policy_payload": {
                "linear": [0.1, 0.0, 0.0],
                "angular": [0.0, 0.0, 0.0],
            },
        },
    )
    trace = _episode_trace()
    trial = trace.trials[0]
    mutated_trial = TrialTrace(
        trial_id=trial.trial_id,
        start_elapsed_sec=trial.start_elapsed_sec,
        end_elapsed_sec=trial.end_elapsed_sec,
        event_count=trial.event_count,
        action_event_count=trial.action_event_count,
        nonzero_action_event_count=trial.nonzero_action_event_count,
        safety_guard_event_count=trial.safety_guard_event_count,
        error_event_count=trial.error_event_count,
        score=trial.score,
        events=(trial.events[0], privileged_action, trial.events[2]),
    )
    privileged_trace = EpisodeTrace(
        run_id=trace.run_id,
        generated_at_utc=trace.generated_at_utc,
        source_artifacts=trace.source_artifacts,
        trials=(mutated_trial,),
        run_events=trace.run_events,
    )
    with pytest.raises(HarnessIOError, match="legal_policy_action_output"):
        derive_training_signal_report(
            episode_trace=privileged_trace,
            source_trace=_source_trace_ref(privileged_trace),
        )
