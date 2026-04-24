from __future__ import annotations

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
    TimelineEventKind,
    TrainingSignalKind,
    TrainingSignalReport,
    TrialScore,
    derive_episode_trace,
    derive_training_signal_report,
)


def _policy_events(run_id: str = "run-a"):
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
        ),
        PolicyTraceEvent(
            run_id=run_id,
            trial_id="task_1__policy_call_0001",
            event_index=1,
            event_type=PolicyTraceEventType.action_published,
            elapsed_sec=0.25,
            emitted_at_utc="2026-04-24T00:00:01Z",
            source="pytest",
            leakage_class=LeakageClass.legal_policy_input,
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
        ),
    )


def _score_report() -> ScoreReport:
    return ScoreReport(
        source="/tmp/scoring.yaml",
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
                source="/tmp/scoring.yaml",
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
                source="/tmp/scoring.yaml",
                leakage_class=LeakageClass.privileged_eval_signal,
                evidence={"max_tier_3": 4.0},
            ),
        ),
    )


def test_episode_trace_round_trip_and_training_signals() -> None:
    source_artifact = ArtifactRef(
        kind="policy_trace_jsonl",
        path="/tmp/policy_trace.jsonl",
        sha256="a" * 64,
    )

    trace = derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=(source_artifact,),
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
        source_trace=ArtifactRef(
            kind="episode_trace",
            path="/tmp/episode_trace.json",
            sha256="b" * 64,
        ),
    )

    assert TrainingSignalReport.from_dict(report.to_dict()) == report
    assert {signal.kind for signal in report.signals} == {
        TrainingSignalKind.behavior_clone_action,
        TrainingSignalKind.safety_guard_avoidance,
        TrainingSignalKind.official_score_term,
        TrainingSignalKind.reward_term,
        TrainingSignalKind.failure_label,
    }


def test_episode_trace_rejects_policy_run_id_mismatch() -> None:
    with pytest.raises(HarnessIOError, match="policy trace events do not match"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(run_id="other-run"),
            source_artifacts=(),
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
