"""Reducers for canonical episode traces and training signals."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from aic_signal_harness.artifacts import HarnessIOError
from aic_signal_harness.episode_trace import (
    EpisodeTrace,
    TimelineEvent,
    TimelineEventKind,
    TrialTrace,
)
from aic_signal_harness.failure import FailureLabel, FailureReport, FailureSeverity
from aic_signal_harness.policy_trace import (
    PolicyTraceEvent,
    PolicyTraceEventType,
    action_payload_is_nonzero,
)
from aic_signal_harness.reward import RewardReport, RewardTerm
from aic_signal_harness.schemas import ArtifactRef, LeakageClass, utc_now_iso
from aic_signal_harness.scoring import ScoreReport, TrialScore
from aic_signal_harness.training_signal import (
    TrainingSignal,
    TrainingSignalKind,
    TrainingSignalReport,
)


@dataclass(frozen=True)
class EpisodeTrainingSignalReduction:
    """Episode trace paired with deterministic training-signal extraction."""

    episode_trace: EpisodeTrace
    training_signal_report: TrainingSignalReport

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.episode_trace, EpisodeTrace):
            errors.append("episode training reduction episode_trace must be an EpisodeTrace")
        if not isinstance(self.training_signal_report, TrainingSignalReport):
            errors.append(
                "episode training reduction training_signal_report must be a TrainingSignalReport"
            )
        if (
            isinstance(self.episode_trace, EpisodeTrace)
            and isinstance(self.training_signal_report, TrainingSignalReport)
            and self.episode_trace.run_id != self.training_signal_report.run_id
        ):
            errors.append("episode trace and training signal report must describe the same run_id")
        if errors:
            raise HarnessIOError("; ".join(errors))


def derive_episode_trace(
    *,
    run_id: str,
    policy_events: tuple[PolicyTraceEvent, ...],
    source_artifacts: tuple[ArtifactRef, ...],
    score_report: ScoreReport | Mapping[str, Any] | None = None,
    reward_report: RewardReport | Mapping[str, Any] | None = None,
    failure_report: FailureReport | Mapping[str, Any] | None = None,
    generated_at_utc: str | None = None,
) -> EpisodeTrace:
    """Fuse policy events with score/reward/failure evidence into an episode trace."""

    if not policy_events:
        raise HarnessIOError("episode trace requires at least one policy event")
    run_id = _require_run_id(run_id)
    observed_run_ids = {event.run_id for event in policy_events}
    if observed_run_ids != {run_id}:
        raise HarnessIOError("policy trace events do not match episode trace run_id")
    typed_score = _optional_score_report(score_report)
    typed_reward = _optional_reward_report(reward_report)
    typed_failure = _optional_failure_report(failure_report)
    _validate_report_run_ids(run_id, typed_reward, typed_failure)
    generated_at = utc_now_iso() if generated_at_utc is None else generated_at_utc

    events_by_trial = _events_by_trial(policy_events)
    trial_scores = _scores_by_policy_trial(events_by_trial, typed_score)
    trials = tuple(
        _trial_trace(
            trial_id=trial_id,
            policy_events=trial_events,
            score=trial_scores.get(trial_id),
        )
        for trial_id, trial_events in events_by_trial.items()
    )
    return EpisodeTrace(
        run_id=run_id,
        generated_at_utc=generated_at,
        source_artifacts=source_artifacts,
        trials=trials,
        run_events=_run_events(
            score_report=typed_score,
            reward_report=typed_reward,
            failure_report=typed_failure,
            start_index=sum(trial.event_count for trial in trials),
        ),
        notes=(
            "Canonical episode trace fused from policy JSONL, official score, reward report, and failure report.",
        ),
    )


def derive_training_signal_report(
    *,
    episode_trace: EpisodeTrace | Mapping[str, Any],
    source_trace: ArtifactRef,
    generated_at_utc: str | None = None,
) -> TrainingSignalReport:
    """Extract deterministic training signals from a canonical episode trace."""

    trace = (
        episode_trace
        if isinstance(episode_trace, EpisodeTrace)
        else EpisodeTrace.from_dict(episode_trace)
    )
    generated_at = trace.generated_at_utc if generated_at_utc is None else generated_at_utc
    signals: list[TrainingSignal] = []
    for trial in trace.trials:
        for event in trial.events:
            if event.event_kind is TimelineEventKind.action_event and _action_event_is_nonzero(event):
                signals.append(_behavior_clone_signal(event))
            elif event.event_kind is TimelineEventKind.safety_guard:
                signals.append(_safety_signal(event))
    for event in trace.run_events:
        if event.event_kind is TimelineEventKind.official_score:
            signals.append(_official_score_signal(event))
        elif event.event_kind is TimelineEventKind.reward_term:
            signals.append(_reward_signal(event))
        elif event.event_kind is TimelineEventKind.failure_label:
            signals.append(_failure_signal(event))
    return TrainingSignalReport(
        run_id=trace.run_id,
        generated_at_utc=generated_at,
        source_trace=source_trace,
        signals=tuple(signals),
        notes=(
            "Signals are deterministic extraction records. They are not a training job and do not enter the live policy loop.",
        ),
    )


def _trial_trace(
    *,
    trial_id: str,
    policy_events: tuple[PolicyTraceEvent, ...],
    score: TrialScore | None,
) -> TrialTrace:
    timeline = tuple(_timeline_event_from_policy(event) for event in policy_events)
    event_kinds = Counter(event.event_kind for event in timeline)
    return TrialTrace(
        trial_id=trial_id,
        start_elapsed_sec=policy_events[0].elapsed_sec,
        end_elapsed_sec=policy_events[-1].elapsed_sec,
        event_count=len(timeline),
        action_event_count=event_kinds[TimelineEventKind.action_event],
        nonzero_action_event_count=sum(
            event.event_kind is TimelineEventKind.action_event and _action_event_is_nonzero(event)
            for event in timeline
        ),
        safety_guard_event_count=event_kinds[TimelineEventKind.safety_guard],
        error_event_count=event_kinds[TimelineEventKind.error],
        score=score,
        events=timeline,
    )


def _timeline_event_from_policy(event: PolicyTraceEvent) -> TimelineEvent:
    return TimelineEvent(
        event_index=event.event_index,
        event_kind=_timeline_kind(event.event_type),
        elapsed_sec=event.elapsed_sec,
        source=event.source,
        leakage_class=event.leakage_class,
        payload={
            "policy_event_type": event.event_type.value,
            "policy_payload": event.payload,
        },
        trial_id=event.trial_id,
        emitted_at_utc=event.emitted_at_utc,
    )


def _timeline_kind(event_type: PolicyTraceEventType) -> TimelineEventKind:
    if event_type in (PolicyTraceEventType.action_selected, PolicyTraceEventType.action_published):
        return TimelineEventKind.action_event
    if event_type is PolicyTraceEventType.safety_guard:
        return TimelineEventKind.safety_guard
    if event_type is PolicyTraceEventType.error:
        return TimelineEventKind.error
    return TimelineEventKind.policy_event


def _run_events(
    *,
    score_report: ScoreReport | None,
    reward_report: RewardReport | None,
    failure_report: FailureReport | None,
    start_index: int,
) -> tuple[TimelineEvent, ...]:
    events: list[TimelineEvent] = []
    next_index = start_index
    if score_report is not None:
        events.append(
            TimelineEvent(
                event_index=next_index,
                event_kind=TimelineEventKind.official_score,
                elapsed_sec=0.0,
                source=score_report.source,
                leakage_class=LeakageClass.privileged_eval_signal,
                payload={
                    "total": score_report.total,
                    "trial_count": score_report.trial_count,
                    "trials": {
                        trial_id: trial_score.to_dict()
                        for trial_id, trial_score in score_report.trials.items()
                    },
                },
                emitted_at_utc=score_report.parsed_at_utc,
            )
        )
        next_index += 1
    if reward_report is not None:
        for term in reward_report.terms:
            events.append(_reward_event(term, next_index))
            next_index += 1
    if failure_report is not None:
        for label in failure_report.labels:
            events.append(_failure_event(label, next_index))
            next_index += 1
    return tuple(events)


def _reward_event(term: RewardTerm, event_index: int) -> TimelineEvent:
    return TimelineEvent(
        event_index=event_index,
        event_kind=TimelineEventKind.reward_term,
        elapsed_sec=0.0,
        source=term.source,
        leakage_class=term.leakage_class,
        payload=term.to_dict(),
    )


def _failure_event(label: FailureLabel, event_index: int) -> TimelineEvent:
    return TimelineEvent(
        event_index=event_index,
        event_kind=TimelineEventKind.failure_label,
        elapsed_sec=0.0,
        source=label.source,
        leakage_class=label.leakage_class,
        payload=label.to_dict(),
    )


def _events_by_trial(
    policy_events: tuple[PolicyTraceEvent, ...],
) -> dict[str, tuple[PolicyTraceEvent, ...]]:
    events_by_trial: dict[str, list[PolicyTraceEvent]] = {}
    for event in policy_events:
        events_by_trial.setdefault(event.trial_id, []).append(event)
    return {trial_id: tuple(events) for trial_id, events in events_by_trial.items()}


def _scores_by_policy_trial(
    events_by_trial: Mapping[str, tuple[PolicyTraceEvent, ...]],
    score_report: ScoreReport | None,
) -> dict[str, TrialScore]:
    if score_report is None:
        return {}
    policy_trial_ids = tuple(events_by_trial)
    official_trial_scores = tuple(score_report.trials.values())
    return {
        policy_trial_id: official_trial_scores[index]
        for index, policy_trial_id in enumerate(policy_trial_ids)
        if index < len(official_trial_scores)
    }


def _action_event_is_nonzero(event: TimelineEvent) -> bool:
    payload = event.payload.get("policy_payload")
    return isinstance(payload, Mapping) and action_payload_is_nonzero(payload)


def _behavior_clone_signal(event: TimelineEvent) -> TrainingSignal:
    return TrainingSignal(
        kind=TrainingSignalKind.behavior_clone_action,
        target="policy.action",
        weight=1.0,
        source=event.source,
        leakage_class=event.leakage_class,
        trial_id=event.trial_id,
        evidence={
            "event_index": event.event_index,
            "elapsed_sec": event.elapsed_sec,
            "action": event.payload.get("policy_payload", {}),
        },
    )


def _safety_signal(event: TimelineEvent) -> TrainingSignal:
    return TrainingSignal(
        kind=TrainingSignalKind.safety_guard_avoidance,
        target="policy.safety_guard",
        weight=1.0,
        source=event.source,
        leakage_class=event.leakage_class,
        trial_id=event.trial_id,
        evidence={
            "event_index": event.event_index,
            "elapsed_sec": event.elapsed_sec,
            "guard": event.payload.get("policy_payload", {}),
        },
    )


def _official_score_signal(event: TimelineEvent) -> TrainingSignal:
    return TrainingSignal(
        kind=TrainingSignalKind.official_score_term,
        target="evaluation.score.total",
        weight=1.0,
        source=event.source,
        leakage_class=event.leakage_class,
        evidence=event.payload,
    )


def _reward_signal(event: TimelineEvent) -> TrainingSignal:
    return TrainingSignal(
        kind=TrainingSignalKind.reward_term,
        target=str(event.payload.get("name", "reward.term")),
        weight=float(event.payload.get("value", 0.0)),
        source=event.source,
        leakage_class=event.leakage_class,
        trial_id=event.payload.get("trial_id") if isinstance(event.payload.get("trial_id"), str) else None,
        evidence=event.payload,
    )


def _failure_signal(event: TimelineEvent) -> TrainingSignal:
    payload = event.payload
    severity = payload.get("severity")
    return TrainingSignal(
        kind=TrainingSignalKind.failure_label,
        target=str(payload.get("kind", "failure.label")),
        weight=_failure_weight(severity),
        source=event.source,
        leakage_class=event.leakage_class,
        trial_id=payload.get("trial_id") if isinstance(payload.get("trial_id"), str) else None,
        evidence=payload,
    )


def _failure_weight(severity: Any) -> float:
    if severity == FailureSeverity.blocker.value:
        return 1.0
    if severity == FailureSeverity.warning.value:
        return 0.5
    return 0.25


def _optional_score_report(value: ScoreReport | Mapping[str, Any] | None) -> ScoreReport | None:
    if value is None:
        return None
    return value if isinstance(value, ScoreReport) else ScoreReport.from_dict(value)


def _optional_reward_report(value: RewardReport | Mapping[str, Any] | None) -> RewardReport | None:
    if value is None:
        return None
    return value if isinstance(value, RewardReport) else RewardReport.from_dict(value)


def _optional_failure_report(value: FailureReport | Mapping[str, Any] | None) -> FailureReport | None:
    if value is None:
        return None
    return value if isinstance(value, FailureReport) else FailureReport.from_dict(value)


def _validate_report_run_ids(
    run_id: str,
    reward_report: RewardReport | None,
    failure_report: FailureReport | None,
) -> None:
    if reward_report is not None and reward_report.run_id != run_id:
        raise HarnessIOError("reward_report.run_id does not match episode trace run_id")
    if failure_report is not None and failure_report.run_id != run_id:
        raise HarnessIOError("failure_report.run_id does not match episode trace run_id")


def _require_run_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessIOError("episode trace run_id must be a nonempty string")
    return value.strip()
