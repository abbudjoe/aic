"""Reducers for canonical episode traces and training signals."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from aic_signal_harness.artifacts import (
    HarnessIOError,
    file_uri_artifact_path,
    local_artifact_path,
    read_json,
    sha256_file,
)
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
from aic_signal_harness.reducers.policy_trace import read_policy_trace_events
from aic_signal_harness.reward import RewardReport, RewardSignalKind, RewardTerm
from aic_signal_harness.schemas import ArtifactRef, LeakageClass
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
    _validate_policy_event_sequence(run_id, policy_events)
    typed_score = _optional_score_report(score_report)
    typed_reward = _optional_reward_report(reward_report)
    typed_failure = _optional_failure_report(failure_report)
    _validate_report_run_ids(run_id, typed_reward, typed_failure)
    _validate_source_artifact_bindings(
        run_id,
        source_artifacts,
        policy_events,
        typed_score,
        typed_reward,
        typed_failure,
    )
    generated_at = (
        _episode_trace_generated_at(
            policy_events=policy_events,
            score_report=typed_score,
            reward_report=typed_reward,
            failure_report=typed_failure,
        )
        if generated_at_utc is None
        else generated_at_utc
    )

    events_by_trial = _events_by_trial(policy_events)
    official_trial_by_policy = _official_trial_ids_by_policy_trial(
        events_by_trial,
        require_complete=typed_score is not None,
    )
    trial_scores = _scores_by_policy_trial(
        official_trial_by_policy,
        typed_score,
    )
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
            trial_bounds={trial.trial_id: (trial.start_elapsed_sec, trial.end_elapsed_sec) for trial in trials},
            official_trial_by_policy=official_trial_by_policy,
        ),
        notes=(_episode_trace_note(typed_score, typed_reward, typed_failure),),
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
    _validate_source_trace(source_trace, trace)
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
            reward_signal = _reward_signal(event)
            if reward_signal is not None:
                signals.append(reward_signal)
        elif event.event_kind is TimelineEventKind.failure_label:
            signals.append(_failure_signal(event))
    return TrainingSignalReport(
        run_id=trace.run_id,
        generated_at_utc=generated_at,
        source_trace=source_trace,
        signals=tuple(signals),
        notes=(
            "Signals are offline-only extraction records and must not enter the live policy runtime.",
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
    payload = {
        "policy_event_type": event.event_type.value,
        "policy_payload": event.payload,
    }
    if event.official_trial_id is not None:
        payload["official_trial_id"] = event.official_trial_id
    return TimelineEvent(
        event_index=event.event_index,
        event_kind=_timeline_kind(event.event_type),
        elapsed_sec=event.elapsed_sec,
        source=event.source,
        leakage_class=event.leakage_class,
        payload=payload,
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
    trial_bounds: Mapping[str, tuple[float, float]],
    official_trial_by_policy: Mapping[str, str],
) -> tuple[TimelineEvent, ...]:
    events: list[TimelineEvent] = []
    next_index = start_index
    run_window = _run_evidence_window(trial_bounds)
    if score_report is not None:
        events.append(
            TimelineEvent(
                event_index=next_index,
                event_kind=TimelineEventKind.official_score,
                elapsed_sec=run_window["end_elapsed_sec"],
                source=score_report.source,
                leakage_class=LeakageClass.privileged_eval_signal,
                payload={
                    "event_scope": "post_hoc_run_summary",
                    "evidence_window": run_window,
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
            events.append(_reward_event(term, next_index, trial_bounds, run_window, official_trial_by_policy))
            next_index += 1
    if failure_report is not None:
        for label in failure_report.labels:
            events.append(_failure_event(label, next_index, trial_bounds, run_window, official_trial_by_policy))
            next_index += 1
    return tuple(events)


def _reward_event(
    term: RewardTerm,
    event_index: int,
    trial_bounds: Mapping[str, tuple[float, float]],
    run_window: Mapping[str, float],
    official_trial_by_policy: Mapping[str, str],
) -> TimelineEvent:
    evidence_window = _evidence_window_for_trial(
        term.trial_id,
        trial_bounds,
        run_window,
        official_trial_by_policy,
    )
    trace_trial_id = _trace_trial_id_for_label(
        term.trial_id,
        trial_bounds,
        official_trial_by_policy,
    )
    payload = {
        **term.to_dict(),
        "event_scope": "post_hoc_trial_label" if term.trial_id is not None else "post_hoc_run_summary",
        "evidence_window": evidence_window,
    }
    if trace_trial_id is not None and term.trial_id != trace_trial_id:
        payload["source_trial_id"] = term.trial_id
    return TimelineEvent(
        event_index=event_index,
        event_kind=TimelineEventKind.reward_term,
        elapsed_sec=evidence_window["end_elapsed_sec"],
        source=term.source,
        leakage_class=term.leakage_class,
        payload=payload,
        trial_id=trace_trial_id,
    )


def _failure_event(
    label: FailureLabel,
    event_index: int,
    trial_bounds: Mapping[str, tuple[float, float]],
    run_window: Mapping[str, float],
    official_trial_by_policy: Mapping[str, str],
) -> TimelineEvent:
    evidence_window = _evidence_window_for_trial(
        label.trial_id,
        trial_bounds,
        run_window,
        official_trial_by_policy,
    )
    trace_trial_id = _trace_trial_id_for_label(
        label.trial_id,
        trial_bounds,
        official_trial_by_policy,
    )
    payload = {
        **label.to_dict(),
        "event_scope": "post_hoc_trial_label" if label.trial_id is not None else "post_hoc_run_summary",
        "evidence_window": evidence_window,
    }
    if trace_trial_id is not None and label.trial_id != trace_trial_id:
        payload["source_trial_id"] = label.trial_id
    return TimelineEvent(
        event_index=event_index,
        event_kind=TimelineEventKind.failure_label,
        elapsed_sec=evidence_window["end_elapsed_sec"],
        source=label.source,
        leakage_class=label.leakage_class,
        payload=payload,
        trial_id=trace_trial_id,
    )


def _events_by_trial(
    policy_events: tuple[PolicyTraceEvent, ...],
) -> dict[str, tuple[PolicyTraceEvent, ...]]:
    events_by_trial: dict[str, list[PolicyTraceEvent]] = {}
    for event in policy_events:
        events_by_trial.setdefault(event.trial_id, []).append(event)
    return {trial_id: tuple(events) for trial_id, events in events_by_trial.items()}


def _official_trial_ids_by_policy_trial(
    events_by_trial: Mapping[str, tuple[PolicyTraceEvent, ...]],
    *,
    require_complete: bool,
) -> dict[str, str]:
    policy_to_official: dict[str, str] = {}
    errors: list[str] = []
    for policy_trial_id, events in events_by_trial.items():
        official_ids = tuple(event.official_trial_id for event in events if event.official_trial_id is not None)
        missing_count = sum(event.official_trial_id is None for event in events)
        distinct_official_ids = sorted(set(official_ids))
        if len(distinct_official_ids) > 1:
            errors.append(
                f"policy trial {policy_trial_id!r} carries multiple official_trial_id values: "
                + ", ".join(distinct_official_ids)
            )
        elif distinct_official_ids:
            if missing_count:
                errors.append(
                    f"policy trial {policy_trial_id!r} must carry official_trial_id on every event"
                )
            else:
                policy_to_official[policy_trial_id] = distinct_official_ids[0]
        elif require_complete:
            errors.append(
                f"policy trial {policy_trial_id!r} must carry official_trial_id for score binding"
            )
    if errors:
        raise HarnessIOError("; ".join(errors))
    return policy_to_official


def _scores_by_policy_trial(
    policy_to_official: Mapping[str, str],
    score_report: ScoreReport | None,
) -> dict[str, TrialScore]:
    if score_report is None:
        return {}
    official_to_policy: dict[str, str] = {}
    duplicate_official = []
    for policy_trial_id, official_trial_id in policy_to_official.items():
        previous = official_to_policy.setdefault(official_trial_id, policy_trial_id)
        if previous != policy_trial_id:
            duplicate_official.append(official_trial_id)
    if duplicate_official:
        raise HarnessIOError(
            "policy trial ids map ambiguously to official scoring trials: "
            + ", ".join(sorted(set(duplicate_official)))
        )
    policy_official_ids = set(policy_to_official.values())
    score_official_ids = set(score_report.trials)
    if policy_official_ids != score_official_ids:
        missing_scores = sorted(policy_official_ids - score_official_ids)
        extra_scores = sorted(score_official_ids - policy_official_ids)
        details = []
        if missing_scores:
            details.append("missing official scores for " + ", ".join(missing_scores))
        if extra_scores:
            details.append("unmatched official scores " + ", ".join(extra_scores))
        raise HarnessIOError("official score trials must explicitly match policy trials: " + "; ".join(details))
    return {
        policy_trial_id: score_report.trials[official_trial_id]
        for policy_trial_id, official_trial_id in policy_to_official.items()
    }


def _action_event_is_nonzero(event: TimelineEvent) -> bool:
    payload = event.payload.get("policy_payload")
    return isinstance(payload, Mapping) and action_payload_is_nonzero(payload)


def _behavior_clone_signal(event: TimelineEvent) -> TrainingSignal:
    if event.leakage_class is not LeakageClass.legal_policy_action_output:
        raise HarnessIOError(
            "behavior-clone action signals require legal_policy_action_output action events"
        )
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


def _reward_signal(event: TimelineEvent) -> TrainingSignal | None:
    if event.payload.get("signal_kind") == RewardSignalKind.official_score.value:
        return None
    return TrainingSignal(
        kind=TrainingSignalKind.reward_term,
        target=str(event.payload.get("name", "reward.term")),
        weight=1.0,
        source=event.source,
        leakage_class=event.leakage_class,
        trial_id=event.trial_id,
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
        trial_id=event.trial_id,
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


def _validate_policy_event_sequence(run_id: str, policy_events: tuple[PolicyTraceEvent, ...]) -> None:
    errors: list[str] = []
    if any(not isinstance(event, PolicyTraceEvent) for event in policy_events):
        errors.append("episode trace policy_events must be PolicyTraceEvent instances")
    if errors:
        raise HarnessIOError("; ".join(errors))
    observed_run_ids = {event.run_id for event in policy_events if isinstance(event, PolicyTraceEvent)}
    if observed_run_ids != {run_id}:
        errors.append("policy trace events do not match episode trace run_id")
    expected_indices = tuple(range(len(policy_events)))
    actual_indices = tuple(event.event_index for event in policy_events)
    if actual_indices != expected_indices:
        errors.append("policy trace event_index values must be contiguous and match event order")
    elapsed_by_trial: dict[str, float] = {}
    for event in policy_events:
        previous_elapsed = elapsed_by_trial.get(event.trial_id)
        if previous_elapsed is not None and event.elapsed_sec < previous_elapsed:
            errors.append(f"policy trace elapsed_sec must be nondecreasing within {event.trial_id}")
            break
        elapsed_by_trial[event.trial_id] = event.elapsed_sec
    if errors:
        raise HarnessIOError("; ".join(errors))


def _run_evidence_window(trial_bounds: Mapping[str, tuple[float, float]]) -> Mapping[str, float]:
    starts = tuple(bounds[0] for bounds in trial_bounds.values())
    ends = tuple(bounds[1] for bounds in trial_bounds.values())
    return {
        "start_elapsed_sec": min(starts),
        "end_elapsed_sec": max(ends),
    }


def _evidence_window_for_trial(
    trial_id: str | None,
    trial_bounds: Mapping[str, tuple[float, float]],
    run_window: Mapping[str, float],
    official_trial_by_policy: Mapping[str, str],
) -> Mapping[str, float]:
    if trial_id is None:
        return run_window
    trace_trial_id = _trace_trial_id_for_label(trial_id, trial_bounds, official_trial_by_policy)
    if trace_trial_id is None:
        raise HarnessIOError(f"post-hoc label references unknown trial_id {trial_id!r}")
    start, end = trial_bounds[trace_trial_id]
    return {"start_elapsed_sec": start, "end_elapsed_sec": end}


def _trace_trial_id_for_label(
    trial_id: str | None,
    trial_bounds: Mapping[str, tuple[float, float]],
    official_trial_by_policy: Mapping[str, str],
) -> str | None:
    if trial_id is None:
        return None
    if trial_id in trial_bounds:
        return trial_id
    matches = tuple(
        policy_trial_id
        for policy_trial_id, official_trial_id in official_trial_by_policy.items()
        if policy_trial_id in trial_bounds and official_trial_id == trial_id
    )
    if len(matches) == 1:
        return matches[0]
    return None


def _episode_trace_note(
    score_report: ScoreReport | None,
    reward_report: RewardReport | None,
    failure_report: FailureReport | None,
) -> str:
    inputs = ["policy JSONL"]
    if score_report is not None:
        inputs.append("official score")
    if reward_report is not None:
        inputs.append("reward report")
    if failure_report is not None:
        inputs.append("failure report")
    return "Canonical episode trace fused from " + ", ".join(inputs) + "."


def _episode_trace_generated_at(
    *,
    policy_events: tuple[PolicyTraceEvent, ...],
    score_report: ScoreReport | None,
    reward_report: RewardReport | None,
    failure_report: FailureReport | None,
) -> str:
    candidates = [("policy trace last event emitted_at_utc", policy_events[-1].emitted_at_utc)]
    if score_report is not None and score_report.parsed_at_utc is not None:
        candidates.append(("score_report.parsed_at_utc", score_report.parsed_at_utc))
    if reward_report is not None:
        candidates.append(("reward_report.generated_at_utc", reward_report.generated_at_utc))
    if failure_report is not None:
        candidates.append(("failure_report.generated_at_utc", failure_report.generated_at_utc))
    return _max_utc_timestamp(candidates)


def _max_utc_timestamp(candidates: list[tuple[str, str]]) -> str:
    parsed_candidates = [
        (_parse_utc_timestamp(value, field_name), value.strip())
        for field_name, value in candidates
    ]
    timestamp, _ = max(parsed_candidates, key=lambda item: item[0])
    return timestamp.isoformat().replace("+00:00", "Z")


def _parse_utc_timestamp(value: str, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise HarnessIOError(f"{field_name} must be a nonempty UTC timestamp")
    text = value.strip()
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HarnessIOError(f"{field_name} must be an ISO-8601 UTC timestamp") from exc
    if timestamp.tzinfo is None:
        raise HarnessIOError(f"{field_name} must include a timezone")
    return timestamp.astimezone(timezone.utc)


def _validate_source_trace(source_trace: ArtifactRef, trace: EpisodeTrace) -> None:
    if source_trace.kind != "episode_trace":
        raise HarnessIOError("source_trace.kind must be 'episode_trace'")
    if source_trace.sha256 is None:
        raise HarnessIOError("source_trace.sha256 must be set")
    if source_trace.provenance.get("run_id") != trace.run_id:
        raise HarnessIOError("source_trace provenance run_id must match episode trace run_id")
    if "producer" not in source_trace.provenance:
        raise HarnessIOError("source_trace provenance must include producer")
    if "derivation" not in source_trace.provenance:
        raise HarnessIOError("source_trace provenance must include derivation")
    trace_path = local_artifact_path(
        path=source_trace.path,
        uri=source_trace.uri,
        field_name="source_trace",
    )
    if trace_path is not None:
        if not trace_path.exists():
            raise HarnessIOError("source_trace.path must exist when set")
        if not trace_path.is_file():
            raise HarnessIOError("source_trace.path must point to an episode trace file")
        if sha256_file(trace_path) != source_trace.sha256:
            raise HarnessIOError("source_trace.sha256 must match source_trace.path")
        disk_trace = EpisodeTrace.from_dict(read_json(trace_path))
        if disk_trace != trace:
            raise HarnessIOError("source_trace.path does not match supplied episode trace")
        return
    if source_trace.uri is None:
        raise HarnessIOError("source_trace.uri must be set for URI-only episode trace artifacts")
    canonical_digest = _canonical_episode_trace_sha256(trace)
    if source_trace.sha256 != canonical_digest:
        raise HarnessIOError("source_trace.sha256 must match supplied episode trace")


def _canonical_episode_trace_sha256(trace: EpisodeTrace) -> str:
    payload = (
        json.dumps(trace.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_source_artifact_bindings(
    run_id: str,
    source_artifacts: tuple[ArtifactRef, ...],
    policy_events: tuple[PolicyTraceEvent, ...],
    score_report: ScoreReport | None,
    reward_report: RewardReport | None,
    failure_report: FailureReport | None,
) -> None:
    errors: list[str] = []
    for artifact in source_artifacts:
        if artifact.provenance.get("run_id") != run_id:
            errors.append(
                f"source artifact {artifact.kind!r} provenance.run_id must match episode trace run_id"
            )
    scoring_artifacts = tuple(
        artifact for artifact in source_artifacts if artifact.kind == "scoring_yaml"
    )
    if len(scoring_artifacts) != 1:
        errors.append("episode trace.source_artifacts must include exactly one scoring_yaml")
    else:
        scoring_artifact = scoring_artifacts[0]
        if score_report is not None:
            if not _artifact_names_source(scoring_artifact, score_report.source):
                errors.append("scoring_yaml source artifact must match score_report.source")
        errors.extend(_scoring_artifact_digest_errors(scoring_artifact))
    policy_artifacts = tuple(
        artifact for artifact in source_artifacts if artifact.kind == "policy_trace_jsonl"
    )
    if len(policy_artifacts) != 1:
        errors.append("episode trace.source_artifacts must include exactly one policy_trace_jsonl")
    else:
        policy_artifact = policy_artifacts[0]
        if policy_artifact.sha256 is None:
            errors.append("policy_trace_jsonl source artifact sha256 must be set")
        try:
            policy_path = local_artifact_path(
                path=policy_artifact.path,
                uri=policy_artifact.uri,
                field_name="policy_trace_jsonl source artifact",
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
            policy_path = None
        if policy_path is None:
            if policy_artifact.uri is None:
                errors.append("policy_trace_jsonl source artifact uri must be set when path is absent")
        else:
            if not policy_path.exists():
                errors.append("policy_trace_jsonl source artifact path must exist")
            elif not policy_path.is_file():
                errors.append("policy_trace_jsonl source artifact path must be a file")
            elif policy_artifact.sha256 != sha256_file(policy_path):
                errors.append("policy_trace_jsonl source artifact sha256 must match path")
            else:
                persisted_events = read_policy_trace_events(policy_path)
                if persisted_events != policy_events:
                    errors.append("policy_trace_jsonl source artifact must match policy_events")
    errors.extend(
        _report_source_artifact_errors(
            source_artifacts=source_artifacts,
            kind="reward_report",
            report=reward_report,
        )
    )
    errors.extend(
        _report_source_artifact_errors(
            source_artifacts=source_artifacts,
            kind="failure_report",
            report=failure_report,
        )
    )
    if errors:
        raise HarnessIOError("; ".join(errors))


def _artifact_names_source(artifact: ArtifactRef, source: str) -> bool:
    if artifact.path == source or artifact.uri == source:
        return True
    try:
        local_path = local_artifact_path(
            path=artifact.path,
            uri=artifact.uri,
            field_name=f"{artifact.kind} source artifact",
        )
    except HarnessIOError:
        return False
    if local_path is not None:
        try:
            resolved = local_path.resolve(strict=False)
        except OSError:
            return False
        if str(resolved) == source:
            return True
        if source.startswith("file:"):
            try:
                source_path = file_uri_artifact_path(source, field_name="score_report.source")
                return source_path is not None and resolved == source_path.resolve(strict=False)
            except (HarnessIOError, OSError):
                return False
    return False


def _scoring_artifact_digest_errors(artifact: ArtifactRef) -> list[str]:
    if artifact.sha256 is None:
        return ["scoring_yaml source artifact sha256 must be set"]
    scoring_path = local_artifact_path(
        path=artifact.path,
        uri=artifact.uri,
        field_name="scoring_yaml source artifact",
    )
    if scoring_path is None:
        return []
    if not scoring_path.exists():
        return ["scoring_yaml source artifact path must exist"]
    if not scoring_path.is_file():
        return ["scoring_yaml source artifact path must be a file"]
    if sha256_file(scoring_path) != artifact.sha256:
        return ["scoring_yaml source artifact sha256 must match path"]
    return []


def _report_source_artifact_errors(
    *,
    source_artifacts: tuple[ArtifactRef, ...],
    kind: str,
    report: RewardReport | FailureReport | None,
) -> list[str]:
    artifacts = tuple(artifact for artifact in source_artifacts if artifact.kind == kind)
    if report is None:
        if len(artifacts) > 1:
            return [f"episode trace.source_artifacts must include at most one {kind}"]
        if len(artifacts) == 1:
            return _report_artifact_binding_errors(
                artifacts[0],
                kind=kind,
                report=None,
                source_artifacts=source_artifacts,
            )
        return []
    if len(artifacts) != 1:
        return [f"episode trace.source_artifacts must include exactly one {kind}"]
    return _report_artifact_binding_errors(
        artifacts[0],
        kind=kind,
        report=report,
        source_artifacts=source_artifacts,
    )


def _report_artifact_binding_errors(
    artifact: ArtifactRef,
    *,
    kind: str,
    report: RewardReport | FailureReport | None,
    source_artifacts: tuple[ArtifactRef, ...],
) -> list[str]:
    errors: list[str] = []
    if artifact.sha256 is None:
        errors.append(f"{kind} source artifact sha256 must be set")
    if report is not None:
        errors.extend(
            _derived_report_artifact_provenance_errors(
                artifact,
                kind=kind,
                report=report,
                source_artifacts=source_artifacts,
            )
        )
    try:
        report_path = local_artifact_path(
            path=artifact.path,
            uri=artifact.uri,
            field_name=f"{kind} source artifact",
        )
    except HarnessIOError as exc:
        return [str(exc)]
    if report_path is None:
        if artifact.uri is None:
            errors.append(f"{kind} source artifact uri must be set when path is absent")
        return errors
    if not report_path.exists():
        errors.append(f"{kind} source artifact path must exist")
    elif not report_path.is_file():
        errors.append(f"{kind} source artifact path must be a file")
    elif artifact.sha256 != sha256_file(report_path):
        errors.append(f"{kind} source artifact sha256 must match path")
    elif report is not None:
        try:
            parsed_report: RewardReport | FailureReport
            if kind == "reward_report":
                parsed_report = RewardReport.from_dict(read_json(report_path))
            else:
                parsed_report = FailureReport.from_dict(read_json(report_path))
        except HarnessIOError as exc:
            errors.append(f"{kind} source artifact must parse as {kind}: {exc}")
        else:
            if parsed_report != report:
                errors.append(f"{kind} source artifact must match supplied report")
    return errors


def _derived_report_artifact_provenance_errors(
    artifact: ArtifactRef,
    *,
    kind: str,
    report: RewardReport | FailureReport,
    source_artifacts: tuple[ArtifactRef, ...],
) -> list[str]:
    errors: list[str] = []
    derivation = artifact.provenance.get("derivation")
    if not isinstance(derivation, str) or not derivation.strip():
        errors.append(f"{kind} source artifact provenance.derivation must be set")
    declared_sources = artifact.provenance.get("source_artifacts")
    if not isinstance(declared_sources, (list, tuple)) or not declared_sources:
        errors.append(f"{kind} source artifact provenance.source_artifacts must be a nonempty list")
        return errors
    actual_source_pairs = {
        (source_artifact.kind, source_artifact.sha256)
        for source_artifact in source_artifacts
        if source_artifact.kind not in {"reward_report", "failure_report"}
        and source_artifact.sha256 is not None
    }
    scoring_source_pairs = {
        pair for pair in actual_source_pairs if pair[0] == "scoring_yaml"
    }
    promotion_source_pairs = {
        pair for pair in actual_source_pairs if pair[0] == "promotion_decision_snapshot"
    }
    claimed_pairs: set[tuple[str, str]] = set()
    for index, declared_source in enumerate(declared_sources):
        if not isinstance(declared_source, Mapping):
            errors.append(
                f"{kind} source artifact provenance.source_artifacts[{index}] must be a mapping"
            )
            continue
        source_kind = declared_source.get("kind")
        source_sha256 = declared_source.get("sha256")
        if not isinstance(source_kind, str) or not source_kind.strip():
            errors.append(
                f"{kind} source artifact provenance.source_artifacts[{index}].kind must be set"
            )
        if not _looks_like_sha256(source_sha256):
            errors.append(
                f"{kind} source artifact provenance.source_artifacts[{index}].sha256 "
                "must be 64 hexadecimal characters"
            )
        if isinstance(source_kind, str) and isinstance(source_sha256, str):
            claimed_pair = (source_kind.strip(), source_sha256)
            claimed_pairs.add(claimed_pair)
            if claimed_pair not in actual_source_pairs:
                errors.append(
                    f"{kind} source artifact provenance.source_artifacts[{index}] "
                    "must match an episode trace source artifact"
                )
    if scoring_source_pairs and not (claimed_pairs & scoring_source_pairs):
        errors.append(
            f"{kind} source artifact provenance.source_artifacts must include "
            "the scoring_yaml source artifact"
        )
    if _report_contains_promotion_evidence(report) and not (claimed_pairs & promotion_source_pairs):
        errors.append(
            f"{kind} source artifact provenance.source_artifacts must include "
            "a promotion_decision_snapshot source artifact"
        )
    return errors


def _report_contains_promotion_evidence(report: RewardReport | FailureReport) -> bool:
    if isinstance(report, RewardReport):
        return any(term.source == "promotion_decision" for term in report.terms)
    return any(label.source == "promotion_decision" for label in report.labels)


def _looks_like_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)
