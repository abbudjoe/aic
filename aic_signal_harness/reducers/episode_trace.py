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
    _canonical_event_projection_sha256,
    _hdf5_report_matches_local_bytes_errors,
)
from aic_signal_harness.failure import FailureLabel, FailureReport, FailureSeverity
from aic_signal_harness.hdf5_dataset import Hdf5DatasetReport
from aic_signal_harness.mcap_eval import McapEvalBundleReport, McapEvalTrialReport
from aic_signal_harness.policy_trace import (
    PolicyTraceEvent,
    PolicyTraceEventType,
    action_payload_is_nonzero,
)
from aic_signal_harness.reducers.policy_trace import read_policy_trace_events
from aic_signal_harness.reward import RewardReport, RewardSignalKind, RewardTerm
from aic_signal_harness.schemas import ArtifactRef, LeakageClass
from aic_signal_harness.scoring import ScoreReport, TrialScore, parse_scoring_yaml
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
    mcap_eval_bundle: McapEvalBundleReport | Mapping[str, Any] | None = None,
    hdf5_dataset_report: Hdf5DatasetReport | Mapping[str, Any] | None = None,
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
    typed_mcap = _optional_mcap_eval_bundle(mcap_eval_bundle)
    typed_hdf5 = _optional_hdf5_dataset_report(hdf5_dataset_report)
    if typed_mcap is not None:
        raise HarnessIOError(
            "MCAP evidence requires the dedicated byte rederive slice/analyzer injection"
        )
    _validate_report_run_ids(run_id, typed_reward, typed_failure)
    _validate_source_artifact_bindings(
        run_id,
        source_artifacts,
        policy_events,
        typed_score,
        typed_reward,
        typed_failure,
        typed_mcap,
        typed_hdf5,
    )
    generated_at = (
        _episode_trace_generated_at(
            policy_events=policy_events,
            score_report=typed_score,
            reward_report=typed_reward,
            failure_report=typed_failure,
            mcap_eval_bundle=typed_mcap,
            hdf5_dataset_report=typed_hdf5,
        )
        if generated_at_utc is None
        else generated_at_utc
    )

    events_by_trial = _events_by_trial(policy_events)
    official_trial_by_policy = _official_trial_ids_by_policy_trial(
        events_by_trial,
        require_complete=typed_score is not None or typed_mcap is not None,
    )
    trial_scores = _scores_by_policy_trial(
        official_trial_by_policy,
        typed_score,
    )
    mcap_trials = _mcap_trials_by_policy_trial(
        official_trial_by_policy,
        typed_mcap,
    )
    trials = tuple(
        _trial_trace(
            trial_id=trial_id,
            policy_events=trial_events,
            score=trial_scores.get(trial_id),
        )
        for trial_id, trial_events in events_by_trial.items()
    )
    run_events = _run_events(
        score_report=typed_score,
        reward_report=typed_reward,
        failure_report=typed_failure,
        mcap_trials=mcap_trials,
        mcap_eval_bundle=typed_mcap,
        hdf5_dataset_report=typed_hdf5,
        start_index=sum(trial.event_count for trial in trials),
        trial_bounds={trial.trial_id: (trial.start_elapsed_sec, trial.end_elapsed_sec) for trial in trials},
        official_trial_by_policy=official_trial_by_policy,
    )
    return EpisodeTrace(
        run_id=run_id,
        generated_at_utc=generated_at,
        source_artifacts=_source_artifacts_with_event_projection(
            source_artifacts,
            run_events,
            score_report=typed_score,
            mcap_eval_bundle=typed_mcap,
            hdf5_dataset_report=typed_hdf5,
        ),
        trials=trials,
        run_events=run_events,
        notes=(_episode_trace_note(typed_score, typed_reward, typed_failure, typed_mcap, typed_hdf5),),
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
        observation_event_count=event_kinds[TimelineEventKind.observation_event],
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
    if event_type is PolicyTraceEventType.observation:
        return TimelineEventKind.observation_event
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
    mcap_trials: Mapping[str, McapEvalTrialReport],
    mcap_eval_bundle: McapEvalBundleReport | None,
    hdf5_dataset_report: Hdf5DatasetReport | None,
    start_index: int,
    trial_bounds: Mapping[str, tuple[float, float]],
    official_trial_by_policy: Mapping[str, str],
) -> tuple[TimelineEvent, ...]:
    events: list[TimelineEvent] = []
    next_index = start_index
    run_window = _run_evidence_window(trial_bounds)
    if hdf5_dataset_report is not None:
        events.append(
            _dataset_summary_event(
                hdf5_dataset_report,
                next_index,
                run_window,
                _report_sha256(hdf5_dataset_report.to_dict()),
            )
        )
        next_index += 1
    if mcap_eval_bundle is not None:
        mcap_report_sha256 = _report_sha256(mcap_eval_bundle.to_dict())
        for policy_trial_id, mcap_trial in mcap_trials.items():
            events.append(
                _controller_summary_event(
                    mcap_trial=mcap_trial,
                    event_index=next_index,
                    policy_trial_id=policy_trial_id,
                    trial_bounds=trial_bounds,
                    analyzed_at_utc=mcap_eval_bundle.analyzed_at_utc,
                    source_report_sha256=mcap_report_sha256,
                )
            )
            next_index += 1
            events.append(
                _contact_evidence_event(
                    mcap_trial=mcap_trial,
                    event_index=next_index,
                    policy_trial_id=policy_trial_id,
                    trial_bounds=trial_bounds,
                    analyzed_at_utc=mcap_eval_bundle.analyzed_at_utc,
                    source_report_sha256=mcap_report_sha256,
                )
            )
            next_index += 1
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


def _source_artifacts_with_event_projection(
    source_artifacts: tuple[ArtifactRef, ...],
    run_events: tuple[TimelineEvent, ...],
    *,
    score_report: ScoreReport | None,
    mcap_eval_bundle: McapEvalBundleReport | None,
    hdf5_dataset_report: Hdf5DatasetReport | None,
) -> tuple[ArtifactRef, ...]:
    mcap_events = tuple(
        event
        for event in run_events
        if event.event_kind in {
            TimelineEventKind.controller_summary,
            TimelineEventKind.contact_evidence,
        }
    )
    dataset_events = tuple(
        event
        for event in run_events
        if event.event_kind is TimelineEventKind.dataset_episode_summary
    )
    score_events = tuple(
        event for event in run_events if event.event_kind is TimelineEventKind.official_score
    )
    return tuple(
        _source_artifact_with_projection(
            artifact,
            score_events=score_events,
            mcap_events=mcap_events,
            dataset_events=dataset_events,
            score_report=score_report,
            mcap_eval_bundle=mcap_eval_bundle,
            hdf5_dataset_report=hdf5_dataset_report,
        )
        for artifact in source_artifacts
    )


def _source_artifact_with_projection(
    artifact: ArtifactRef,
    *,
    score_events: tuple[TimelineEvent, ...],
    mcap_events: tuple[TimelineEvent, ...],
    dataset_events: tuple[TimelineEvent, ...],
    score_report: ScoreReport | None,
    mcap_eval_bundle: McapEvalBundleReport | None,
    hdf5_dataset_report: Hdf5DatasetReport | None,
) -> ArtifactRef:
    if artifact.kind == "scoring_yaml" and score_events and score_report is not None:
        return _artifact_with_event_projection(
            artifact,
            score_events,
            source_report=score_report.to_dict(),
            include_source_report=True,
        )
    if artifact.kind == "mcap_eval_bundle" and mcap_events and mcap_eval_bundle is not None:
        return _artifact_with_event_projection(artifact, mcap_events, source_report=mcap_eval_bundle.to_dict())
    if artifact.kind == "hdf5_dataset" and dataset_events and hdf5_dataset_report is not None:
        return _artifact_with_event_projection(artifact, dataset_events, source_report=hdf5_dataset_report.to_dict())
    return artifact


def _artifact_with_event_projection(
    artifact: ArtifactRef,
    events: tuple[TimelineEvent, ...],
    *,
    source_report: Mapping[str, Any],
    include_source_report: bool = False,
) -> ArtifactRef:
    event_sha256 = _canonical_event_projection_sha256(events)
    report_sha256 = _report_sha256(source_report)
    existing_report_sha256 = artifact.provenance.get("report_sha256")
    if existing_report_sha256 is not None and existing_report_sha256 != report_sha256:
        raise HarnessIOError(
            f"{artifact.kind} source artifact provenance.report_sha256 must match supplied report"
        )
    existing_report_projection = artifact.provenance.get("source_report_event_projection_sha256")
    if existing_report_projection is not None and existing_report_projection != event_sha256:
        raise HarnessIOError(
            f"{artifact.kind} source artifact provenance.source_report_event_projection_sha256 "
            "must match derived events"
        )
    existing = artifact.provenance.get("canonical_event_projection_sha256")
    if existing is not None and existing != event_sha256:
        raise HarnessIOError(
            f"{artifact.kind} source artifact provenance.canonical_event_projection_sha256 "
            "must match derived events"
        )
    existing_source_report = artifact.provenance.get("source_report")
    if existing_source_report is not None:
        if not include_source_report:
            raise HarnessIOError(
                f"{artifact.kind} source artifact provenance.source_report must not be set; "
                "use the typed report source artifact instead"
            )
        if not isinstance(existing_source_report, Mapping):
            raise HarnessIOError(
                f"{artifact.kind} source artifact provenance.source_report must be a mapping"
            )
        if _report_sha256(existing_source_report) != report_sha256:
            raise HarnessIOError(
                f"{artifact.kind} source artifact provenance.source_report must match supplied report"
            )
    provenance = {key: value for key, value in artifact.provenance.items() if key != "source_report"}
    provenance.update(
        {
            "report_sha256": report_sha256,
            "source_report_event_projection_sha256": event_sha256,
            "canonical_event_projection_sha256": event_sha256,
        }
    )
    if include_source_report:
        provenance["source_report"] = source_report
    return ArtifactRef(
        kind=artifact.kind,
        path=artifact.path,
        uri=artifact.uri,
        sha256=artifact.sha256,
        provenance=provenance,
    )


def _dataset_summary_event(
    report: Hdf5DatasetReport,
    event_index: int,
    run_window: Mapping[str, float],
    source_report_sha256: str,
) -> TimelineEvent:
    return TimelineEvent(
        event_index=event_index,
        event_kind=TimelineEventKind.dataset_episode_summary,
        elapsed_sec=run_window["start_elapsed_sec"],
        source=report.source,
        leakage_class=LeakageClass.privileged_training_signal,
        payload={
            "event_scope": "offline_dataset_summary",
            "source_report_sha256": source_report_sha256,
            "evidence_window": run_window,
            "validated_at_utc": report.validated_at_utc,
            "ok": report.ok,
            "episode_count": report.episode_count,
            "step_count": report.step_count,
            "episode_lengths": list(report.episode_lengths),
            "episode_offsets": list(report.episode_offsets),
            "required_datasets": list(report.required_datasets),
            "missing_datasets": list(report.missing_datasets),
            "observation_datasets": _dataset_stats_payload(
                report,
                ("pixels", "left_pixels", "right_pixels", "proprio", "state"),
            ),
            "action_datasets": _dataset_stats_payload(report, ("action",)),
            "task_datasets": _dataset_stats_payload(
                report,
                ("task_id", "plug_type", "port_type", "target_module_name"),
            ),
        },
        emitted_at_utc=report.validated_at_utc,
    )


def _controller_summary_event(
    *,
    mcap_trial: McapEvalTrialReport,
    event_index: int,
    policy_trial_id: str,
    trial_bounds: Mapping[str, tuple[float, float]],
    analyzed_at_utc: str,
    source_report_sha256: str,
) -> TimelineEvent:
    evidence_window = _trial_evidence_window(policy_trial_id, trial_bounds)
    payload: dict[str, Any] = {
        "event_scope": "post_hoc_trial_evidence",
        "source_report_sha256": source_report_sha256,
        "official_trial_id": mcap_trial.trial_id,
        "evidence_window": evidence_window,
        "mcap_trial_source": mcap_trial.source,
        "controller_state_count": mcap_trial.controller_state_count,
        "pose_command_count": mcap_trial.pose_command_count,
    }
    for field_name in (
        "controller_stamp_start_sec",
        "controller_stamp_end_sec",
        "controller_duration_sec",
    ):
        field_value = getattr(mcap_trial, field_name)
        if field_value is not None:
            payload[field_name] = field_value
    for field_name in ("final_tcp_position", "final_tcp_error"):
        field_value = getattr(mcap_trial, field_name)
        if field_value is not None:
            payload[field_name] = list(field_value)
    if mcap_trial.task_hints is not None:
        payload["task_hints"] = mcap_trial.task_hints.to_dict()
    return TimelineEvent(
        event_index=event_index,
        event_kind=TimelineEventKind.controller_summary,
        elapsed_sec=evidence_window["end_elapsed_sec"],
        source=mcap_trial.source,
        leakage_class=LeakageClass.privileged_eval_signal,
        payload=payload,
        trial_id=policy_trial_id,
        emitted_at_utc=analyzed_at_utc,
    )


def _contact_evidence_event(
    *,
    mcap_trial: McapEvalTrialReport,
    event_index: int,
    policy_trial_id: str,
    trial_bounds: Mapping[str, tuple[float, float]],
    analyzed_at_utc: str,
    source_report_sha256: str,
) -> TimelineEvent:
    evidence_window = _trial_evidence_window(policy_trial_id, trial_bounds)
    payload: dict[str, Any] = {
        "event_scope": "post_hoc_trial_evidence",
        "source_report_sha256": source_report_sha256,
        "official_trial_id": mcap_trial.trial_id,
        "evidence_window": evidence_window,
        "mcap_trial_source": mcap_trial.source,
        "off_limit_contact_count": mcap_trial.off_limit_contact_count,
    }
    if mcap_trial.first_off_limit_contact is not None:
        payload["first_off_limit_contact"] = mcap_trial.first_off_limit_contact.to_dict()
    return TimelineEvent(
        event_index=event_index,
        event_kind=TimelineEventKind.contact_evidence,
        elapsed_sec=evidence_window["end_elapsed_sec"],
        source=mcap_trial.source,
        leakage_class=LeakageClass.privileged_eval_signal,
        payload=payload,
        trial_id=policy_trial_id,
        emitted_at_utc=analyzed_at_utc,
    )


def _dataset_stats_payload(
    report: Hdf5DatasetReport,
    dataset_names: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    return {
        dataset_name: report.datasets[dataset_name].to_dict()
        for dataset_name in dataset_names
        if dataset_name in report.datasets
    }


def _trial_evidence_window(
    trial_id: str,
    trial_bounds: Mapping[str, tuple[float, float]],
) -> Mapping[str, float]:
    start_elapsed_sec, end_elapsed_sec = trial_bounds[trial_id]
    return {
        "start_elapsed_sec": start_elapsed_sec,
        "end_elapsed_sec": end_elapsed_sec,
    }


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


def _mcap_trials_by_policy_trial(
    policy_to_official: Mapping[str, str],
    mcap_eval_bundle: McapEvalBundleReport | None,
) -> dict[str, McapEvalTrialReport]:
    if mcap_eval_bundle is None:
        return {}
    duplicate_official = _duplicate_values(policy_to_official)
    if duplicate_official:
        raise HarnessIOError(
            "policy trial ids map ambiguously to MCAP eval trials: "
            + ", ".join(duplicate_official)
        )
    official_to_mcap = {trial.trial_id: trial for trial in mcap_eval_bundle.trials}
    policy_official_ids = set(policy_to_official.values())
    mcap_official_ids = set(official_to_mcap)
    if policy_official_ids != mcap_official_ids:
        missing_mcap = sorted(policy_official_ids - mcap_official_ids)
        extra_mcap = sorted(mcap_official_ids - policy_official_ids)
        details = []
        if missing_mcap:
            details.append("missing MCAP evidence for " + ", ".join(missing_mcap))
        if extra_mcap:
            details.append("unmatched MCAP evidence " + ", ".join(extra_mcap))
        raise HarnessIOError("MCAP eval trials must explicitly match policy trials: " + "; ".join(details))
    return {
        policy_trial_id: official_to_mcap[official_trial_id]
        for policy_trial_id, official_trial_id in policy_to_official.items()
    }


def _duplicate_values(value: Mapping[str, str]) -> list[str]:
    counts = Counter(value.values())
    return sorted(item for item, count in counts.items() if count > 1)


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


def _optional_mcap_eval_bundle(
    value: McapEvalBundleReport | Mapping[str, Any] | None,
) -> McapEvalBundleReport | None:
    if value is None:
        return None
    return value if isinstance(value, McapEvalBundleReport) else McapEvalBundleReport.from_dict(value)


def _optional_hdf5_dataset_report(
    value: Hdf5DatasetReport | Mapping[str, Any] | None,
) -> Hdf5DatasetReport | None:
    if value is None:
        return None
    return value if isinstance(value, Hdf5DatasetReport) else Hdf5DatasetReport.from_dict(value)


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
    mcap_eval_bundle: McapEvalBundleReport | None,
    hdf5_dataset_report: Hdf5DatasetReport | None,
) -> str:
    inputs = ["policy JSONL"]
    if score_report is not None:
        inputs.append("official score")
    if reward_report is not None:
        inputs.append("reward report")
    if failure_report is not None:
        inputs.append("failure report")
    if mcap_eval_bundle is not None:
        inputs.append("MCAP eval bundle")
    if hdf5_dataset_report is not None:
        inputs.append("HDF5 dataset report")
    return "Canonical episode trace fused from " + ", ".join(inputs) + "."


def _episode_trace_generated_at(
    *,
    policy_events: tuple[PolicyTraceEvent, ...],
    score_report: ScoreReport | None,
    reward_report: RewardReport | None,
    failure_report: FailureReport | None,
    mcap_eval_bundle: McapEvalBundleReport | None,
    hdf5_dataset_report: Hdf5DatasetReport | None,
) -> str:
    candidates = [("policy trace last event emitted_at_utc", policy_events[-1].emitted_at_utc)]
    if score_report is not None and score_report.parsed_at_utc is not None:
        candidates.append(("score_report.parsed_at_utc", score_report.parsed_at_utc))
    if reward_report is not None:
        candidates.append(("reward_report.generated_at_utc", reward_report.generated_at_utc))
    if failure_report is not None:
        candidates.append(("failure_report.generated_at_utc", failure_report.generated_at_utc))
    if mcap_eval_bundle is not None:
        candidates.append(("mcap_eval_bundle.analyzed_at_utc", mcap_eval_bundle.analyzed_at_utc))
    if hdf5_dataset_report is not None:
        candidates.append(("hdf5_dataset_report.validated_at_utc", hdf5_dataset_report.validated_at_utc))
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
    mcap_eval_bundle: McapEvalBundleReport | None,
    hdf5_dataset_report: Hdf5DatasetReport | None,
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
        if score_report is not None:
            errors.extend(_score_report_matches_scoring_artifact_errors(scoring_artifact, score_report))
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
    errors.extend(
        _mcap_source_artifact_errors(
            source_artifacts=source_artifacts,
            report=mcap_eval_bundle,
        )
    )
    errors.extend(
        _hdf5_source_artifact_errors(
            source_artifacts=source_artifacts,
            report=hdf5_dataset_report,
        )
    )
    if errors:
        raise HarnessIOError("; ".join(errors))


def _mcap_source_artifact_errors(
    *,
    source_artifacts: tuple[ArtifactRef, ...],
    report: McapEvalBundleReport | None,
) -> list[str]:
    artifacts = tuple(artifact for artifact in source_artifacts if artifact.kind == "mcap_eval_bundle")
    source_report_errors = _forbidden_raw_source_report_errors(
        artifacts,
        kind="mcap_eval_bundle",
        report_kind="mcap_eval_report",
    )
    unattached_report_errors = _unattached_typed_report_artifact_errors(
        source_artifacts,
        kind="mcap_eval_report",
        report_name="mcap_eval_bundle",
    )
    if report is None:
        if len(artifacts) > 1:
            return (
                source_report_errors
                + unattached_report_errors
                + ["episode trace.source_artifacts must include at most one mcap_eval_bundle"]
            )
        if len(artifacts) == 1:
            return (
                source_report_errors
                + unattached_report_errors
                + _quiet_local_mcap_bundle_artifact_errors(artifacts[0])
            )
        return source_report_errors + unattached_report_errors
    if len(artifacts) != 1:
        return source_report_errors + ["episode trace.source_artifacts must include exactly one mcap_eval_bundle"]
    artifact = artifacts[0]
    errors: list[str] = list(source_report_errors)
    if artifact.sha256 is None:
        errors.append("mcap_eval_bundle source artifact sha256 must be set")
    elif artifact.sha256 != _mcap_bundle_sha256_from_report(report):
        errors.append("mcap_eval_bundle source artifact sha256 must match supplied report")
    errors.extend(
        _report_sha256_provenance_errors(
            artifact=artifact,
            expected_report_sha256=_report_sha256(report.to_dict()),
            label="mcap_eval_bundle",
        )
    )
    errors.extend(
        _typed_report_source_artifact_errors(
            source_artifacts=source_artifacts,
            kind="mcap_eval_report",
            report=report.to_dict(),
        )
    )
    if not _artifact_names_source(artifact, report.source):
        errors.append("mcap_eval_bundle source artifact must match mcap_eval_bundle.source")
    errors.extend(_local_mcap_artifact_errors(artifact, report))
    return errors


def _hdf5_source_artifact_errors(
    *,
    source_artifacts: tuple[ArtifactRef, ...],
    report: Hdf5DatasetReport | None,
) -> list[str]:
    artifacts = tuple(artifact for artifact in source_artifacts if artifact.kind == "hdf5_dataset")
    source_report_errors = _forbidden_raw_source_report_errors(
        artifacts,
        kind="hdf5_dataset",
        report_kind="hdf5_dataset_report",
    )
    if report is None:
        unattached_report_errors = _unattached_typed_report_artifact_errors(
            source_artifacts,
            kind="hdf5_dataset_report",
            report_name="hdf5_dataset",
        )
        if len(artifacts) > 1:
            return (
                source_report_errors
                + unattached_report_errors
                + ["episode trace.source_artifacts must include at most one hdf5_dataset"]
            )
        if len(artifacts) == 1:
            return (
                source_report_errors
                + unattached_report_errors
                + _quiet_local_hdf5_artifact_errors(artifacts[0])
            )
        return source_report_errors + unattached_report_errors
    if len(artifacts) != 1:
        return source_report_errors + ["episode trace.source_artifacts must include exactly one hdf5_dataset"]
    artifact = artifacts[0]
    errors: list[str] = list(source_report_errors)
    if artifact.sha256 is None:
        errors.append("hdf5_dataset source artifact sha256 must be set")
    elif artifact.sha256 != report.sha256:
        errors.append("hdf5_dataset source artifact sha256 must match hdf5_dataset_report.sha256")
    errors.extend(
        _report_sha256_provenance_errors(
            artifact=artifact,
            expected_report_sha256=_report_sha256(report.to_dict()),
            label="hdf5_dataset",
        )
    )
    errors.extend(
        _typed_report_source_artifact_errors(
            source_artifacts=source_artifacts,
            kind="hdf5_dataset_report",
            report=report.to_dict(),
        )
    )
    if not _artifact_names_source(artifact, report.source):
        errors.append("hdf5_dataset source artifact must match hdf5_dataset_report.source")
    errors.extend(_local_hdf5_artifact_errors(artifact))
    errors.extend(
        _hdf5_report_matches_local_bytes_errors(
            source_artifact=artifact,
            report=report,
            label="hdf5_dataset",
        )
    )
    return errors


def _forbidden_raw_source_report_errors(
    artifacts: tuple[ArtifactRef, ...],
    *,
    kind: str,
    report_kind: str,
) -> list[str]:
    return [
        f"{kind} source artifact provenance.source_report must not be set; "
        f"use the {report_kind} source artifact instead"
        for artifact in artifacts
        if artifact.provenance.get("source_report") is not None
    ]


def _unattached_typed_report_artifact_errors(
    source_artifacts: tuple[ArtifactRef, ...],
    *,
    kind: str,
    report_name: str,
) -> list[str]:
    artifacts = tuple(artifact for artifact in source_artifacts if artifact.kind == kind)
    if not artifacts:
        return []
    return [
        f"episode trace.source_artifacts must not include {kind} without "
        f"{report_name} evidence"
    ]


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


def _score_report_matches_scoring_artifact_errors(
    artifact: ArtifactRef,
    score_report: ScoreReport,
) -> list[str]:
    expected_report_sha256 = _report_sha256(score_report.to_dict())
    artifact_report_sha256 = artifact.provenance.get("report_sha256")
    if artifact_report_sha256 is not None and artifact_report_sha256 != expected_report_sha256:
        return ["scoring_yaml source artifact provenance.report_sha256 must match score_report"]
    try:
        scoring_path = local_artifact_path(
            path=artifact.path,
            uri=artifact.uri,
            field_name="scoring_yaml source artifact",
        )
    except HarnessIOError:
        scoring_path = None
    if scoring_path is None or not scoring_path.exists() or not scoring_path.is_file():
        return ["scoring_yaml source artifact must be a local scoring.yaml snapshot when score_report is supplied"]
    try:
        parsed_score = parse_scoring_yaml(scoring_path)
    except HarnessIOError as exc:
        return [f"scoring_yaml source artifact must parse as a ScoreReport: {exc}"]
    source_aliases = tuple(
        source
        for source in (
            parsed_score.source,
            score_report.source,
            artifact.path,
            artifact.uri,
            str(scoring_path.resolve(strict=False)),
        )
        if source is not None
    )
    if not parsed_score.equivalent_to(score_report, source_aliases=source_aliases):
        return ["score_report must match scoring_yaml source artifact bytes"]
    return []


def _local_hdf5_artifact_errors(artifact: ArtifactRef) -> list[str]:
    try:
        dataset_path = local_artifact_path(
            path=artifact.path,
            uri=artifact.uri,
            field_name="hdf5_dataset source artifact",
        )
    except HarnessIOError as exc:
        return [str(exc)]
    if dataset_path is None:
        return ["hdf5_dataset source artifact must be a local byte-verifiable file"]
    if not dataset_path.exists():
        return ["hdf5_dataset source artifact path must exist"]
    if not dataset_path.is_file():
        return ["hdf5_dataset source artifact path must be a file"]
    if artifact.sha256 is not None and sha256_file(dataset_path) != artifact.sha256:
        return ["hdf5_dataset source artifact sha256 must match path"]
    return []


def _quiet_local_hdf5_artifact_errors(artifact: ArtifactRef) -> list[str]:
    try:
        dataset_path = local_artifact_path(
            path=artifact.path,
            uri=artifact.uri,
            field_name="hdf5_dataset source artifact",
        )
    except HarnessIOError as exc:
        return [str(exc)]
    if dataset_path is None:
        return []
    if not dataset_path.exists():
        return ["hdf5_dataset source artifact path must exist"]
    if not dataset_path.is_file():
        return ["hdf5_dataset source artifact path must be a file"]
    if artifact.sha256 is not None and sha256_file(dataset_path) != artifact.sha256:
        return ["hdf5_dataset source artifact sha256 must match path"]
    return []


def _local_mcap_artifact_errors(
    artifact: ArtifactRef,
    report: McapEvalBundleReport,
) -> list[str]:
    errors = _local_mcap_bundle_artifact_errors(artifact)
    if errors:
        return errors
    for trial in report.trials:
        if _is_uri_source(trial.source):
            continue
        trial_path = Path(trial.source)
        if not trial_path.exists():
            errors.append(f"mcap_eval_bundle trial source path must exist: {trial.trial_id}")
        elif not trial_path.is_file():
            errors.append(f"mcap_eval_bundle trial source path must be a file: {trial.trial_id}")
        elif sha256_file(trial_path) != trial.sha256:
            errors.append(f"mcap_eval_bundle trial source sha256 must match path: {trial.trial_id}")
    return errors


def _quiet_local_mcap_bundle_artifact_errors(artifact: ArtifactRef) -> list[str]:
    try:
        bundle_path = local_artifact_path(
            path=artifact.path,
            uri=artifact.uri,
            field_name="mcap_eval_bundle source artifact",
        )
    except HarnessIOError as exc:
        return [str(exc)]
    if bundle_path is None:
        return []
    errors: list[str] = []
    if not bundle_path.exists():
        errors.append("mcap_eval_bundle source artifact path must exist")
    elif not bundle_path.is_dir():
        errors.append("mcap_eval_bundle source artifact path must be a directory")
    return errors


def _local_mcap_bundle_artifact_errors(artifact: ArtifactRef) -> list[str]:
    try:
        bundle_path = local_artifact_path(
            path=artifact.path,
            uri=artifact.uri,
            field_name="mcap_eval_bundle source artifact",
        )
    except HarnessIOError as exc:
        return [str(exc)]
    if bundle_path is None:
        return ["mcap_eval_bundle source artifact must be a local byte-verifiable directory"]
    errors: list[str] = []
    if not bundle_path.exists():
        errors.append("mcap_eval_bundle source artifact path must exist")
    elif not bundle_path.is_dir():
        errors.append("mcap_eval_bundle source artifact path must be a directory")
    return errors


def _mcap_bundle_sha256_from_report(report: McapEvalBundleReport) -> str:
    digest = hashlib.sha256()
    bundle_root = Path(report.source) if not _is_uri_source(report.source) else None
    for trial in report.trials:
        trial_identity = (
            Path(trial.source).relative_to(bundle_root).as_posix()
            if bundle_root is not None
            else trial.source
        )
        digest.update(trial.trial_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(trial_identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(trial.size_bytes).encode("utf-8"))
        digest.update(b"\0")
        digest.update(trial.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _report_sha256(mapping: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(_plain_json_value(mapping), allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item) for item in value]
    return value


def _typed_report_source_artifact_errors(
    *,
    source_artifacts: tuple[ArtifactRef, ...],
    kind: str,
    report: Mapping[str, Any],
) -> list[str]:
    artifacts = tuple(artifact for artifact in source_artifacts if artifact.kind == kind)
    if len(artifacts) != 1:
        return [f"episode trace.source_artifacts must include exactly one {kind}"]
    artifact = artifacts[0]
    errors: list[str] = []
    if artifact.sha256 is None:
        errors.append(f"{kind} source artifact sha256 must be set")
    try:
        report_path = local_artifact_path(
            path=artifact.path,
            uri=artifact.uri,
            field_name=f"{kind} source artifact",
        )
    except HarnessIOError as exc:
        return errors + [str(exc)]
    if report_path is None:
        errors.append(f"{kind} source artifact must be a local JSON file")
    elif not report_path.exists():
        errors.append(f"{kind} source artifact path must exist")
    elif not report_path.is_file():
        errors.append(f"{kind} source artifact path must be a file")
    elif artifact.sha256 != sha256_file(report_path):
        errors.append(f"{kind} source artifact sha256 must match path")
    else:
        persisted = read_json(report_path)
        if persisted != report:
            errors.append(f"{kind} source artifact must match supplied report")
    return errors


def _report_sha256_provenance_errors(
    *,
    artifact: ArtifactRef,
    expected_report_sha256: str,
    label: str,
) -> list[str]:
    value = artifact.provenance.get("report_sha256")
    if not _looks_like_sha256(value):
        return [f"{label} source artifact provenance.report_sha256 must be set"]
    if value != expected_report_sha256:
        return [f"{label} source artifact provenance.report_sha256 must match supplied report"]
    return []


def _is_uri_source(source: str) -> bool:
    return "://" in source


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
