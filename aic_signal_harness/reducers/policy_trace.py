"""Reducer for backend-emitted AIC policy trace JSONL."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Mapping

from aic_signal_harness.artifacts import HarnessIOError, sha256_file
from aic_signal_harness.policy_trace import (
    PolicyTraceEvent,
    PolicyTraceEventType,
    PolicyTraceReport,
    PolicyTraceTrialReport,
    action_payload_is_nonzero,
)
from aic_signal_harness.schemas import ArtifactRef


@dataclass(frozen=True)
class PolicyTraceReduction:
    """Policy trace report bound to the source JSONL artifact identity."""

    artifact: ArtifactRef
    report: PolicyTraceReport
    events: tuple[PolicyTraceEvent, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.artifact, ArtifactRef):
            errors.append("policy trace reduction artifact must be an ArtifactRef")
        if not isinstance(self.report, PolicyTraceReport):
            errors.append("policy trace reduction report must be a PolicyTraceReport")
        if not isinstance(self.events, tuple):
            errors.append("policy trace reduction events must be a tuple")
        if errors:
            raise HarnessIOError("; ".join(errors))
        if self.artifact.kind != "policy_trace_jsonl":
            errors.append("policy trace reduction artifact.kind must be 'policy_trace_jsonl'")
        if self.artifact.sha256 is None:
            errors.append("policy trace reduction artifact.sha256 must be set")
        if self.artifact.path != self.report.source:
            errors.append("policy trace reduction artifact.path must match report.source")
        if self.artifact.sha256 is not None and self.artifact.sha256 != sha256_file(self.report.source):
            errors.append("policy trace reduction artifact.sha256 must match source file")
        if self.events:
            if self.events[0].run_id != self.report.run_id:
                errors.append("policy trace reduction events must match report.run_id")
            if len(self.events) != self.report.event_count:
                errors.append("policy trace reduction events must match report.event_count")
        if errors:
            raise HarnessIOError("; ".join(errors))


def analyze_policy_trace_jsonl(
    path: str | Path,
    *,
    reduced_at_utc: str | None = None,
) -> PolicyTraceReport:
    """Analyze a policy trace JSONL file into a deterministic typed report."""

    trace_path = _resolve_existing_file(path)
    events = read_policy_trace_events(trace_path)
    reduced_at = events[-1].emitted_at_utc if reduced_at_utc is None else reduced_at_utc
    return _report_from_events(
        source=str(trace_path),
        events=events,
        reduced_at_utc=reduced_at,
    )


def reduce_policy_trace_jsonl(
    path: str | Path,
    *,
    uri: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    reduced_at_utc: str | None = None,
) -> PolicyTraceReduction:
    """Reduce policy trace JSONL and bind it to a digest-addressed artifact."""

    trace_path = _resolve_existing_file(path)
    before = _stat_snapshot(trace_path)
    events = read_policy_trace_events(trace_path)
    reduced_at = events[-1].emitted_at_utc if reduced_at_utc is None else reduced_at_utc
    report = _report_from_events(
        source=str(trace_path),
        events=events,
        reduced_at_utc=reduced_at,
    )
    digest = sha256_file(trace_path)
    after = _stat_snapshot(trace_path)
    _assert_same_file_identity(
        trace_path,
        before,
        after,
        "policy trace JSONL changed during reduction",
    )
    return PolicyTraceReduction(
        artifact=ArtifactRef(
            kind="policy_trace_jsonl",
            path=report.source,
            uri=uri,
            sha256=digest,
            provenance={} if provenance is None else provenance,
        ),
        report=report,
        events=events,
    )


def read_policy_trace_events(path: str | Path) -> tuple[PolicyTraceEvent, ...]:
    """Read and validate policy trace JSONL events without reducing them."""

    trace_path = _resolve_existing_file(path)
    events = _read_trace_events(trace_path)
    _validate_event_sequence(events, trace_path)
    return events


def _read_trace_events(path: Path) -> tuple[PolicyTraceEvent, ...]:
    events: list[PolicyTraceEvent] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.rstrip("\n")
                if not line.strip():
                    raise HarnessIOError(
                        f"policy trace JSONL line {line_number} in {path} must not be blank"
                    )
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise HarnessIOError(
                        f"invalid policy trace JSONL line {line_number} in {path}: {exc.msg}"
                    ) from exc
                if not isinstance(payload, Mapping):
                    raise HarnessIOError(
                        f"policy trace JSONL line {line_number} in {path} must be a JSON object"
                    )
                try:
                    events.append(PolicyTraceEvent.from_dict(payload))
                except HarnessIOError as exc:
                    raise HarnessIOError(
                        f"invalid policy trace JSONL line {line_number} in {path}: {exc}"
                    ) from exc
    except FileNotFoundError as exc:
        raise HarnessIOError(f"policy trace JSONL file not found: {path}") from exc
    except OSError as exc:
        raise HarnessIOError(f"failed to read policy trace JSONL {path}: {exc}") from exc
    if not events:
        raise HarnessIOError(f"policy trace JSONL must contain at least one event: {path}")
    return tuple(events)


def _validate_event_sequence(events: tuple[PolicyTraceEvent, ...], path: Path) -> None:
    errors: list[str] = []
    expected_indices = tuple(range(len(events)))
    actual_indices = tuple(event.event_index for event in events)
    if actual_indices != expected_indices:
        errors.append("policy trace event_index values must be contiguous and match JSONL order")
    run_ids = {event.run_id for event in events}
    if len(run_ids) != 1:
        errors.append("policy trace events must have exactly one run_id")
    elapsed_by_trial: dict[str, float] = {}
    for event in events:
        previous_elapsed = elapsed_by_trial.get(event.trial_id)
        if previous_elapsed is not None and event.elapsed_sec < previous_elapsed:
            errors.append(
                f"policy trace elapsed_sec must be nondecreasing within {event.trial_id}"
            )
            break
        elapsed_by_trial[event.trial_id] = event.elapsed_sec
    if errors:
        raise HarnessIOError(f"invalid policy trace JSONL {path}: " + "; ".join(errors))


def _report_from_events(
    *,
    source: str,
    events: tuple[PolicyTraceEvent, ...],
    reduced_at_utc: str,
) -> PolicyTraceReport:
    trials = tuple(
        _trial_report(trial_id=trial_id, events=trial_events)
        for trial_id, trial_events in _events_by_trial(events).items()
    )
    return PolicyTraceReport(
        source=source,
        run_id=events[0].run_id,
        reduced_at_utc=reduced_at_utc,
        event_count=len(events),
        start_elapsed_sec=min(event.elapsed_sec for event in events),
        end_elapsed_sec=max(event.elapsed_sec for event in events),
        event_type_counts=_event_type_counts(events),
        leakage_classes=tuple(
            sorted({event.leakage_class for event in events}, key=lambda item: item.value)
        ),
        trials=trials,
    )


def _trial_report(
    *,
    trial_id: str,
    events: tuple[PolicyTraceEvent, ...],
) -> PolicyTraceTrialReport:
    action_events = tuple(
        event for event in events if event.event_type.value in {"action_selected", "action_published"}
    )
    return PolicyTraceTrialReport(
        trial_id=trial_id,
        event_count=len(events),
        first_event_index=events[0].event_index,
        last_event_index=events[-1].event_index,
        start_elapsed_sec=events[0].elapsed_sec,
        end_elapsed_sec=events[-1].elapsed_sec,
        action_event_count=len(action_events),
        nonzero_action_event_count=sum(
            int(action_payload_is_nonzero(event.payload)) for event in action_events
        ),
        safety_guard_event_count=sum(
            int(event.event_type is PolicyTraceEventType.safety_guard) for event in events
        ),
        error_event_count=sum(
            int(event.event_type is PolicyTraceEventType.error) for event in events
        ),
        event_type_counts=_event_type_counts(events),
    )


def _events_by_trial(
    events: tuple[PolicyTraceEvent, ...],
) -> dict[str, tuple[PolicyTraceEvent, ...]]:
    by_trial: dict[str, list[PolicyTraceEvent]] = {}
    for event in events:
        by_trial.setdefault(event.trial_id, []).append(event)
    return {trial_id: tuple(items) for trial_id, items in by_trial.items()}


def _event_type_counts(events: tuple[PolicyTraceEvent, ...]) -> dict[str, int]:
    return dict(Counter(event.event_type.value for event in events))


def _resolve_existing_file(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise HarnessIOError(f"policy trace JSONL file not found: {resolved}")
    if not resolved.is_file():
        raise HarnessIOError(f"policy trace JSONL path is not a file: {resolved}")
    return resolved


def _stat_snapshot(path: Path) -> os.stat_result:
    try:
        return path.stat()
    except FileNotFoundError as exc:
        raise HarnessIOError(f"policy trace JSONL file not found: {path}") from exc
    except OSError as exc:
        raise HarnessIOError(f"failed to stat policy trace JSONL {path}: {exc}") from exc


def _assert_same_file_identity(
    path: Path,
    before: os.stat_result,
    after: os.stat_result,
    message: str,
) -> None:
    if (
        before.st_ino != after.st_ino
        or before.st_dev != after.st_dev
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise HarnessIOError(f"{message}: {path}")
