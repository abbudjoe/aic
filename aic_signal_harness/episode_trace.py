"""Canonical episode trace contracts for AIC experiment signal."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, TypeVar, cast
from urllib.parse import unquote, urlparse

from aic_signal_harness.artifacts import HarnessIOError, local_artifact_path, read_json, sha256_file
from aic_signal_harness.hdf5_dataset import Hdf5DatasetReport, REQUIRED_HDF5_DATASET_KEYS
from aic_signal_harness.reducers.hdf5_dataset import validate_hdf5_dataset
from aic_signal_harness.mcap_eval import McapEvalBundleReport
from aic_signal_harness.schemas import (
    ArtifactRef,
    LeakageClass,
    SCHEMA_VERSION,
    SchemaValidationError,
)
from aic_signal_harness.scoring import ScoreReport, TrialScore


_OFFICIAL_TRIAL_ID_RE = re.compile(r"^trial_[1-9][0-9]*$")
_TIMELINE_EVENT_KEYS = frozenset(
    {
        "event_index",
        "event_kind",
        "elapsed_sec",
        "source",
        "leakage_class",
        "payload",
        "trial_id",
        "emitted_at_utc",
    }
)
_TRIAL_TRACE_KEYS = frozenset(
    {
        "trial_id",
        "start_elapsed_sec",
        "end_elapsed_sec",
        "event_count",
        "observation_event_count",
        "action_event_count",
        "nonzero_action_event_count",
        "safety_guard_event_count",
        "error_event_count",
        "score",
        "events",
    }
)
_EPISODE_TRACE_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "generated_at_utc",
        "source_artifacts",
        "trials",
        "run_events",
        "notes",
    }
)


class _StrEnum(str, Enum):
    @classmethod
    def parse(cls: type["_EnumT"], value: Any) -> "_EnumT":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise HarnessIOError(f"{cls.__name__} must be a string")
        try:
            return cls(value)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise HarnessIOError(
                f"unknown {cls.__name__} {value!r}; expected one of: {allowed}"
            ) from exc


_EnumT = TypeVar("_EnumT", bound=_StrEnum)


class TimelineEventKind(_StrEnum):
    """Normalized event families inside a canonical episode trace."""

    policy_event = "policy_event"
    observation_event = "observation_event"
    action_event = "action_event"
    safety_guard = "safety_guard"
    error = "error"
    controller_summary = "controller_summary"
    contact_evidence = "contact_evidence"
    dataset_episode_summary = "dataset_episode_summary"
    official_score = "official_score"
    reward_term = "reward_term"
    failure_label = "failure_label"


@dataclass(frozen=True)
class TimelineEvent:
    """One normalized event in a run or trial timeline."""

    event_index: int
    event_kind: TimelineEventKind
    elapsed_sec: float
    source: str
    leakage_class: LeakageClass
    payload: Mapping[str, Any] = field(default_factory=dict)
    trial_id: str | None = None
    emitted_at_utc: str | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(
                self,
                "event_index",
                _require_nonnegative_int(self.event_index, "timeline event.event_index"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "event_kind",
                TimelineEventKind.parse(self.event_kind),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "elapsed_sec",
                _require_nonnegative_float(self.elapsed_sec, "timeline event.elapsed_sec"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "source", _require_nonempty_text(self.source, "timeline event.source"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "leakage_class", LeakageClass.parse(self.leakage_class))
        except SchemaValidationError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "trial_id",
                _optional_nonempty_text(self.trial_id, "timeline event.trial_id"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "emitted_at_utc",
                _optional_nonempty_text(self.emitted_at_utc, "timeline event.emitted_at_utc"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "payload",
                _copy_json_mapping(self.payload, "timeline event.payload"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not errors:
            errors.extend(_timeline_event_kind_errors(self))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "event_index": self.event_index,
            "event_kind": self.event_kind.value,
            "elapsed_sec": self.elapsed_sec,
            "source": self.source,
            "leakage_class": self.leakage_class.value,
            "payload": _thaw_json(self.payload),
        }
        if self.trial_id is not None:
            value["trial_id"] = self.trial_id
        if self.emitted_at_utc is not None:
            value["emitted_at_utc"] = self.emitted_at_utc
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TimelineEvent":
        if not isinstance(value, Mapping):
            raise HarnessIOError("timeline event must be a mapping")
        _reject_unknown_keys(value, _TIMELINE_EVENT_KEYS, "timeline event")
        return cls(
            event_index=cast(Any, value.get("event_index")),
            event_kind=cast(Any, value.get("event_kind")),
            elapsed_sec=cast(Any, value.get("elapsed_sec")),
            source=cast(Any, value.get("source")),
            leakage_class=cast(Any, value.get("leakage_class")),
            payload=value.get("payload", {}),
            trial_id=value.get("trial_id"),
            emitted_at_utc=value.get("emitted_at_utc"),
        )


@dataclass(frozen=True)
class TrialTrace:
    """Canonical trace for one official or policy-observed trial."""

    trial_id: str
    start_elapsed_sec: float
    end_elapsed_sec: float
    event_count: int
    observation_event_count: int = 0
    action_event_count: int = 0
    nonzero_action_event_count: int = 0
    safety_guard_event_count: int = 0
    error_event_count: int = 0
    score: TrialScore | None = None
    events: tuple[TimelineEvent, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(self, "trial_id", _require_nonempty_text(self.trial_id, "trial trace.trial_id"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("start_elapsed_sec", "end_elapsed_sec"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonnegative_float(
                        getattr(self, field_name),
                        f"trial trace.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        for field_name in (
            "event_count",
            "observation_event_count",
            "action_event_count",
            "nonzero_action_event_count",
            "safety_guard_event_count",
            "error_event_count",
        ):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonnegative_int(
                        getattr(self, field_name),
                        f"trial trace.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "events",
                tuple(
                    event if isinstance(event, TimelineEvent) else TimelineEvent.from_dict(event)
                    for event in _as_sequence(self.events, "trial trace.events")
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if self.score is not None and not isinstance(self.score, TrialScore):
            try:
                object.__setattr__(self, "score", TrialScore.from_dict(self.score))
            except HarnessIOError as exc:
                errors.append(str(exc))
        if not errors:
            events = cast(tuple[TimelineEvent, ...], self.events)
            mismatched = tuple(event.trial_id for event in events if event.trial_id != self.trial_id)
            if mismatched:
                errors.append("trial trace events must all match trial_id")
            if self.event_count != len(events):
                errors.append("trial trace.event_count must match events")
            if self.observation_event_count != sum(
                event.event_kind is TimelineEventKind.observation_event for event in events
            ):
                errors.append("trial trace.observation_event_count must match events")
            if self.action_event_count != sum(
                event.event_kind is TimelineEventKind.action_event for event in events
            ):
                errors.append("trial trace.action_event_count must match events")
            if self.nonzero_action_event_count != sum(
                event.event_kind is TimelineEventKind.action_event
                and _timeline_action_payload_is_nonzero(event.payload)
                for event in events
            ):
                errors.append("trial trace.nonzero_action_event_count must match events")
            if self.safety_guard_event_count != sum(
                event.event_kind is TimelineEventKind.safety_guard for event in events
            ):
                errors.append("trial trace.safety_guard_event_count must match events")
            if self.error_event_count != sum(
                event.event_kind is TimelineEventKind.error for event in events
            ):
                errors.append("trial trace.error_event_count must match events")
            if self.start_elapsed_sec > self.end_elapsed_sec:
                errors.append("trial trace start_elapsed_sec must be <= end_elapsed_sec")
            if events:
                elapsed_values = tuple(event.elapsed_sec for event in events)
                if min(elapsed_values) < self.start_elapsed_sec or max(elapsed_values) > self.end_elapsed_sec:
                    errors.append("trial trace event elapsed_sec values must be within trial bounds")
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "start_elapsed_sec": self.start_elapsed_sec,
            "end_elapsed_sec": self.end_elapsed_sec,
            "event_count": self.event_count,
            "observation_event_count": self.observation_event_count,
            "action_event_count": self.action_event_count,
            "nonzero_action_event_count": self.nonzero_action_event_count,
            "safety_guard_event_count": self.safety_guard_event_count,
            "error_event_count": self.error_event_count,
            "score": None if self.score is None else self.score.to_dict(),
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrialTrace":
        if not isinstance(value, Mapping):
            raise HarnessIOError("trial trace must be a mapping")
        _reject_unknown_keys(value, _TRIAL_TRACE_KEYS, "trial trace")
        return cls(
            trial_id=cast(Any, value.get("trial_id")),
            start_elapsed_sec=cast(Any, value.get("start_elapsed_sec")),
            end_elapsed_sec=cast(Any, value.get("end_elapsed_sec")),
            event_count=cast(Any, value.get("event_count")),
            observation_event_count=cast(Any, value.get("observation_event_count", 0)),
            action_event_count=cast(Any, value.get("action_event_count", 0)),
            nonzero_action_event_count=cast(Any, value.get("nonzero_action_event_count", 0)),
            safety_guard_event_count=cast(Any, value.get("safety_guard_event_count", 0)),
            error_event_count=cast(Any, value.get("error_event_count", 0)),
            score=value.get("score"),
            events=cast(Any, value.get("events", ())),
        )


@dataclass(frozen=True)
class EpisodeTrace:
    """One run's canonical trace over policy events, scores, labels, and rewards."""

    run_id: str
    generated_at_utc: str
    source_artifacts: tuple[ArtifactRef, ...] = field(default_factory=tuple)
    trials: tuple[TrialTrace, ...] = field(default_factory=tuple)
    run_events: tuple[TimelineEvent, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        try:
            object.__setattr__(self, "run_id", _require_nonempty_text(self.run_id, "episode trace.run_id"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "generated_at_utc",
                _require_nonempty_text(self.generated_at_utc, "episode trace.generated_at_utc"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "source_artifacts",
                tuple(
                    artifact if isinstance(artifact, ArtifactRef) else ArtifactRef.from_dict(artifact)
                    for artifact in _as_sequence(self.source_artifacts, "episode trace.source_artifacts")
                ),
            )
        except (HarnessIOError, SchemaValidationError) as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "trials",
                tuple(
                    trial if isinstance(trial, TrialTrace) else TrialTrace.from_dict(trial)
                    for trial in _as_sequence(self.trials, "episode trace.trials")
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "run_events",
                tuple(
                    event if isinstance(event, TimelineEvent) else TimelineEvent.from_dict(event)
                    for event in _as_sequence(self.run_events, "episode trace.run_events")
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "notes",
                tuple(
                    _require_nonempty_text(note, "episode trace.note")
                    for note in _as_sequence(self.notes, "episode trace.notes")
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not errors:
            trial_ids = tuple(trial.trial_id for trial in self.trials)
            if len(set(trial_ids)) != len(trial_ids):
                errors.append("episode trace trials must have unique trial_id values")
            if not self.trials:
                errors.append("episode trace must contain at least one trial")
            errors.extend(_source_artifact_errors(self.source_artifacts, self.run_id))
            errors.extend(
                _post_hoc_evidence_source_artifact_errors(
                    source_artifacts=self.source_artifacts,
                    trials=self.trials,
                    run_events=self.run_events,
                )
            )
            errors.extend(
                _episode_evidence_consistency_errors(
                    trials=self.trials,
                    run_events=self.run_events,
                )
            )
            known_trial_ids = set(trial_ids)
            unknown_run_trial_ids = sorted(
                {
                    event.trial_id
                    for event in self.run_events
                    if event.trial_id is not None and event.trial_id not in known_trial_ids
                }
            )
            if unknown_run_trial_ids:
                errors.append(
                    "episode trace run_events trial_id must reference a trace trial: "
                    + ", ".join(unknown_run_trial_ids)
                )
            all_event_indices = tuple(
                event.event_index
                for trial in self.trials
                for event in trial.events
            ) + tuple(event.event_index for event in self.run_events)
            if all_event_indices != tuple(range(len(all_event_indices))):
                errors.append("episode trace event_index values must be unique, contiguous, and stored in order")
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generated_at_utc": self.generated_at_utc,
            "source_artifacts": [artifact.to_dict() for artifact in self.source_artifacts],
            "trials": [trial.to_dict() for trial in self.trials],
            "run_events": [event.to_dict() for event in self.run_events],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeTrace":
        if not isinstance(value, Mapping):
            raise HarnessIOError("episode trace must be a mapping")
        _reject_unknown_keys(value, _EPISODE_TRACE_KEYS, "episode trace")
        if _raw_episode_trace_contains_mcap_evidence(value):
            raise HarnessIOError(
                "episode trace fused MCAP evidence requires the dedicated byte rederive slice/analyzer injection"
            )
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            run_id=cast(Any, value.get("run_id")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            source_artifacts=cast(Any, value.get("source_artifacts", ())),
            trials=cast(Any, value.get("trials", ())),
            run_events=cast(Any, value.get("run_events", ())),
            notes=cast(Any, value.get("notes", ())),
        )


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed_keys: frozenset[str],
    field_name: str,
) -> None:
    unknown_keys = sorted(repr(key) for key in value if key not in allowed_keys)
    if unknown_keys:
        raise HarnessIOError(f"{field_name} has unknown fields: {', '.join(unknown_keys)}")


def _raw_episode_trace_contains_mcap_evidence(value: Mapping[str, Any]) -> bool:
    for field_name in ("trials", "run_events"):
        raw_items = value.get(field_name, ())
        if isinstance(raw_items, (str, bytes, bytearray)) or not isinstance(raw_items, (list, tuple)):
            continue
        for item in raw_items:
            if not isinstance(item, Mapping):
                continue
            if field_name == "run_events":
                raw_event_kind = item.get("event_kind")
                if raw_event_kind in {TimelineEventKind.controller_summary.value, TimelineEventKind.contact_evidence.value}:
                    return True
            else:
                raw_events = item.get("events", ())
                if isinstance(raw_events, (str, bytes, bytearray)) or not isinstance(raw_events, (list, tuple)):
                    continue
                for event in raw_events:
                    if not isinstance(event, Mapping):
                        continue
                    raw_event_kind = event.get("event_kind")
                    if raw_event_kind in {
                        TimelineEventKind.controller_summary.value,
                        TimelineEventKind.contact_evidence.value,
                    }:
                        return True
    return False


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessIOError(f"{field_name} must be a nonempty string")
    return value


def _optional_nonempty_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_text(value, field_name)


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise HarnessIOError(f"{field_name} must be a nonnegative integer")
    return value


def _require_nonnegative_float(value: Any, field_name: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(float(value)) or float(value) < 0.0:
        raise HarnessIOError(f"{field_name} must be a nonnegative finite number")
    return float(value)


def _as_sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if value is None:
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    return tuple(value)


def _copy_json_mapping(value: Any, field_name: str) -> MappingProxyType[str, Any]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise HarnessIOError(f"{field_name} keys must be nonempty strings")
        if key in copied:
            raise HarnessIOError(f"{field_name} has duplicate key after normalization: {key}")
        copied[key] = _copy_json_value(item, f"{field_name}.{key}")
    return MappingProxyType(copied)


def _copy_json_value(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        return _copy_json_mapping(value, field_name)
    if isinstance(value, (list, tuple)):
        return tuple(_copy_json_value(item, f"{field_name}[]") for item in value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise HarnessIOError(f"{field_name} must be JSON-compatible")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _timeline_event_kind_errors(event: TimelineEvent) -> list[str]:
    if event.event_kind is TimelineEventKind.observation_event:
        return _observation_event_errors(event)
    if event.event_kind is TimelineEventKind.dataset_episode_summary:
        return _dataset_episode_summary_errors(event)
    if event.event_kind is TimelineEventKind.controller_summary:
        return _controller_summary_errors(event)
    if event.event_kind is TimelineEventKind.contact_evidence:
        return _contact_evidence_errors(event)
    if event.event_kind is TimelineEventKind.official_score:
        return _official_score_errors(event)
    return []


def _observation_event_errors(event: TimelineEvent) -> list[str]:
    if event.leakage_class is not LeakageClass.legal_policy_input:
        return ["observation_event events must use legal_policy_input leakage_class"]
    return []


def _dataset_episode_summary_errors(event: TimelineEvent) -> list[str]:
    errors: list[str] = []
    if event.leakage_class is not LeakageClass.privileged_training_signal:
        errors.append(
            "dataset_episode_summary events must use privileged_training_signal leakage_class"
        )
    if event.trial_id is not None:
        errors.append("dataset_episode_summary events must not set trial_id")
    errors.extend(
        _require_payload_text_value(
            event.payload,
            "event_scope",
            "offline_dataset_summary",
            "dataset_episode_summary.event_scope",
        )
    )
    errors.extend(_require_payload_sha256(event.payload, "dataset_episode_summary"))
    evidence_window, window_errors = _payload_evidence_window(
        event.payload,
        "dataset_episode_summary.evidence_window",
    )
    errors.extend(window_errors)
    if evidence_window is not None and not math.isclose(
        event.elapsed_sec,
        evidence_window["start_elapsed_sec"],
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        errors.append("dataset_episode_summary elapsed_sec must match evidence_window.start_elapsed_sec")
    try:
        _require_nonempty_text(
            _payload_required(event.payload, "validated_at_utc"),
            "dataset_episode_summary.validated_at_utc",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
    ok = event.payload.get("ok")
    if type(ok) is not bool:
        errors.append("dataset_episode_summary.ok must be a boolean")
    episode_count = _optional_payload_nonnegative_int(
        event.payload,
        "episode_count",
        "dataset_episode_summary.episode_count",
        errors,
    )
    step_count = _optional_payload_nonnegative_int(
        event.payload,
        "step_count",
        "dataset_episode_summary.step_count",
        errors,
    )
    episode_lengths = _payload_nonnegative_int_tuple(
        event.payload,
        "episode_lengths",
        "dataset_episode_summary.episode_lengths",
        errors,
    )
    episode_offsets = _payload_nonnegative_int_tuple(
        event.payload,
        "episode_offsets",
        "dataset_episode_summary.episode_offsets",
        errors,
    )
    if (episode_count is None) != (step_count is None):
        errors.append("dataset_episode_summary episode_count and step_count must both be set or both be null")
    if episode_count is None:
        if episode_lengths:
            errors.append("dataset_episode_summary.episode_lengths must be empty when episode_count is null")
        if episode_offsets:
            errors.append("dataset_episode_summary.episode_offsets must be empty when episode_count is null")
    else:
        if len(episode_lengths) != episode_count:
            errors.append("dataset_episode_summary.episode_lengths length must match episode_count")
        if len(episode_offsets) != episode_count:
            errors.append("dataset_episode_summary.episode_offsets length must match episode_count")
        if step_count is not None and sum(episode_lengths) != step_count:
            errors.append("dataset_episode_summary.step_count must match sum of episode_lengths")
        if episode_offsets != _expected_episode_offsets(episode_lengths):
            errors.append("dataset_episode_summary.episode_offsets must match cumulative episode_lengths")
    for key in ("required_datasets", "missing_datasets"):
        errors.extend(
            _payload_nonempty_text_tuple_errors(
                event.payload,
                key,
                f"dataset_episode_summary.{key}",
                allow_empty=key == "missing_datasets",
            )
        )
    required_datasets = _text_tuple(event.payload.get("required_datasets"))
    if required_datasets and required_datasets != REQUIRED_HDF5_DATASET_KEYS:
        errors.append(
            "dataset_episode_summary.required_datasets must match the canonical AIC HDF5 dataset contract"
        )
    for key in ("observation_datasets", "action_datasets", "task_datasets"):
        errors.extend(
            _dataset_stats_mapping_errors(
                event.payload.get(key),
                f"dataset_episode_summary.{key}",
            )
        )
    missing_datasets = _text_tuple(event.payload.get("missing_datasets"))
    observation_datasets = event.payload.get("observation_datasets")
    action_datasets = event.payload.get("action_datasets")
    if ok is True and missing_datasets:
        errors.append("dataset_episode_summary.missing_datasets must be empty when ok is true")
    if ok is True and isinstance(observation_datasets, Mapping):
        missing_required_observations = tuple(
            dataset_name
            for dataset_name in ("pixels", "left_pixels", "right_pixels", "proprio", "state")
            if dataset_name not in observation_datasets
        )
        if missing_required_observations:
            errors.append(
                "dataset_episode_summary.observation_datasets must include "
                + ", ".join(missing_required_observations)
                + " when ok is true"
            )
    if ok is True and isinstance(action_datasets, Mapping) and "action" not in action_datasets:
        errors.append("dataset_episode_summary.action_datasets must include action when ok is true")
    task_datasets = event.payload.get("task_datasets")
    if ok is True and isinstance(task_datasets, Mapping):
        missing_required_tasks = tuple(
            dataset_name
            for dataset_name in ("task_id", "plug_type", "port_type", "target_module_name")
            if dataset_name not in task_datasets
        )
        if missing_required_tasks:
            errors.append(
                "dataset_episode_summary.task_datasets must include "
                + ", ".join(missing_required_tasks)
                + " when ok is true"
            )
    if ok is True and step_count is not None:
        errors.extend(
            _ok_dataset_shape_errors(
                step_count=step_count,
                observation_datasets=observation_datasets,
                action_datasets=action_datasets,
                task_datasets=task_datasets,
            )
        )
    if missing_datasets and isinstance(observation_datasets, Mapping) and isinstance(action_datasets, Mapping):
        present_dataset_names = set(observation_datasets) | set(action_datasets)
        if isinstance(task_datasets, Mapping):
            present_dataset_names |= set(task_datasets)
        contradictions = sorted(set(missing_datasets) & present_dataset_names)
        if contradictions:
            errors.append(
                "dataset_episode_summary missing_datasets contradict present dataset stats: "
                + ", ".join(contradictions)
            )
    return errors


def _official_score_errors(event: TimelineEvent) -> list[str]:
    errors: list[str] = []
    if event.leakage_class is not LeakageClass.privileged_eval_signal:
        errors.append("official_score events must use privileged_eval_signal leakage_class")
    if event.trial_id is not None:
        errors.append("official_score events must not set trial_id")
    errors.extend(
        _require_payload_text_value(
            event.payload,
            "event_scope",
            "post_hoc_run_summary",
            "official_score.event_scope",
        )
    )
    evidence_window, window_errors = _payload_evidence_window(
        event.payload,
        "official_score.evidence_window",
    )
    errors.extend(window_errors)
    if evidence_window is not None and not math.isclose(
        event.elapsed_sec,
        evidence_window["end_elapsed_sec"],
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        errors.append("official_score elapsed_sec must match evidence_window.end_elapsed_sec")
    total = _payload_finite_float(event.payload, "total", "official_score.total", errors)
    trial_count = _payload_nonnegative_int(
        event.payload,
        "trial_count",
        "official_score.trial_count",
        errors,
    )
    raw_trials = event.payload.get("trials")
    trial_scores: dict[str, TrialScore] = {}
    if not isinstance(raw_trials, Mapping):
        errors.append("official_score.trials must be a mapping")
    elif not raw_trials:
        errors.append("official_score.trials must not be empty")
    else:
        for trial_name, trial_score in raw_trials.items():
            if not isinstance(trial_name, str) or _OFFICIAL_TRIAL_ID_RE.fullmatch(trial_name) is None:
                errors.append("official_score.trials keys must match 'trial_<positive integer>'")
                continue
            try:
                trial_scores[trial_name] = (
                    trial_score
                    if isinstance(trial_score, TrialScore)
                    else TrialScore.from_dict(cast(Mapping[str, Any], trial_score))
                )
            except (HarnessIOError, TypeError) as exc:
                errors.append(f"official_score.trials.{trial_name}: {exc}")
    if trial_count is not None and trial_scores and trial_count != len(trial_scores):
        errors.append("official_score.trial_count must match trials")
    if total is not None and trial_scores:
        trial_total = sum(score.total for score in trial_scores.values())
        if not math.isclose(total, trial_total, rel_tol=0.0, abs_tol=1e-9):
            errors.append("official_score.total must match sum of trial totals")
    return errors


def _controller_summary_errors(event: TimelineEvent) -> list[str]:
    errors: list[str] = _mcap_trial_evidence_common_errors(event, "controller_summary")
    controller_state_count = _payload_nonnegative_int(
        event.payload,
        "controller_state_count",
        "controller_summary.controller_state_count",
        errors,
    )
    _payload_nonnegative_int(
        event.payload,
        "pose_command_count",
        "controller_summary.pose_command_count",
        errors,
    )
    if controller_state_count is None:
        return errors
    controller_fields = (
        "controller_stamp_start_sec",
        "controller_stamp_end_sec",
        "controller_duration_sec",
        "final_tcp_position",
        "final_tcp_error",
    )
    present_fields = tuple(field_name for field_name in controller_fields if field_name in event.payload)
    if controller_state_count == 0:
        if present_fields:
            errors.append(
                "controller_summary with controller_state_count == 0 must not set controller timing or final tcp evidence"
            )
        return errors
    missing_fields = tuple(field_name for field_name in controller_fields if field_name not in event.payload)
    if missing_fields:
        errors.append(
            "controller_summary with controller_state_count > 0 must set "
            + ", ".join(missing_fields)
        )
        return errors
    start_sec = _payload_nonnegative_float(
        event.payload,
        "controller_stamp_start_sec",
        "controller_summary.controller_stamp_start_sec",
        errors,
    )
    end_sec = _payload_nonnegative_float(
        event.payload,
        "controller_stamp_end_sec",
        "controller_summary.controller_stamp_end_sec",
        errors,
    )
    duration_sec = _payload_nonnegative_float(
        event.payload,
        "controller_duration_sec",
        "controller_summary.controller_duration_sec",
        errors,
    )
    _payload_vec3(event.payload, "final_tcp_position", "controller_summary.final_tcp_position", errors)
    _payload_vec3(event.payload, "final_tcp_error", "controller_summary.final_tcp_error", errors)
    if start_sec is not None and end_sec is not None and end_sec < start_sec:
        errors.append("controller_summary.controller_stamp_end_sec must be >= controller_stamp_start_sec")
    if start_sec is not None and end_sec is not None and duration_sec is not None:
        expected_duration = end_sec - start_sec
        if not math.isclose(duration_sec, expected_duration, rel_tol=0.0, abs_tol=1e-9):
            errors.append(
                "controller_summary.controller_duration_sec must match "
                "controller_stamp_end_sec - controller_stamp_start_sec"
            )
    task_hints = event.payload.get("task_hints")
    if task_hints is not None and not isinstance(task_hints, Mapping):
        errors.append("controller_summary.task_hints must be a mapping when set")
    return errors


def _contact_evidence_errors(event: TimelineEvent) -> list[str]:
    errors: list[str] = _mcap_trial_evidence_common_errors(event, "contact_evidence")
    off_limit_contact_count = _payload_nonnegative_int(
        event.payload,
        "off_limit_contact_count",
        "contact_evidence.off_limit_contact_count",
        errors,
    )
    if off_limit_contact_count is None:
        return errors
    first_contact = event.payload.get("first_off_limit_contact")
    if off_limit_contact_count == 0:
        if first_contact is not None:
            errors.append(
                "contact_evidence.first_off_limit_contact must be absent when off_limit_contact_count is 0"
            )
        return errors
    if not isinstance(first_contact, Mapping):
        errors.append(
            "contact_evidence.first_off_limit_contact must be set when off_limit_contact_count is positive"
        )
        return errors
    errors.extend(_first_contact_errors(first_contact))
    return errors


def _mcap_trial_evidence_common_errors(event: TimelineEvent, label: str) -> list[str]:
    errors: list[str] = []
    if event.leakage_class is not LeakageClass.privileged_eval_signal:
        errors.append(f"{label} events must use privileged_eval_signal leakage_class")
    if event.trial_id is None:
        errors.append(f"{label} events must set trial_id")
    errors.extend(
        _require_payload_text_value(
            event.payload,
            "event_scope",
            "post_hoc_trial_evidence",
            f"{label}.event_scope",
        )
    )
    errors.extend(_require_payload_sha256(event.payload, label))
    try:
        official_trial_id = _require_nonempty_text(
            _payload_required(event.payload, "official_trial_id"),
            f"{label}.official_trial_id",
        )
        if _OFFICIAL_TRIAL_ID_RE.fullmatch(official_trial_id) is None:
            errors.append(f"{label}.official_trial_id must match 'trial_<positive integer>'")
    except HarnessIOError as exc:
        errors.append(str(exc))
    try:
        source = _require_nonempty_text(
            _payload_required(event.payload, "mcap_trial_source"),
            f"{label}.mcap_trial_source",
        )
        if source != event.source:
            errors.append(f"{label}.mcap_trial_source must match event source")
    except HarnessIOError as exc:
        errors.append(str(exc))
    evidence_window, window_errors = _payload_evidence_window(
        event.payload,
        f"{label}.evidence_window",
    )
    errors.extend(window_errors)
    if evidence_window is not None and not math.isclose(
        event.elapsed_sec,
        evidence_window["end_elapsed_sec"],
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        errors.append(f"{label} elapsed_sec must match evidence_window.end_elapsed_sec")
    return errors


def _first_contact_errors(value: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        _require_nonnegative_int(
            _payload_required(value, "log_time_ns"),
            "contact_evidence.first_off_limit_contact.log_time_ns",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
    for key in ("collision1", "collision2"):
        try:
            _require_nonempty_text(
                _payload_required(value, key),
                f"contact_evidence.first_off_limit_contact.{key}",
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
    for key in (
        "log_elapsed_sec",
        "nearest_controller_elapsed_sec",
        "nearest_command_elapsed_sec",
        "recommended_stop_sec",
    ):
        if key in value:
            _payload_nonnegative_float(
                value,
                key,
                f"contact_evidence.first_off_limit_contact.{key}",
                errors,
            )
    for key in (
        "nearest_tcp_position",
        "nearest_tcp_error",
        "nearest_command_linear",
        "nearest_command_angular",
    ):
        if key in value:
            _payload_vec3(value, key, f"contact_evidence.first_off_limit_contact.{key}", errors)
    recommended_stop_sec = value.get("recommended_stop_sec")
    nearest_controller_elapsed_sec = value.get("nearest_controller_elapsed_sec")
    controller_context_keys = (
        "nearest_controller_elapsed_sec",
        "nearest_tcp_position",
        "nearest_tcp_error",
    )
    present_controller_context = tuple(key for key in controller_context_keys if key in value)
    if present_controller_context and len(present_controller_context) != len(controller_context_keys):
        errors.append(
            "contact_evidence.first_off_limit_contact nearest controller context must set "
            "nearest_controller_elapsed_sec, nearest_tcp_position, and nearest_tcp_error together"
        )
    command_context_keys = (
        "nearest_command_elapsed_sec",
        "nearest_command_linear",
        "nearest_command_angular",
    )
    present_command_context = tuple(key for key in command_context_keys if key in value)
    if present_command_context and len(present_command_context) != len(command_context_keys):
        errors.append(
            "contact_evidence.first_off_limit_contact nearest command context must set "
            "nearest_command_elapsed_sec, nearest_command_linear, and nearest_command_angular together"
        )
    if recommended_stop_sec is not None and nearest_controller_elapsed_sec is None:
        errors.append(
            "contact_evidence.first_off_limit_contact.recommended_stop_sec requires "
            "nearest_controller_elapsed_sec"
        )
    if (
        isinstance(recommended_stop_sec, (float, int))
        and isinstance(nearest_controller_elapsed_sec, (float, int))
        and float(recommended_stop_sec) > float(nearest_controller_elapsed_sec) + 1e-9
    ):
        errors.append(
            "contact_evidence.first_off_limit_contact.recommended_stop_sec must not exceed "
            "nearest_controller_elapsed_sec"
        )
    return errors


def _payload_required(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise HarnessIOError(f"timeline event.payload.{key} must be set")
    return payload[key]


def _require_payload_text_value(
    payload: Mapping[str, Any],
    key: str,
    expected: str,
    field_name: str,
) -> list[str]:
    try:
        value = _require_nonempty_text(_payload_required(payload, key), field_name)
    except HarnessIOError as exc:
        return [str(exc)]
    if value != expected:
        return [f"{field_name} must be {expected!r}"]
    return []


def _require_payload_sha256(payload: Mapping[str, Any], label: str) -> list[str]:
    value = payload.get("source_report_sha256")
    if not _looks_like_sha256(value):
        return [f"{label}.source_report_sha256 must be 64 hexadecimal characters"]
    return []


def _payload_evidence_window(
    payload: Mapping[str, Any],
    field_name: str,
) -> tuple[dict[str, float] | None, list[str]]:
    value = payload.get("evidence_window")
    if not isinstance(value, Mapping):
        return None, [f"{field_name} must be a mapping"]
    errors: list[str] = []
    start = _payload_nonnegative_float(value, "start_elapsed_sec", f"{field_name}.start_elapsed_sec", errors)
    end = _payload_nonnegative_float(value, "end_elapsed_sec", f"{field_name}.end_elapsed_sec", errors)
    if start is None or end is None:
        return None, errors
    if end < start:
        errors.append(f"{field_name}.end_elapsed_sec must be >= start_elapsed_sec")
    return {"start_elapsed_sec": start, "end_elapsed_sec": end}, errors


def _payload_nonnegative_int(
    payload: Mapping[str, Any],
    key: str,
    field_name: str,
    errors: list[str],
) -> int | None:
    try:
        return _require_nonnegative_int(_payload_required(payload, key), field_name)
    except HarnessIOError as exc:
        errors.append(str(exc))
        return None


def _optional_payload_nonnegative_int(
    payload: Mapping[str, Any],
    key: str,
    field_name: str,
    errors: list[str],
) -> int | None:
    if key not in payload or payload[key] is None:
        return None
    return _payload_nonnegative_int(payload, key, field_name, errors)


def _payload_nonnegative_float(
    payload: Mapping[str, Any],
    key: str,
    field_name: str,
    errors: list[str],
) -> float | None:
    try:
        return _require_nonnegative_float(_payload_required(payload, key), field_name)
    except HarnessIOError as exc:
        errors.append(str(exc))
        return None


def _payload_finite_float(
    payload: Mapping[str, Any],
    key: str,
    field_name: str,
    errors: list[str],
) -> float | None:
    try:
        return _require_finite_float(_payload_required(payload, key), field_name)
    except HarnessIOError as exc:
        errors.append(str(exc))
        return None


def _payload_vec3(
    payload: Mapping[str, Any],
    key: str,
    field_name: str,
    errors: list[str],
) -> tuple[float, float, float] | None:
    try:
        return _as_vec3(_payload_required(payload, key), field_name)
    except HarnessIOError as exc:
        errors.append(str(exc))
        return None


def _payload_nonnegative_int_tuple(
    payload: Mapping[str, Any],
    key: str,
    field_name: str,
    errors: list[str],
) -> tuple[int, ...]:
    value = payload.get(key)
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        errors.append(f"{field_name} must be a list or tuple")
        return tuple()
    items: list[int] = []
    for index, item in enumerate(value):
        try:
            items.append(_require_nonnegative_int(item, f"{field_name}[{index}]"))
        except HarnessIOError as exc:
            errors.append(str(exc))
    return tuple(items)


def _payload_nonempty_text_tuple_errors(
    payload: Mapping[str, Any],
    key: str,
    field_name: str,
    *,
    allow_empty: bool,
) -> list[str]:
    value = payload.get(key)
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        return [f"{field_name} must be a list or tuple"]
    errors: list[str] = []
    normalized: list[str] = []
    for index, item in enumerate(value):
        try:
            normalized.append(_require_nonempty_text(item, f"{field_name}[{index}]"))
        except HarnessIOError as exc:
            errors.append(str(exc))
    if not value and not allow_empty:
        errors.append(f"{field_name} must not be empty")
    if len(set(normalized)) != len(normalized):
        errors.append(f"{field_name} must not contain duplicates")
    return errors


def _dataset_stats_mapping_errors(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{field_name} must be a mapping"]
    errors: list[str] = []
    for dataset_name, stats in value.items():
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            errors.append(f"{field_name} keys must be nonempty strings")
            continue
        if not isinstance(stats, Mapping):
            errors.append(f"{field_name}.{dataset_name} must be a mapping")
            continue
        unknown_keys = sorted(key for key in stats if key not in {"shape", "dtype"})
        if unknown_keys:
            errors.append(f"{field_name}.{dataset_name} has unknown fields: {', '.join(unknown_keys)}")
        _payload_nonnegative_int_tuple(
            stats,
            "shape",
            f"{field_name}.{dataset_name}.shape",
            errors,
        )
        try:
            _require_nonempty_text(
                _payload_required(stats, "dtype"),
                f"{field_name}.{dataset_name}.dtype",
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
    return errors


def _ok_dataset_shape_errors(
    *,
    step_count: int,
    observation_datasets: Any,
    action_datasets: Any,
    task_datasets: Any,
) -> list[str]:
    errors: list[str] = []
    if isinstance(observation_datasets, Mapping):
        for dataset_name in ("pixels", "left_pixels", "right_pixels"):
            shape = _dataset_stats_shape(observation_datasets, dataset_name)
            dtype = _dataset_stats_dtype(observation_datasets, dataset_name)
            if shape is not None:
                if len(shape) != 4 or shape[0] != step_count or shape[-1] != 3:
                    errors.append(
                        f"dataset_episode_summary.observation_datasets.{dataset_name}.shape "
                        "must be (step_count, height, width, 3) when ok is true"
                    )
            if dtype is not None and dtype != "uint8":
                errors.append(
                    f"dataset_episode_summary.observation_datasets.{dataset_name}.dtype "
                    "must be uint8 when ok is true"
                )
        for dataset_name in ("proprio", "state"):
            shape = _dataset_stats_shape(observation_datasets, dataset_name)
            if shape is not None and (len(shape) != 2 or shape[0] != step_count or shape[1] != 32):
                errors.append(
                    f"dataset_episode_summary.observation_datasets.{dataset_name}.shape "
                    "must be (step_count, 32) when ok is true"
                )
    if isinstance(action_datasets, Mapping):
        shape = _dataset_stats_shape(action_datasets, "action")
        if shape is not None and (len(shape) != 2 or shape[0] != step_count or shape[1] != 6):
            errors.append(
                "dataset_episode_summary.action_datasets.action.shape must be "
                "(step_count, 6) when ok is true"
            )
    if isinstance(task_datasets, Mapping):
        for dataset_name in ("task_id", "plug_type", "port_type", "target_module_name"):
            shape = _dataset_stats_shape(task_datasets, dataset_name)
            if shape is not None and (len(shape) < 1 or shape[0] != step_count):
                errors.append(
                    f"dataset_episode_summary.task_datasets.{dataset_name}.shape "
                    "first dimension must match step_count when ok is true"
                )
    return errors


def _dataset_stats_shape(datasets: Mapping[str, Any], dataset_name: str) -> tuple[int, ...] | None:
    stats = datasets.get(dataset_name)
    if not isinstance(stats, Mapping):
        return None
    shape = stats.get("shape")
    if isinstance(shape, (str, bytes, bytearray)) or not isinstance(shape, (list, tuple)):
        return None
    normalized: list[int] = []
    for dimension in shape:
        if type(dimension) is not int:
            return None
        normalized.append(dimension)
    return tuple(normalized)


def _dataset_stats_dtype(datasets: Mapping[str, Any], dataset_name: str) -> str | None:
    stats = datasets.get(dataset_name)
    if not isinstance(stats, Mapping):
        return None
    dtype = stats.get("dtype")
    return dtype if isinstance(dtype, str) else None


def _expected_episode_offsets(episode_lengths: tuple[int, ...]) -> tuple[int, ...]:
    offsets: list[int] = []
    running_offset = 0
    for length in episode_lengths:
        offsets.append(running_offset)
        running_offset += length
    return tuple(offsets)


def _text_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        return tuple()
    return tuple(item for item in value if isinstance(item, str))


def _looks_like_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _timeline_action_payload_is_nonzero(payload: Mapping[str, Any]) -> bool:
    raw_payload = payload.get("policy_payload")
    if not isinstance(raw_payload, Mapping):
        return False
    try:
        linear = _as_vec3(raw_payload.get("linear"), "timeline action payload.linear")
        angular = _as_vec3(raw_payload.get("angular"), "timeline action payload.angular")
    except HarnessIOError:
        return False
    return any(abs(value) > 1e-12 for value in (*linear, *angular))


def _as_vec3(value: Any, field_name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a 3-number list or tuple")
    if len(value) != 3:
        raise HarnessIOError(f"{field_name} must contain exactly 3 numbers")
    return (
        _require_finite_float(value[0], f"{field_name}[0]"),
        _require_finite_float(value[1], f"{field_name}[1]"),
        _require_finite_float(value[2], f"{field_name}[2]"),
    )


def _require_finite_float(value: Any, field_name: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(float(value)):
        raise HarnessIOError(f"{field_name} must be a finite number")
    return float(value)


def _source_artifact_errors(source_artifacts: tuple[ArtifactRef, ...], run_id: str) -> list[str]:
    errors: list[str] = []
    if not source_artifacts:
        return ["episode trace.source_artifacts must not be empty"]
    artifacts_by_kind: dict[str, list[ArtifactRef]] = {}
    for artifact in source_artifacts:
        artifacts_by_kind.setdefault(artifact.kind, []).append(artifact)
        if artifact.sha256 is None:
            errors.append(f"episode trace source artifact {artifact.kind!r} must set sha256")
        if "producer" not in artifact.provenance:
            errors.append(f"episode trace source artifact {artifact.kind!r} must set provenance.producer")
        if "run_id" not in artifact.provenance:
            errors.append(f"episode trace source artifact {artifact.kind!r} must set provenance.run_id")
        elif artifact.provenance.get("run_id") != run_id:
            errors.append(
                f"episode trace source artifact {artifact.kind!r} provenance.run_id must match episode trace run_id"
            )
        if artifact.kind in {"mcap_eval_bundle", "hdf5_dataset"} and artifact.provenance.get("source_report") is not None:
            errors.append(
                f"episode trace source artifact {artifact.kind!r} provenance.source_report "
                "must not be set; use the typed report source artifact instead"
            )
    for required_kind in ("policy_trace_jsonl", "scoring_yaml"):
        if required_kind not in artifacts_by_kind:
            errors.append(f"episode trace.source_artifacts must include {required_kind}")
        elif len(artifacts_by_kind[required_kind]) != 1:
            errors.append(f"episode trace.source_artifacts must include exactly one {required_kind}")
    return errors


def _post_hoc_evidence_source_artifact_errors(
    *,
    source_artifacts: tuple[ArtifactRef, ...],
    trials: tuple[TrialTrace, ...],
    run_events: tuple[TimelineEvent, ...],
) -> list[str]:
    artifacts_by_kind: dict[str, list[ArtifactRef]] = {}
    for artifact in source_artifacts:
        artifacts_by_kind.setdefault(artifact.kind, []).append(artifact)
    events = tuple(event for trial in trials for event in trial.events) + tuple(run_events)
    mcap_events = tuple(
        event
        for event in events
        if event.event_kind in {
            TimelineEventKind.controller_summary,
            TimelineEventKind.contact_evidence,
        }
    )
    dataset_events = tuple(
        event
        for event in events
        if event.event_kind is TimelineEventKind.dataset_episode_summary
    )
    score_events = tuple(
        event for event in events if event.event_kind is TimelineEventKind.official_score
    )
    errors: list[str] = []
    scoring_artifacts = tuple(artifacts_by_kind.get("scoring_yaml", ()))
    if len(scoring_artifacts) == 1:
        scoring_artifact = scoring_artifacts[0]
        errors.extend(
            _official_score_artifact_binding_errors(
                artifact=scoring_artifact,
                events=score_events,
            )
        )
    if score_events:
        if len(scoring_artifacts) != 1:
            errors.append(
                "episode trace official_score events require exactly one scoring_yaml source artifact"
            )
        else:
            scoring_artifact = scoring_artifacts[0]
            errors.extend(
                _source_event_projection_sha256_binding_errors(
                    artifact=scoring_artifact,
                    events=score_events,
                    label="official score",
                )
            )
            for event in score_events:
                if not _event_source_names_artifact(event.source, scoring_artifact):
                    errors.append(
                        "episode trace official_score event source must match the scoring_yaml source artifact"
                    )
                    break
    if not mcap_events:
        mcap_report_artifacts = tuple(artifacts_by_kind.get("mcap_eval_report", ()))
        if mcap_report_artifacts:
            errors.append(
                "episode trace.source_artifacts must not include mcap_eval_report "
                "without MCAP evidence events"
            )
    if mcap_events:
        mcap_artifacts = tuple(artifacts_by_kind.get("mcap_eval_bundle", ()))
        if len(mcap_artifacts) != 1:
            errors.append(
                "episode trace MCAP evidence events require exactly one mcap_eval_bundle source artifact"
            )
        else:
            mcap_artifact = mcap_artifacts[0]
            errors.extend(
                _source_report_sha256_binding_errors(
                    artifact=mcap_artifact,
                    events=mcap_events,
                    label="MCAP evidence",
                )
            )
            errors.extend(
                _source_event_projection_sha256_binding_errors(
                    artifact=mcap_artifact,
                    events=mcap_events,
                    label="MCAP evidence",
                )
            )
            errors.extend(
                _typed_report_artifact_binding_errors(
                    source_artifact=mcap_artifact,
                    artifacts_by_kind=artifacts_by_kind,
                    report_artifact_kind="mcap_eval_report",
                    events=mcap_events,
                    report_kind="mcap_eval_bundle",
                    label="MCAP evidence",
                )
            )
            for event in mcap_events:
                if not _event_source_under_artifact(event.source, mcap_artifact):
                    errors.append(
                        "episode trace MCAP evidence event source must be rooted under "
                        "the mcap_eval_bundle source artifact"
                    )
                    break
    if not dataset_events:
        dataset_report_artifacts = tuple(artifacts_by_kind.get("hdf5_dataset_report", ()))
        if dataset_report_artifacts:
            errors.append(
                "episode trace.source_artifacts must not include hdf5_dataset_report "
                "without dataset summary events"
            )
    if dataset_events:
        dataset_artifacts = tuple(artifacts_by_kind.get("hdf5_dataset", ()))
        if len(dataset_artifacts) != 1:
            errors.append(
                "episode trace dataset summary events require exactly one hdf5_dataset source artifact"
            )
        else:
            dataset_artifact = dataset_artifacts[0]
            errors.extend(
                _source_report_sha256_binding_errors(
                    artifact=dataset_artifact,
                    events=dataset_events,
                    label="dataset summary",
                )
            )
            errors.extend(
                _source_event_projection_sha256_binding_errors(
                    artifact=dataset_artifact,
                    events=dataset_events,
                    label="dataset summary",
                )
            )
            errors.extend(
                _typed_report_artifact_binding_errors(
                    source_artifact=dataset_artifact,
                    artifacts_by_kind=artifacts_by_kind,
                    report_artifact_kind="hdf5_dataset_report",
                    events=dataset_events,
                    report_kind="hdf5_dataset",
                    label="dataset summary",
                )
            )
            for event in dataset_events:
                if not _event_source_matches_artifact(event.source, dataset_artifact):
                    errors.append(
                        "episode trace dataset summary event source must match the hdf5_dataset source artifact"
                    )
                    break
    return errors


def _episode_evidence_consistency_errors(
    *,
    trials: tuple[TrialTrace, ...],
    run_events: tuple[TimelineEvent, ...],
) -> list[str]:
    errors: list[str] = []
    for trial in trials:
        if any(
            event.event_kind in {
                TimelineEventKind.controller_summary,
                TimelineEventKind.contact_evidence,
                TimelineEventKind.dataset_episode_summary,
                TimelineEventKind.official_score,
            }
            for event in trial.events
        ):
            errors.append("post-hoc evidence events must be stored in episode trace run_events")
            break
    score_events = tuple(
        event for event in run_events if event.event_kind is TimelineEventKind.official_score
    )
    if score_events:
        if len(score_events) != 1:
            errors.append("episode trace must include exactly one official_score event")
        else:
            score_event = score_events[0]
            if _event_evidence_window(score_event) != _run_trace_window(trials):
                errors.append("official_score evidence_window must match episode trace run window")
            errors.extend(_official_score_event_matches_trials_errors(score_event, trials))
    official_trial_by_policy = _official_trial_by_policy_trial(trials)
    mcap_run_events = tuple(
        event
        for event in run_events
        if event.event_kind in {
            TimelineEventKind.controller_summary,
            TimelineEventKind.contact_evidence,
        }
    )
    if mcap_run_events:
        errors.extend(_official_trial_mapping_errors(trials, official_trial_by_policy))
        event_trial_ids = {
            event.trial_id
            for event in mcap_run_events
            if event.trial_id is not None
        }
        expected_trial_ids = {trial.trial_id for trial in trials}
        if event_trial_ids != expected_trial_ids:
            missing_trial_ids = sorted(expected_trial_ids - event_trial_ids)
            extra_trial_ids = sorted(event_trial_ids - expected_trial_ids)
            details = []
            if missing_trial_ids:
                details.append("missing MCAP evidence for " + ", ".join(missing_trial_ids))
            if extra_trial_ids:
                details.append("unknown MCAP evidence for " + ", ".join(extra_trial_ids))
            errors.append(
                "episode trace MCAP evidence must cover every policy trial exactly once: "
                + "; ".join(details)
            )
    for event in mcap_run_events:
        if event.trial_id is None:
            continue
        expected_official = official_trial_by_policy.get(event.trial_id)
        actual_official = event.payload.get("official_trial_id")
        if expected_official is not None and actual_official != expected_official:
            errors.append(
                "episode trace MCAP evidence official_trial_id must match "
                f"policy trial {event.trial_id!r}"
            )
    mcap_events_by_trial: dict[str, list[TimelineEvent]] = {}
    for event in mcap_run_events:
        if event.trial_id is None:
            continue
        mcap_events_by_trial.setdefault(event.trial_id, []).append(event)
    for trial_id, events in mcap_events_by_trial.items():
        controller_events = tuple(
            event for event in events if event.event_kind is TimelineEventKind.controller_summary
        )
        contact_events = tuple(
            event for event in events if event.event_kind is TimelineEventKind.contact_evidence
        )
        if len(controller_events) != 1 or len(contact_events) != 1:
            errors.append(
                f"episode trace MCAP evidence for {trial_id!r} must include exactly one "
                "controller_summary and one contact_evidence event"
            )
            continue
        errors.extend(_mcap_pair_consistency_errors(controller_events[0], contact_events[0]))
        expected_bounds = _trial_bounds_by_id(trials).get(trial_id)
        if expected_bounds is not None:
            for event in (controller_events[0], contact_events[0]):
                event_window = _event_evidence_window(event)
                if event_window != expected_bounds:
                    errors.append(
                        f"episode trace MCAP evidence window for {trial_id!r} must match trial bounds"
                    )
                    break
    dataset_events = tuple(
        event
        for event in run_events
        if event.event_kind is TimelineEventKind.dataset_episode_summary
    )
    if dataset_events:
        if len(dataset_events) != 1:
            errors.append("episode trace must include exactly one dataset_episode_summary event")
        run_window = _run_trace_window(trials)
        for event in dataset_events:
            if _event_evidence_window(event) != run_window:
                errors.append("dataset_episode_summary evidence_window must match episode trace run window")
                break
    return errors


def _official_score_event_matches_trials_errors(
    score_event: TimelineEvent,
    trials: tuple[TrialTrace, ...],
) -> list[str]:
    errors: list[str] = []
    official_trial_by_policy = _official_trial_by_policy_trial(trials)
    errors.extend(_official_trial_mapping_errors(trials, official_trial_by_policy))
    raw_score_trials = score_event.payload.get("trials")
    if not isinstance(raw_score_trials, Mapping):
        return errors
    score_by_official: dict[str, TrialScore] = {}
    for official_trial_id, raw_trial_score in raw_score_trials.items():
        if not isinstance(official_trial_id, str):
            continue
        try:
            score_by_official[official_trial_id] = (
                raw_trial_score
                if isinstance(raw_trial_score, TrialScore)
                else TrialScore.from_dict(cast(Mapping[str, Any], raw_trial_score))
            )
        except (HarnessIOError, TypeError):
            continue
    expected_official_ids = set(official_trial_by_policy.values())
    score_official_ids = set(score_by_official)
    if expected_official_ids != score_official_ids:
        missing = sorted(expected_official_ids - score_official_ids)
        extra = sorted(score_official_ids - expected_official_ids)
        details = []
        if missing:
            details.append("missing scores for " + ", ".join(missing))
        if extra:
            details.append("unknown scores for " + ", ".join(extra))
        errors.append("official_score.trials must match policy official_trial_id mapping: " + "; ".join(details))
    for trial in trials:
        official_trial_id = official_trial_by_policy.get(trial.trial_id)
        if official_trial_id is None:
            continue
        event_score = score_by_official.get(official_trial_id)
        if event_score is None:
            continue
        if trial.score is None:
            errors.append(f"policy trial {trial.trial_id!r} must carry score when official_score is present")
            continue
        if event_score.to_dict() != trial.score.to_dict():
            errors.append(
                "official_score.trials must match TrialTrace.score for "
                f"policy trial {trial.trial_id!r}"
            )
    total = score_event.payload.get("total")
    if type(total) in (float, int):
        total_value = cast(float | int, total)
        trial_total = sum(trial.score.total for trial in trials if trial.score is not None)
        if not math.isclose(float(total_value), trial_total, rel_tol=0.0, abs_tol=1e-9):
            errors.append("official_score.total must match TrialTrace.score totals")
    return errors


def _official_trial_by_policy_trial(trials: tuple[TrialTrace, ...]) -> dict[str, str]:
    policy_to_official: dict[str, str] = {}
    for trial in trials:
        observed: set[str] = set()
        for event in trial.events:
            if event.event_kind not in {
                TimelineEventKind.policy_event,
                TimelineEventKind.observation_event,
                TimelineEventKind.action_event,
                TimelineEventKind.safety_guard,
                TimelineEventKind.error,
            }:
                continue
            official_trial_id = event.payload.get("official_trial_id")
            if isinstance(official_trial_id, str):
                observed.add(official_trial_id)
        if len(observed) == 1:
            policy_to_official[trial.trial_id] = next(iter(observed))
    return policy_to_official


def _official_trial_mapping_errors(
    trials: tuple[TrialTrace, ...],
    policy_to_official: Mapping[str, str],
) -> list[str]:
    errors: list[str] = []
    for trial in trials:
        observed: set[str] = set()
        missing_count = 0
        for event in trial.events:
            if event.event_kind not in {
                TimelineEventKind.policy_event,
                TimelineEventKind.observation_event,
                TimelineEventKind.action_event,
                TimelineEventKind.safety_guard,
                TimelineEventKind.error,
            }:
                continue
            official_trial_id = event.payload.get("official_trial_id")
            if isinstance(official_trial_id, str):
                if _OFFICIAL_TRIAL_ID_RE.fullmatch(official_trial_id) is None:
                    errors.append(
                        f"policy trial {trial.trial_id!r} official_trial_id must match "
                        "'trial_<positive integer>'"
                    )
                observed.add(official_trial_id)
            else:
                missing_count += 1
        if not observed:
            errors.append(
                f"policy trial {trial.trial_id!r} must carry official_trial_id "
                "when MCAP evidence is present"
            )
        elif len(observed) > 1:
            errors.append(
                f"policy trial {trial.trial_id!r} carries multiple official_trial_id values"
            )
        elif missing_count:
            errors.append(
                f"policy trial {trial.trial_id!r} must carry official_trial_id on every event "
                "when MCAP evidence is present"
            )
    if len(policy_to_official) != len(trials):
        missing = sorted(trial.trial_id for trial in trials if trial.trial_id not in policy_to_official)
        if missing:
            errors.append(
                "episode trace MCAP evidence requires explicit policy trial to official trial mapping for: "
                + ", ".join(missing)
            )
    duplicate_official_ids = sorted(
        official_trial_id
        for official_trial_id in set(policy_to_official.values())
        if tuple(policy_to_official.values()).count(official_trial_id) > 1
    )
    if duplicate_official_ids:
        errors.append(
            "episode trace MCAP evidence requires one-to-one policy trial to official trial mapping; "
            "duplicate official_trial_id values: "
            + ", ".join(duplicate_official_ids)
        )
    return errors


def _mcap_pair_consistency_errors(
    controller_event: TimelineEvent,
    contact_event: TimelineEvent,
) -> list[str]:
    errors: list[str] = []
    if controller_event.source != contact_event.source:
        errors.append("paired MCAP controller/contact events must share source")
    if controller_event.payload.get("official_trial_id") != contact_event.payload.get("official_trial_id"):
        errors.append("paired MCAP controller/contact events must share official_trial_id")
    controller_count = controller_event.payload.get("controller_state_count")
    pose_command_count = controller_event.payload.get("pose_command_count")
    first_contact = contact_event.payload.get("first_off_limit_contact")
    if not isinstance(first_contact, Mapping):
        return errors
    if controller_count == 0:
        for key in (
            "log_elapsed_sec",
            "nearest_controller_elapsed_sec",
            "nearest_tcp_position",
            "nearest_tcp_error",
            "recommended_stop_sec",
        ):
            if key in first_contact:
                errors.append(
                    "contact_evidence first_off_limit_contact controller context requires "
                    "controller_summary.controller_state_count > 0"
                )
                break
    if pose_command_count == 0:
        for key in (
            "nearest_command_elapsed_sec",
            "nearest_command_linear",
            "nearest_command_angular",
        ):
            if key in first_contact:
                errors.append(
                    "contact_evidence first_off_limit_contact command context requires "
                    "controller_summary.pose_command_count > 0"
                )
                break
    return errors


def _trial_bounds_by_id(trials: tuple[TrialTrace, ...]) -> dict[str, dict[str, float]]:
    return {
        trial.trial_id: {
            "start_elapsed_sec": trial.start_elapsed_sec,
            "end_elapsed_sec": trial.end_elapsed_sec,
        }
        for trial in trials
    }


def _run_trace_window(trials: tuple[TrialTrace, ...]) -> dict[str, float]:
    return {
        "start_elapsed_sec": min(trial.start_elapsed_sec for trial in trials),
        "end_elapsed_sec": max(trial.end_elapsed_sec for trial in trials),
    }


def _event_evidence_window(event: TimelineEvent) -> dict[str, float] | None:
    raw_window = event.payload.get("evidence_window")
    if not isinstance(raw_window, Mapping):
        return None
    start = raw_window.get("start_elapsed_sec")
    end = raw_window.get("end_elapsed_sec")
    if type(start) not in (float, int) or type(end) not in (float, int):
        return None
    start_value = cast(float | int, start)
    end_value = cast(float | int, end)
    return {
        "start_elapsed_sec": float(start_value),
        "end_elapsed_sec": float(end_value),
    }


def _source_report_sha256_binding_errors(
    *,
    artifact: ArtifactRef,
    events: tuple[TimelineEvent, ...],
    label: str,
) -> list[str]:
    artifact_report_sha256 = artifact.provenance.get("report_sha256")
    if not _looks_like_sha256(artifact_report_sha256):
        return [f"episode trace {label} source artifact provenance.report_sha256 must be set"]
    mismatches = tuple(
        event
        for event in events
        if event.payload.get("source_report_sha256") != artifact_report_sha256
    )
    if mismatches:
        return [f"episode trace {label} events must match source artifact provenance.report_sha256"]
    return []


def _official_score_artifact_binding_errors(
    *,
    artifact: ArtifactRef,
    events: tuple[TimelineEvent, ...],
) -> list[str]:
    try:
        scoring_path = local_artifact_path(
            path=artifact.path,
            uri=artifact.uri,
            field_name="scoring_yaml source artifact",
        )
    except HarnessIOError as exc:
        return [str(exc)]
    if scoring_path is None:
        return ["episode trace official_score requires a local scoring_yaml source artifact"]
    errors: list[str] = []
    if not scoring_path.exists():
        errors.append("episode trace official_score scoring_yaml source artifact path must exist")
    elif not scoring_path.is_file():
        errors.append("episode trace official_score scoring_yaml source artifact path must be a file")
    elif artifact.sha256 != sha256_file(scoring_path):
        errors.append("episode trace official_score scoring_yaml source artifact sha256 must match path")
    else:
        try:
            from aic_signal_harness.scoring import parse_scoring_yaml

            report = parse_scoring_yaml(scoring_path)
            errors.extend(
                _scoring_source_report_provenance_errors(
                    artifact=artifact,
                    report=report,
                    scoring_path=scoring_path,
                )
            )
            errors.extend(_score_events_match_source_report_errors(report, events))
        except HarnessIOError as exc:
            errors.append(f"episode trace official_score scoring_yaml must parse: {exc}")
    return errors


def _typed_report_artifact_binding_errors(
    *,
    source_artifact: ArtifactRef,
    artifacts_by_kind: Mapping[str, list[ArtifactRef]],
    report_artifact_kind: str,
    events: tuple[TimelineEvent, ...],
    report_kind: str,
    label: str,
) -> list[str]:
    report_artifacts = tuple(artifacts_by_kind.get(report_artifact_kind, ()))
    if len(report_artifacts) != 1:
        return [
            f"episode trace {label} events require exactly one {report_artifact_kind} source artifact"
        ]
    report_artifact = report_artifacts[0]
    report_payload, report_errors = _read_typed_report_artifact(report_artifact, report_artifact_kind)
    if report_payload is None:
        return report_errors
    try:
        if report_kind == "mcap_eval_bundle":
            typed_report: McapEvalBundleReport | Hdf5DatasetReport = McapEvalBundleReport.from_dict(report_payload)
        elif report_kind == "hdf5_dataset":
            typed_report = Hdf5DatasetReport.from_dict(report_payload)
        else:
            return [f"episode trace {label} source artifact has unknown report kind"]
    except HarnessIOError as exc:
        return [f"episode trace {label} {report_artifact_kind} is invalid: {exc}"]
    report_sha256 = _canonical_mapping_sha256(typed_report.to_dict())
    errors = list(report_errors)
    if source_artifact.provenance.get("source_report") is not None:
        errors.append(
            f"episode trace {label} source artifact provenance.source_report "
            f"must not be set; use the {report_artifact_kind} source artifact"
        )
    artifact_report_sha256 = source_artifact.provenance.get("report_sha256")
    if artifact_report_sha256 != report_sha256:
        errors.append(f"episode trace {label} source artifact provenance.report_sha256 must match {report_artifact_kind}")
    if isinstance(typed_report, McapEvalBundleReport):
        errors.extend(
            _mcap_raw_source_artifact_binding_errors(
                source_artifact,
                typed_report,
                label,
            )
        )
        errors.extend(_mcap_events_match_source_report_errors(typed_report, events, report_sha256))
        return errors
    hdf5_report = cast(Hdf5DatasetReport, typed_report)
    errors.extend(_hdf5_raw_source_artifact_binding_errors(source_artifact, hdf5_report, label))
    errors.extend(
        _hdf5_report_matches_local_bytes_errors(
            source_artifact=source_artifact,
            report=hdf5_report,
            label=label,
        )
    )
    errors.extend(
        _dataset_events_match_source_report_errors(
            hdf5_report,
            events,
            report_sha256,
        )
    )
    return errors


def _read_typed_report_artifact(
    artifact: ArtifactRef,
    label: str,
) -> tuple[Mapping[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        report_path = local_artifact_path(
            path=artifact.path,
            uri=artifact.uri,
            field_name=f"{label} source artifact",
        )
    except HarnessIOError as exc:
        return None, [str(exc)]
    if report_path is None:
        return None, [f"episode trace {label} source artifact must be a local JSON file"]
    if not report_path.exists():
        return None, [f"episode trace {label} source artifact path must exist"]
    if not report_path.is_file():
        return None, [f"episode trace {label} source artifact path must be a file"]
    if artifact.sha256 is None:
        errors.append(f"episode trace {label} source artifact sha256 must be set")
    elif artifact.sha256 != sha256_file(report_path):
        errors.append(f"episode trace {label} source artifact sha256 must match path")
    try:
        return read_json(report_path), errors
    except HarnessIOError as exc:
        return None, errors + [str(exc)]


def _scoring_source_report_provenance_errors(
    *,
    artifact: ArtifactRef,
    report: ScoreReport,
    scoring_path: Path,
) -> list[str]:
    errors: list[str] = []
    source_report_payload = artifact.provenance.get("source_report")
    if source_report_payload is None:
        errors.append("episode trace scoring_yaml source artifact provenance.source_report must be set")
        provenance_report: ScoreReport | None = None
    elif not isinstance(source_report_payload, Mapping):
        errors.append(
            "episode trace scoring_yaml source artifact provenance.source_report must be a mapping"
        )
        provenance_report = None
    else:
        try:
            provenance_report = ScoreReport.from_dict(source_report_payload)
        except HarnessIOError as exc:
            errors.append(
                "episode trace scoring_yaml source artifact provenance.source_report "
                f"must parse as a ScoreReport: {exc}"
            )
            provenance_report = None

    if provenance_report is not None:
        report_sha256 = artifact.provenance.get("report_sha256")
        expected_report_sha256 = _canonical_mapping_sha256(provenance_report.to_dict())
        if report_sha256 != expected_report_sha256:
            errors.append(
                "episode trace scoring_yaml source artifact provenance.report_sha256 "
                "must match source_report"
            )
        source_aliases = tuple(
            source
            for source in (
                report.source,
                provenance_report.source,
                artifact.path,
                artifact.uri,
                str(scoring_path.resolve(strict=False)),
            )
            if source is not None
        )
        if not report.equivalent_to(provenance_report, source_aliases=source_aliases):
            errors.append(
                "episode trace scoring_yaml source artifact provenance.source_report "
                "must match parsed scoring.yaml bytes"
            )
    elif artifact.provenance.get("report_sha256") is None:
        errors.append("episode trace scoring_yaml source artifact provenance.report_sha256 must be set")
    return errors


def _mcap_raw_source_artifact_binding_errors(
    artifact: ArtifactRef,
    report: McapEvalBundleReport,
    label: str,
) -> list[str]:
    errors: list[str] = []
    expected_sha256 = _mcap_bundle_sha256_from_report(report)
    if artifact.sha256 is None:
        errors.append(f"episode trace {label} source artifact sha256 must be set")
    elif artifact.sha256 != expected_sha256:
        errors.append(
            f"episode trace {label} source artifact sha256 must match mcap_eval_report"
        )
    try:
        bundle_path = local_artifact_path(
            path=artifact.path,
            uri=artifact.uri,
            field_name=f"{label} source artifact",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        return errors
    if bundle_path is None:
        errors.append(
            f"episode trace {label} source artifact must be a local byte-verifiable directory"
        )
        return errors
    if not bundle_path.exists():
        errors.append(f"episode trace {label} source artifact path must exist")
        return errors
    if not bundle_path.is_dir():
        errors.append(f"episode trace {label} source artifact path must be a directory")
        return errors
    for trial in report.trials:
        trial_path = _local_source_path(trial.source)
        if trial_path is None:
            errors.append(
                f"episode trace {label} local source artifact has nonlocal trial source: "
                f"{trial.trial_id}"
            )
            continue
        try:
            trial_path.relative_to(bundle_path.resolve(strict=False))
        except ValueError:
            errors.append(
                f"episode trace {label} trial source must be rooted under source artifact: "
                f"{trial.trial_id}"
            )
            continue
        if not trial_path.exists():
            errors.append(
                f"episode trace {label} trial source path must exist: {trial.trial_id}"
            )
        elif not trial_path.is_file():
            errors.append(
                f"episode trace {label} trial source path must be a file: {trial.trial_id}"
            )
        elif sha256_file(trial_path) != trial.sha256:
            errors.append(
                f"episode trace {label} trial source sha256 must match path: {trial.trial_id}"
            )
    return errors


def _hdf5_raw_source_artifact_binding_errors(
    artifact: ArtifactRef,
    report: Hdf5DatasetReport,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if artifact.sha256 is None:
        errors.append(f"episode trace {label} source artifact sha256 must be set")
    elif artifact.sha256 != report.sha256:
        errors.append(
            f"episode trace {label} source artifact sha256 must match hdf5_dataset_report"
        )
    try:
        dataset_path = local_artifact_path(
            path=artifact.path,
            uri=artifact.uri,
            field_name=f"{label} source artifact",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        return errors
    if dataset_path is None:
        errors.append(f"episode trace {label} source artifact must be a local byte-verifiable file")
        return errors
    if not dataset_path.exists():
        errors.append(f"episode trace {label} source artifact path must exist")
    elif not dataset_path.is_file():
        errors.append(f"episode trace {label} source artifact path must be a file")
    elif artifact.sha256 is not None and sha256_file(dataset_path) != artifact.sha256:
        errors.append(f"episode trace {label} source artifact sha256 must match path")
    return errors


def _hdf5_report_matches_local_bytes_errors(
    *,
    source_artifact: ArtifactRef,
    report: Hdf5DatasetReport,
    label: str,
) -> list[str]:
    try:
        dataset_path = local_artifact_path(
            path=source_artifact.path,
            uri=source_artifact.uri,
            field_name=f"{label} source artifact",
        )
    except HarnessIOError as exc:
        return [str(exc)]
    if dataset_path is None:
        return [f"episode trace {label} source artifact must be a local byte-verifiable file"]
    if not dataset_path.exists():
        return [f"episode trace {label} source artifact path must exist"]
    if not dataset_path.is_file():
        return [f"episode trace {label} source artifact path must be a file"]
    if source_artifact.sha256 is not None and sha256_file(dataset_path) != source_artifact.sha256:
        return [f"episode trace {label} source artifact sha256 must match path"]
    try:
        validated_report = validate_hdf5_dataset(
            dataset_path,
            min_episodes=report.thresholds.min_episodes,
            min_steps=report.thresholds.min_steps,
            validated_at_utc=report.validated_at_utc,
        )
    except HarnessIOError as exc:
        return [f"episode trace {label} source artifact must validate as hdf5_dataset: {exc}"]
    validated_report_payload = validated_report.to_dict()
    validated_report_payload["source"] = report.source
    if Hdf5DatasetReport.from_dict(validated_report_payload) != report:
        return [f"episode trace {label} source artifact must match supplied hdf5_dataset_report bytes"]
    return []


def _mcap_bundle_sha256_from_report(report: McapEvalBundleReport) -> str:
    digest = hashlib.sha256()
    bundle_root = _local_source_path(report.source) if not _is_uri_source(report.source) else None
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


def _is_uri_source(source: str) -> bool:
    return bool(urlparse(source).scheme)


def _score_events_match_source_report_errors(
    report: ScoreReport,
    events: tuple[TimelineEvent, ...],
) -> list[str]:
    if len(events) != 1:
        return []
    event = events[0]
    errors: list[str] = []
    if event.source != report.source and _local_source_path(event.source) != _local_source_path(report.source):
        errors.append("official_score source must match source_report.source")
    if report.parsed_at_utc is not None and event.emitted_at_utc != report.parsed_at_utc:
        errors.append("official_score emitted_at_utc must match source_report.parsed_at_utc")
    expected_trials = report.to_dict()["trials"]
    expected_payload = {
        "event_scope": "post_hoc_run_summary",
        "total": report.total,
        "trial_count": report.trial_count,
        "trials": expected_trials,
    }
    if not _payload_exactly_matches_expected_report_projection(
        event.payload,
        expected_payload,
        allowed_extra_keys=("evidence_window",),
    ):
        errors.append("official_score payload must exactly match source_report projection")
    return errors


def _mcap_events_match_source_report_errors(
    report: McapEvalBundleReport,
    events: tuple[TimelineEvent, ...],
    report_sha256: str,
) -> list[str]:
    errors: list[str] = []
    trials_by_id = {trial.trial_id: trial for trial in report.trials}
    for event in events:
        official_trial_id = event.payload.get("official_trial_id")
        if not isinstance(official_trial_id, str):
            continue
        trial = trials_by_id.get(official_trial_id)
        if trial is None:
            errors.append("MCAP evidence official_trial_id must exist in source_report.trials")
            continue
        if event.source != trial.source:
            errors.append("MCAP evidence source must match source_report trial source")
        if event.emitted_at_utc != report.analyzed_at_utc:
            errors.append("MCAP evidence emitted_at_utc must match source_report.analyzed_at_utc")
        expected_payload = (
            _mcap_controller_source_report_payload(trial, report_sha256)
            if event.event_kind is TimelineEventKind.controller_summary
            else _mcap_contact_source_report_payload(trial, report_sha256)
        )
        if not _payload_exactly_matches_expected_report_projection(
            event.payload,
            expected_payload,
            allowed_extra_keys=("evidence_window",),
        ):
            errors.append(
                f"{event.event_kind.value} payload must match source_report trial {official_trial_id!r}"
            )
    return errors


def _mcap_controller_source_report_payload(
    trial: Any,
    report_sha256: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event_scope": "post_hoc_trial_evidence",
        "source_report_sha256": report_sha256,
        "official_trial_id": trial.trial_id,
        "mcap_trial_source": trial.source,
        "controller_state_count": trial.controller_state_count,
        "pose_command_count": trial.pose_command_count,
    }
    for field_name in (
        "controller_stamp_start_sec",
        "controller_stamp_end_sec",
        "controller_duration_sec",
    ):
        field_value = getattr(trial, field_name)
        if field_value is not None:
            payload[field_name] = field_value
    for field_name in ("final_tcp_position", "final_tcp_error"):
        field_value = getattr(trial, field_name)
        if field_value is not None:
            payload[field_name] = list(field_value)
    if trial.task_hints is not None:
        payload["task_hints"] = trial.task_hints.to_dict()
    return payload


def _mcap_contact_source_report_payload(
    trial: Any,
    report_sha256: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event_scope": "post_hoc_trial_evidence",
        "source_report_sha256": report_sha256,
        "official_trial_id": trial.trial_id,
        "mcap_trial_source": trial.source,
        "off_limit_contact_count": trial.off_limit_contact_count,
    }
    if trial.first_off_limit_contact is not None:
        payload["first_off_limit_contact"] = trial.first_off_limit_contact.to_dict()
    return payload


def _dataset_events_match_source_report_errors(
    report: Hdf5DatasetReport,
    events: tuple[TimelineEvent, ...],
    report_sha256: str,
) -> list[str]:
    if len(events) != 1:
        return []
    event = events[0]
    errors: list[str] = []
    if event.source != report.source:
        errors.append("dataset_episode_summary source must match source_report.source")
    if event.emitted_at_utc != report.validated_at_utc:
        errors.append("dataset_episode_summary emitted_at_utc must match source_report.validated_at_utc")
    expected_payload = {
        "event_scope": "offline_dataset_summary",
        "source_report_sha256": report_sha256,
        "validated_at_utc": report.validated_at_utc,
        "ok": report.ok,
        "episode_count": report.episode_count,
        "step_count": report.step_count,
        "episode_lengths": list(report.episode_lengths),
        "episode_offsets": list(report.episode_offsets),
        "required_datasets": list(report.required_datasets),
        "missing_datasets": list(report.missing_datasets),
        "observation_datasets": _dataset_stats_source_report_payload(
            report,
            ("pixels", "left_pixels", "right_pixels", "proprio", "state"),
        ),
        "action_datasets": _dataset_stats_source_report_payload(report, ("action",)),
        "task_datasets": _dataset_stats_source_report_payload(
            report,
            ("task_id", "plug_type", "port_type", "target_module_name"),
        ),
    }
    if not _payload_exactly_matches_expected_report_projection(
        event.payload,
        expected_payload,
        allowed_extra_keys=("evidence_window",),
    ):
        errors.append("dataset_episode_summary payload must exactly match source_report projection")
    return errors


def _dataset_stats_source_report_payload(
    report: Hdf5DatasetReport,
    dataset_names: tuple[str, ...],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for dataset_name in dataset_names:
        stats = report.datasets.get(dataset_name)
        if stats is not None:
            payload[dataset_name] = stats.to_dict()
    return payload


def _payload_exactly_matches_expected_report_projection(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    allowed_extra_keys: tuple[str, ...],
) -> bool:
    expected_keys = set(expected) | set(allowed_extra_keys)
    if set(payload) != expected_keys:
        return False
    for key, expected_value in expected.items():
        if _thaw_json(payload.get(key)) != expected_value:
            return False
    return True


def _source_event_projection_sha256_binding_errors(
    *,
    artifact: ArtifactRef,
    events: tuple[TimelineEvent, ...],
    label: str,
) -> list[str]:
    actual_sha256 = _canonical_event_projection_sha256(events)
    expected_sha256 = artifact.provenance.get("source_report_event_projection_sha256")
    if not _looks_like_sha256(expected_sha256):
        return [
            f"episode trace {label} source artifact "
            "provenance.source_report_event_projection_sha256 must be set"
        ]
    if expected_sha256 != actual_sha256:
        return [
            f"episode trace {label} events must match source artifact "
            "provenance.source_report_event_projection_sha256"
        ]
    legacy_sha256 = artifact.provenance.get("canonical_event_projection_sha256")
    if legacy_sha256 is not None and legacy_sha256 != actual_sha256:
        return [
            f"episode trace {label} events must match source artifact "
            "provenance.canonical_event_projection_sha256"
        ]
    return []


def _canonical_event_projection_sha256(events: tuple[TimelineEvent, ...]) -> str:
    payload = (
        json.dumps(
            [event.to_dict() for event in events],
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_mapping_sha256(mapping: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(mapping, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _event_source_matches_artifact(source: str, artifact: ArtifactRef) -> bool:
    return source == artifact.path or source == artifact.uri


def _event_source_names_artifact(source: str, artifact: ArtifactRef) -> bool:
    if _event_source_matches_artifact(source, artifact):
        return True
    source_path = _local_source_path(source)
    artifact_paths = tuple(
        path
        for path in (
            _local_source_path(artifact.path),
            _local_source_path(artifact.uri),
        )
        if path is not None
    )
    return source_path is not None and any(source_path == artifact_path for artifact_path in artifact_paths)


def _local_source_path(source: str | None) -> Path | None:
    if source is None:
        return None
    parsed = urlparse(source)
    if parsed.scheme:
        if parsed.scheme != "file" or parsed.netloc not in ("", "localhost") or not parsed.path:
            return None
        path = unquote(parsed.path)
    else:
        path = source
    try:
        return Path(path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def _event_source_under_artifact(source: str, artifact: ArtifactRef) -> bool:
    for root in (artifact.path, artifact.uri):
        if root is None:
            continue
        if _source_under_root(source, root):
            return True
    return False


def _source_under_root(source: str, root: str) -> bool:
    if _contains_dot_segment(source) or _contains_dot_segment(root):
        return False
    source_uri = urlparse(source)
    root_uri = urlparse(root)
    if source_uri.scheme or root_uri.scheme:
        if source_uri.scheme != root_uri.scheme or source_uri.netloc != root_uri.netloc:
            return False
        source_path = PurePosixPath(source_uri.path)
        root_path = PurePosixPath(root_uri.path)
        return source_path == root_path or root_path in source_path.parents
    try:
        source_path = Path(source).expanduser().resolve(strict=False)
        root_path = Path(root).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False
    return source_path == root_path or source_path.is_relative_to(root_path)


def _contains_dot_segment(source: str) -> bool:
    parsed = urlparse(source)
    path = unquote(parsed.path if parsed.scheme else source)
    return any(part in {".", ".."} for part in PurePosixPath(path).parts)
