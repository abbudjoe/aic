"""Durable outer-loop state machine contracts for AIC policy iteration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, TypeVar, cast

from aic_signal_harness.artifacts import (
    HarnessIOError,
    local_artifact_path,
    read_json,
    sha256_file,
    write_json,
)
from aic_signal_harness.candidate_registry import (
    CANDIDATE_POLICY_REGISTRY_KIND,
    read_candidate_policy_registry_artifact,
)
from aic_signal_harness.schemas import ArtifactRef, SCHEMA_VERSION, SchemaValidationError


OUTER_LOOP_STATE_KIND = "outer_loop_state"

_TRANSITION_KEYS = frozenset(
    {
        "transition_id",
        "recorded_at_utc",
        "from_stage",
        "to_stage",
        "artifacts",
        "reason",
        "actor",
        "notes",
    }
)
_STATE_KEYS = frozenset(
    {
        "schema_version",
        "loop_id",
        "created_at_utc",
        "updated_at_utc",
        "current_stage",
        "candidate_registry",
        "selected_candidate_id",
        "artifacts",
        "transitions",
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


class OuterLoopStage(_StrEnum):
    """Durable restart stage for the train-eval-promote loop."""

    planned = "planned"
    dataset_ready = "dataset_ready"
    training = "training"
    trained = "trained"
    eval_ready = "eval_ready"
    eval_running = "eval_running"
    finalized = "finalized"
    promoted = "promoted"
    rejected = "rejected"
    failed = "failed"


class OuterLoopArtifactRole(_StrEnum):
    """Typed artifact slots consumed by outer-loop state transitions."""

    candidate_registry = "candidate_registry"
    training_dataset_report = "training_dataset_report"
    policy_training_execution_report = "policy_training_execution_report"
    policy_training_report = "policy_training_report"
    policy_checkpoint = "policy_checkpoint"
    eval_launch_plan = "eval_launch_plan"
    eval_run_manifest = "eval_run_manifest"
    ledger_entry = "ledger_entry"
    promotion_decision = "promotion_decision"


_TERMINAL_STAGES = frozenset(
    {
        OuterLoopStage.promoted,
        OuterLoopStage.rejected,
        OuterLoopStage.failed,
    }
)
_ALLOWED_TRANSITIONS: Mapping[OuterLoopStage, frozenset[OuterLoopStage]] = {
    OuterLoopStage.planned: frozenset({OuterLoopStage.dataset_ready, OuterLoopStage.failed}),
    OuterLoopStage.dataset_ready: frozenset({OuterLoopStage.training, OuterLoopStage.failed}),
    OuterLoopStage.training: frozenset({OuterLoopStage.trained, OuterLoopStage.failed}),
    OuterLoopStage.trained: frozenset({OuterLoopStage.eval_ready, OuterLoopStage.failed}),
    OuterLoopStage.eval_ready: frozenset({OuterLoopStage.eval_running, OuterLoopStage.failed}),
    OuterLoopStage.eval_running: frozenset({OuterLoopStage.finalized, OuterLoopStage.failed}),
    OuterLoopStage.finalized: frozenset(
        {OuterLoopStage.promoted, OuterLoopStage.rejected, OuterLoopStage.failed}
    ),
    OuterLoopStage.promoted: frozenset(),
    OuterLoopStage.rejected: frozenset(),
    OuterLoopStage.failed: frozenset(),
}
_TRANSITION_ARTIFACT_ROLES: Mapping[
    tuple[OuterLoopStage, OuterLoopStage],
    frozenset[OuterLoopArtifactRole],
] = {
    (OuterLoopStage.planned, OuterLoopStage.dataset_ready): frozenset(
        {OuterLoopArtifactRole.training_dataset_report}
    ),
    (OuterLoopStage.planned, OuterLoopStage.failed): frozenset(),
    (OuterLoopStage.dataset_ready, OuterLoopStage.training): frozenset(),
    (OuterLoopStage.dataset_ready, OuterLoopStage.failed): frozenset(),
    (OuterLoopStage.training, OuterLoopStage.trained): frozenset(
        {
            OuterLoopArtifactRole.policy_training_execution_report,
            OuterLoopArtifactRole.policy_training_report,
            OuterLoopArtifactRole.policy_checkpoint,
        }
    ),
    (OuterLoopStage.training, OuterLoopStage.failed): frozenset(),
    (OuterLoopStage.trained, OuterLoopStage.eval_ready): frozenset(
        {OuterLoopArtifactRole.eval_launch_plan}
    ),
    (OuterLoopStage.trained, OuterLoopStage.failed): frozenset(),
    (OuterLoopStage.eval_ready, OuterLoopStage.eval_running): frozenset(
        {OuterLoopArtifactRole.eval_run_manifest}
    ),
    (OuterLoopStage.eval_ready, OuterLoopStage.failed): frozenset(),
    (OuterLoopStage.eval_running, OuterLoopStage.finalized): frozenset(
        {OuterLoopArtifactRole.ledger_entry}
    ),
    (OuterLoopStage.eval_running, OuterLoopStage.failed): frozenset(),
    (OuterLoopStage.finalized, OuterLoopStage.promoted): frozenset(
        {OuterLoopArtifactRole.promotion_decision}
    ),
    (OuterLoopStage.finalized, OuterLoopStage.rejected): frozenset(
        {OuterLoopArtifactRole.promotion_decision}
    ),
    (OuterLoopStage.finalized, OuterLoopStage.failed): frozenset(),
}
_ROLE_KINDS: Mapping[OuterLoopArtifactRole, str] = {
    OuterLoopArtifactRole.candidate_registry: CANDIDATE_POLICY_REGISTRY_KIND,
    OuterLoopArtifactRole.training_dataset_report: "training_dataset_report",
    OuterLoopArtifactRole.policy_training_execution_report: "policy_training_execution_report",
    OuterLoopArtifactRole.policy_training_report: "policy_training_report",
    OuterLoopArtifactRole.policy_checkpoint: "policy_checkpoint",
    OuterLoopArtifactRole.eval_launch_plan: "eval_launch_plan",
    OuterLoopArtifactRole.eval_run_manifest: "run_manifest",
    OuterLoopArtifactRole.ledger_entry: "ledger_entry",
    OuterLoopArtifactRole.promotion_decision: "promotion_decision",
}
_STAGE_REQUIRED_ROLES: Mapping[OuterLoopStage, frozenset[OuterLoopArtifactRole]] = {
    OuterLoopStage.planned: frozenset({OuterLoopArtifactRole.candidate_registry}),
    OuterLoopStage.dataset_ready: frozenset(
        {
            OuterLoopArtifactRole.candidate_registry,
            OuterLoopArtifactRole.training_dataset_report,
        }
    ),
    OuterLoopStage.training: frozenset(
        {
            OuterLoopArtifactRole.candidate_registry,
            OuterLoopArtifactRole.training_dataset_report,
        }
    ),
    OuterLoopStage.trained: frozenset(
        {
            OuterLoopArtifactRole.candidate_registry,
            OuterLoopArtifactRole.training_dataset_report,
            OuterLoopArtifactRole.policy_training_execution_report,
            OuterLoopArtifactRole.policy_training_report,
            OuterLoopArtifactRole.policy_checkpoint,
        }
    ),
    OuterLoopStage.eval_ready: frozenset(
        {
            OuterLoopArtifactRole.candidate_registry,
            OuterLoopArtifactRole.training_dataset_report,
            OuterLoopArtifactRole.policy_training_execution_report,
            OuterLoopArtifactRole.policy_training_report,
            OuterLoopArtifactRole.policy_checkpoint,
            OuterLoopArtifactRole.eval_launch_plan,
        }
    ),
    OuterLoopStage.eval_running: frozenset(
        {
            OuterLoopArtifactRole.candidate_registry,
            OuterLoopArtifactRole.training_dataset_report,
            OuterLoopArtifactRole.policy_training_execution_report,
            OuterLoopArtifactRole.policy_training_report,
            OuterLoopArtifactRole.policy_checkpoint,
            OuterLoopArtifactRole.eval_launch_plan,
            OuterLoopArtifactRole.eval_run_manifest,
        }
    ),
    OuterLoopStage.finalized: frozenset(
        {
            OuterLoopArtifactRole.candidate_registry,
            OuterLoopArtifactRole.training_dataset_report,
            OuterLoopArtifactRole.policy_training_execution_report,
            OuterLoopArtifactRole.policy_training_report,
            OuterLoopArtifactRole.policy_checkpoint,
            OuterLoopArtifactRole.eval_launch_plan,
            OuterLoopArtifactRole.eval_run_manifest,
            OuterLoopArtifactRole.ledger_entry,
        }
    ),
    OuterLoopStage.promoted: frozenset(
        {
            OuterLoopArtifactRole.candidate_registry,
            OuterLoopArtifactRole.training_dataset_report,
            OuterLoopArtifactRole.policy_training_execution_report,
            OuterLoopArtifactRole.policy_training_report,
            OuterLoopArtifactRole.policy_checkpoint,
            OuterLoopArtifactRole.eval_launch_plan,
            OuterLoopArtifactRole.eval_run_manifest,
            OuterLoopArtifactRole.ledger_entry,
            OuterLoopArtifactRole.promotion_decision,
        }
    ),
    OuterLoopStage.rejected: frozenset(
        {
            OuterLoopArtifactRole.candidate_registry,
            OuterLoopArtifactRole.training_dataset_report,
            OuterLoopArtifactRole.policy_training_execution_report,
            OuterLoopArtifactRole.policy_training_report,
            OuterLoopArtifactRole.policy_checkpoint,
            OuterLoopArtifactRole.eval_launch_plan,
            OuterLoopArtifactRole.eval_run_manifest,
            OuterLoopArtifactRole.ledger_entry,
            OuterLoopArtifactRole.promotion_decision,
        }
    ),
    OuterLoopStage.failed: frozenset({OuterLoopArtifactRole.candidate_registry}),
}


@dataclass(frozen=True)
class OuterLoopTransition:
    """One append-only state transition in the train-eval-promote loop."""

    recorded_at_utc: str
    from_stage: OuterLoopStage
    to_stage: OuterLoopStage
    artifacts: Mapping[OuterLoopArtifactRole, ArtifactRef] = field(default_factory=dict)
    reason: str = ""
    actor: str = "codex"
    notes: tuple[str, ...] = field(default_factory=tuple)
    transition_id: str | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(
                self,
                "recorded_at_utc",
                _require_utc_timestamp(self.recorded_at_utc, "outer loop transition.recorded_at_utc"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "from_stage", OuterLoopStage.parse(self.from_stage))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "to_stage", OuterLoopStage.parse(self.to_stage))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "reason",
                _require_nonempty_text(self.reason, "outer loop transition.reason"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "actor",
                _require_nonempty_text(self.actor, "outer loop transition.actor"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "artifacts",
                _artifact_mapping(self.artifacts, "outer loop transition.artifacts"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "notes",
                _as_text_sequence(self.notes, "outer loop transition.notes"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not errors:
            if self.from_stage in _TERMINAL_STAGES:
                errors.append(f"terminal outer loop stage {self.from_stage.value} cannot advance")
            allowed = _ALLOWED_TRANSITIONS[self.from_stage]
            if self.to_stage not in allowed:
                errors.append(
                    f"outer loop transition {self.from_stage.value}->{self.to_stage.value} is not allowed"
                )
            errors.extend(_transition_artifact_role_errors(self))
            fingerprint = outer_loop_transition_fingerprint_sha256(self)
            expected_id = outer_loop_transition_id_from_fingerprint(fingerprint)
            if self.transition_id is None:
                object.__setattr__(self, "transition_id", expected_id)
            elif self.transition_id != expected_id:
                errors.append("outer loop transition.transition_id must be deterministic from content")
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "recorded_at_utc": self.recorded_at_utc,
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "artifacts": {
                role.value: artifact.to_dict()
                for role, artifact in sorted(self.artifacts.items(), key=lambda item: item[0].value)
            },
            "reason": self.reason,
            "actor": self.actor,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OuterLoopTransition":
        if not isinstance(value, Mapping):
            raise HarnessIOError("outer loop transition must be a mapping")
        _reject_unknown_keys(value, _TRANSITION_KEYS, "outer loop transition")
        _require_keys(
            value,
            _TRANSITION_KEYS - {"artifacts", "notes"},
            "outer loop transition",
        )
        if value.get("transition_id") is None:
            raise HarnessIOError("outer loop transition.transition_id must be set")
        return cls(
            transition_id=cast(str | None, value.get("transition_id")),
            recorded_at_utc=cast(Any, value.get("recorded_at_utc")),
            from_stage=cast(Any, value.get("from_stage")),
            to_stage=cast(Any, value.get("to_stage")),
            artifacts=cast(Any, value.get("artifacts", {})),
            reason=cast(Any, value.get("reason")),
            actor=cast(Any, value.get("actor", "codex")),
            notes=cast(Any, value.get("notes", ())),
        )


@dataclass(frozen=True)
class OuterLoopRunState:
    """Durable state snapshot reconstructed from append-only transitions."""

    loop_id: str
    created_at_utc: str
    updated_at_utc: str
    current_stage: OuterLoopStage
    candidate_registry: ArtifactRef
    selected_candidate_id: str
    artifacts: Mapping[OuterLoopArtifactRole, ArtifactRef]
    transitions: tuple[OuterLoopTransition, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        try:
            object.__setattr__(self, "loop_id", _require_nonempty_text(self.loop_id, "outer loop state.loop_id"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("created_at_utc", "updated_at_utc"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_utc_timestamp(
                        getattr(self, field_name),
                        f"outer loop state.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(self, "current_stage", OuterLoopStage.parse(self.current_stage))
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not isinstance(self.candidate_registry, ArtifactRef):
            try:
                object.__setattr__(
                    self,
                    "candidate_registry",
                    ArtifactRef.from_dict(self.candidate_registry),
                )
            except SchemaValidationError as exc:
                errors.extend(exc.errors)
        try:
            object.__setattr__(
                self,
                "selected_candidate_id",
                _require_nonempty_text(
                    self.selected_candidate_id,
                    "outer loop state.selected_candidate_id",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "artifacts",
                _artifact_mapping(self.artifacts, "outer loop state.artifacts"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            transitions = tuple(
                transition
                if isinstance(transition, OuterLoopTransition)
                else OuterLoopTransition.from_dict(transition)
                for transition in _as_sequence(self.transitions, "outer loop state.transitions")
            )
            object.__setattr__(self, "transitions", transitions)
        except HarnessIOError as exc:
            errors.append(str(exc))
            transitions = tuple()
        try:
            object.__setattr__(self, "notes", _as_text_sequence(self.notes, "outer loop state.notes"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not errors:
            errors.extend(_state_consistency_errors(self))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "loop_id": self.loop_id,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "current_stage": self.current_stage.value,
            "candidate_registry": self.candidate_registry.to_dict(),
            "selected_candidate_id": self.selected_candidate_id,
            "artifacts": {
                role.value: artifact.to_dict()
                for role, artifact in sorted(self.artifacts.items(), key=lambda item: item[0].value)
            },
            "transitions": [transition.to_dict() for transition in self.transitions],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OuterLoopRunState":
        if not isinstance(value, Mapping):
            raise HarnessIOError("outer loop state must be a mapping")
        _reject_unknown_keys(value, _STATE_KEYS, "outer loop state")
        _require_keys(value, _STATE_KEYS - {"notes"}, "outer loop state")
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            loop_id=cast(Any, value.get("loop_id")),
            created_at_utc=cast(Any, value.get("created_at_utc")),
            updated_at_utc=cast(Any, value.get("updated_at_utc")),
            current_stage=cast(Any, value.get("current_stage")),
            candidate_registry=cast(Any, value.get("candidate_registry")),
            selected_candidate_id=cast(Any, value.get("selected_candidate_id")),
            artifacts=cast(Any, value.get("artifacts")),
            transitions=cast(Any, value.get("transitions")),
            notes=cast(Any, value.get("notes", ())),
        )


def initialize_outer_loop_state(
    *,
    loop_id: str,
    candidate_registry: ArtifactRef | Mapping[str, Any],
    created_at_utc: str,
    notes: tuple[str, ...] = (),
) -> OuterLoopRunState:
    """Create a planned state bound to a selected candidate registry artifact."""

    typed_registry_artifact = (
        candidate_registry
        if isinstance(candidate_registry, ArtifactRef)
        else ArtifactRef.from_dict(candidate_registry)
    )
    registry = read_candidate_policy_registry_artifact(typed_registry_artifact)
    if registry.selected_candidate_id is None:
        raise HarnessIOError("outer loop state candidate registry must select a candidate")
    return OuterLoopRunState(
        loop_id=loop_id,
        created_at_utc=created_at_utc,
        updated_at_utc=created_at_utc,
        current_stage=OuterLoopStage.planned,
        candidate_registry=typed_registry_artifact,
        selected_candidate_id=registry.selected_candidate_id,
        artifacts={OuterLoopArtifactRole.candidate_registry: typed_registry_artifact},
        transitions=(),
        notes=notes,
    )


def advance_outer_loop_state(
    state: OuterLoopRunState | Mapping[str, Any],
    *,
    to_stage: OuterLoopStage | str,
    recorded_at_utc: str,
    reason: str,
    artifacts: Mapping[OuterLoopArtifactRole | str, ArtifactRef | Mapping[str, Any]] | None = None,
    actor: str = "codex",
    notes: tuple[str, ...] = (),
) -> OuterLoopRunState:
    """Append one legal transition and return the resulting durable state."""

    typed_state = state if isinstance(state, OuterLoopRunState) else OuterLoopRunState.from_dict(state)
    transition_artifacts = _artifact_mapping(
        {} if artifacts is None else artifacts,
        "outer loop transition.artifacts",
    )
    transition = OuterLoopTransition(
        recorded_at_utc=recorded_at_utc,
        from_stage=typed_state.current_stage,
        to_stage=OuterLoopStage.parse(to_stage),
        artifacts=transition_artifacts,
        reason=reason,
        actor=actor,
        notes=notes,
    )
    next_artifacts = dict(typed_state.artifacts)
    next_artifacts.update(transition.artifacts)
    return OuterLoopRunState(
        loop_id=typed_state.loop_id,
        created_at_utc=typed_state.created_at_utc,
        updated_at_utc=transition.recorded_at_utc,
        current_stage=transition.to_stage,
        candidate_registry=typed_state.candidate_registry,
        selected_candidate_id=typed_state.selected_candidate_id,
        artifacts=next_artifacts,
        transitions=typed_state.transitions + (transition,),
        notes=typed_state.notes,
    )


def outer_loop_transition_fingerprint_sha256(transition: OuterLoopTransition) -> str:
    """Return a stable fingerprint for transition identity fields."""

    payload = {
        "recorded_at_utc": transition.recorded_at_utc,
        "from_stage": transition.from_stage.value,
        "to_stage": transition.to_stage.value,
        "artifacts": {
            role.value: artifact.to_dict()
            for role, artifact in sorted(transition.artifacts.items(), key=lambda item: item[0].value)
        },
        "reason": transition.reason,
        "actor": transition.actor,
        "notes": list(transition.notes),
    }
    return _stable_json_sha256(payload)


def outer_loop_transition_id_from_fingerprint(fingerprint_sha256: str) -> str:
    return "oltr_" + _require_sha256(fingerprint_sha256, "outer loop transition fingerprint")[:20]


def write_outer_loop_state(
    path: str | Path,
    state: OuterLoopRunState | Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> ArtifactRef:
    """Write one durable state snapshot and return its artifact reference."""

    typed_state = state if isinstance(state, OuterLoopRunState) else OuterLoopRunState.from_dict(state)
    output_path = Path(path).expanduser().resolve(strict=False)
    write_json(output_path, typed_state.to_dict(), overwrite=overwrite)
    return ArtifactRef(
        kind=OUTER_LOOP_STATE_KIND,
        path=str(output_path),
        sha256=sha256_file(output_path),
        provenance={
            "producer": "aic_signal_harness.outer_loop_state",
            "derivation": "write_outer_loop_state",
            "loop_id": typed_state.loop_id,
            "current_stage": typed_state.current_stage.value,
            "selected_candidate_id": typed_state.selected_candidate_id,
        },
    )


def read_outer_loop_state_artifact(artifact: ArtifactRef | Mapping[str, Any]) -> OuterLoopRunState:
    """Read and validate a local outer-loop state artifact."""

    typed_artifact = artifact if isinstance(artifact, ArtifactRef) else ArtifactRef.from_dict(artifact)
    errors: list[str] = []
    if typed_artifact.kind != OUTER_LOOP_STATE_KIND:
        errors.append(f"outer loop state artifact.kind must be {OUTER_LOOP_STATE_KIND!r}")
    if typed_artifact.sha256 is None:
        errors.append("outer loop state artifact.sha256 must be set")
    for provenance_key in (
        "producer",
        "derivation",
        "loop_id",
        "current_stage",
        "selected_candidate_id",
    ):
        if provenance_key not in typed_artifact.provenance:
            errors.append(f"outer loop state artifact provenance must set {provenance_key}")
        else:
            try:
                _require_nonempty_text(
                    typed_artifact.provenance.get(provenance_key),
                    f"outer loop state artifact provenance.{provenance_key}",
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
    if typed_artifact.provenance.get("producer") != "aic_signal_harness.outer_loop_state":
        errors.append(
            "outer loop state artifact provenance producer must be "
            "'aic_signal_harness.outer_loop_state'"
        )
    if typed_artifact.provenance.get("derivation") != "write_outer_loop_state":
        errors.append(
            "outer loop state artifact provenance derivation must be 'write_outer_loop_state'"
        )
    try:
        state_path = local_artifact_path(
            path=typed_artifact.path,
            uri=typed_artifact.uri,
            field_name="outer loop state artifact",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        state_path = None
    if state_path is None:
        errors.append("outer loop state artifact must be local byte-verifiable")
    elif not state_path.exists():
        errors.append("outer loop state artifact.path must exist")
    elif not state_path.is_file():
        errors.append("outer loop state artifact.path must be a file")
    elif typed_artifact.sha256 is not None and sha256_file(state_path) != typed_artifact.sha256:
        errors.append("outer loop state artifact.sha256 must match path")
    if errors:
        raise HarnessIOError("; ".join(errors))
    assert state_path is not None
    state = OuterLoopRunState.from_dict(read_json(state_path))
    if typed_artifact.provenance.get("loop_id") != state.loop_id:
        raise HarnessIOError("outer loop state artifact provenance loop_id must match")
    if typed_artifact.provenance.get("current_stage") != state.current_stage.value:
        raise HarnessIOError("outer loop state artifact provenance current_stage must match")
    if typed_artifact.provenance.get("selected_candidate_id") != state.selected_candidate_id:
        raise HarnessIOError(
            "outer loop state artifact provenance selected_candidate_id must match"
        )
    return state


def _state_consistency_errors(state: OuterLoopRunState) -> list[str]:
    errors: list[str] = []
    try:
        registry = read_candidate_policy_registry_artifact(state.candidate_registry)
    except HarnessIOError as exc:
        errors.append("outer loop state.candidate_registry must validate: " + str(exc))
        registry = None
    if registry is not None and registry.selected_candidate_id != state.selected_candidate_id:
        errors.append("outer loop state.selected_candidate_id must match candidate registry")
    if state.artifacts.get(OuterLoopArtifactRole.candidate_registry) != state.candidate_registry:
        errors.append("outer loop state.artifacts.candidate_registry must match candidate_registry")

    replay_stage = OuterLoopStage.planned
    replay_artifacts: dict[OuterLoopArtifactRole, ArtifactRef] = {
        OuterLoopArtifactRole.candidate_registry: state.candidate_registry
    }
    transition_ids: set[str] = set()
    previous_timestamp = state.created_at_utc
    for transition in state.transitions:
        if transition.transition_id in transition_ids:
            errors.append(
                f"outer loop state.transitions must not repeat transition_id {transition.transition_id}"
            )
        assert transition.transition_id is not None
        transition_ids.add(transition.transition_id)
        if transition.from_stage is not replay_stage:
            errors.append(
                "outer loop transition.from_stage must match replayed current stage: "
                f"expected {replay_stage.value}, got {transition.from_stage.value}"
            )
        if _timestamp_key(transition.recorded_at_utc) < _timestamp_key(previous_timestamp):
            errors.append("outer loop transitions must be chronological")
        previous_timestamp = transition.recorded_at_utc
        replay_artifacts.update(transition.artifacts)
        missing_roles = sorted(
            role.value
            for role in _STAGE_REQUIRED_ROLES[transition.to_stage]
            if role not in replay_artifacts
        )
        if missing_roles:
            errors.append(
                f"outer loop stage {transition.to_stage.value} missing required artifacts: "
                + ", ".join(missing_roles)
            )
        replay_stage = transition.to_stage

    if state.current_stage is not replay_stage:
        errors.append(
            "outer loop state.current_stage must equal replayed stage: "
            f"expected {replay_stage.value}, got {state.current_stage.value}"
        )
    missing_current_roles = sorted(
        role.value for role in _STAGE_REQUIRED_ROLES[state.current_stage] if role not in replay_artifacts
    )
    if missing_current_roles:
        errors.append(
            f"outer loop current stage {state.current_stage.value} missing required artifacts: "
            + ", ".join(missing_current_roles)
        )
    if dict(state.artifacts) != replay_artifacts:
        errors.append("outer loop state.artifacts must equal replayed transition artifacts")
    if state.transitions and state.updated_at_utc != state.transitions[-1].recorded_at_utc:
        errors.append("outer loop state.updated_at_utc must match latest transition")
    if not state.transitions and state.updated_at_utc != state.created_at_utc:
        errors.append("outer loop planned state.updated_at_utc must match created_at_utc")
    return errors


def _artifact_mapping(value: Any, field_name: str) -> Mapping[OuterLoopArtifactRole, ArtifactRef]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    parsed: dict[OuterLoopArtifactRole, ArtifactRef] = {}
    errors: list[str] = []
    for raw_role, raw_artifact in value.items():
        try:
            role = OuterLoopArtifactRole.parse(raw_role)
        except HarnessIOError as exc:
            errors.append(str(exc))
            continue
        try:
            artifact = raw_artifact if isinstance(raw_artifact, ArtifactRef) else ArtifactRef.from_dict(raw_artifact)
            errors.extend(_artifact_ref_errors(role, artifact))
            parsed[role] = artifact
        except (HarnessIOError, SchemaValidationError) as exc:
            errors.append(str(exc))
    if errors:
        raise HarnessIOError("; ".join(errors))
    return MappingProxyType(parsed)


def _transition_artifact_role_errors(transition: OuterLoopTransition) -> list[str]:
    errors: list[str] = []
    allowed_roles = _TRANSITION_ARTIFACT_ROLES.get(
        (transition.from_stage, transition.to_stage),
        frozenset(),
    )
    disallowed_roles = sorted(
        role.value for role in transition.artifacts if role not in allowed_roles
    )
    if disallowed_roles:
        errors.append(
            f"outer loop transition {transition.from_stage.value}->{transition.to_stage.value} "
            "may not introduce artifact roles: "
            + ", ".join(disallowed_roles)
        )
    return errors


def _artifact_ref_errors(role: OuterLoopArtifactRole, artifact: ArtifactRef) -> list[str]:
    errors: list[str] = []
    expected_kind = _ROLE_KINDS[role]
    if artifact.kind != expected_kind:
        errors.append(f"outer loop artifact {role.value}.kind must be {expected_kind!r}")
    if artifact.sha256 is None:
        errors.append(f"outer loop artifact {role.value}.sha256 must be set")
    for provenance_key in ("producer", "derivation"):
        if provenance_key not in artifact.provenance:
            errors.append(f"outer loop artifact {role.value} provenance must set {provenance_key}")
        else:
            try:
                _require_nonempty_text(
                    artifact.provenance.get(provenance_key),
                    f"outer loop artifact {role.value} provenance.{provenance_key}",
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
    try:
        artifact_path = local_artifact_path(
            path=artifact.path,
            uri=artifact.uri,
            field_name=f"outer loop artifact {role.value}",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        artifact_path = None
    if artifact_path is not None:
        if not artifact_path.is_absolute():
            errors.append(f"outer loop artifact {role.value}.path must be absolute")
        elif not artifact_path.exists():
            errors.append(f"outer loop artifact {role.value}.path must exist")
        elif not artifact_path.is_file():
            errors.append(f"outer loop artifact {role.value}.path must be a file")
        elif artifact.sha256 is not None and sha256_file(artifact_path) != artifact.sha256:
            errors.append(f"outer loop artifact {role.value}.sha256 must match path")
    else:
        errors.append(f"outer loop artifact {role.value} must be local byte-verifiable")
    return errors


def _require_keys(
    value: Mapping[str, Any],
    required_keys: frozenset[str],
    field_name: str,
) -> None:
    missing = sorted(key for key in required_keys if key not in value)
    if missing:
        raise HarnessIOError(f"{field_name} missing required fields: {', '.join(missing)}")


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed_keys: frozenset[str],
    field_name: str,
) -> None:
    unknown_keys = sorted(repr(key) for key in value if key not in allowed_keys)
    if unknown_keys:
        raise HarnessIOError(f"{field_name} has unknown fields: {', '.join(unknown_keys)}")


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessIOError(f"{field_name} must be a nonempty string")
    return value.strip()


def _require_utc_timestamp(value: Any, field_name: str) -> str:
    timestamp = _require_nonempty_text(value, field_name)
    try:
        parsed = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise HarnessIOError(f"{field_name} must be a UTC timestamp ending in Z") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != timestamp:
        raise HarnessIOError(f"{field_name} must be a canonical UTC timestamp")
    return timestamp


def _timestamp_key(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def _as_sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if value is None:
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    return tuple(value)


def _as_text_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    items = _as_sequence(value, field_name)
    return tuple(_require_nonempty_text(item, f"{field_name}[{index}]") for index, item in enumerate(items))


def _require_sha256(value: Any, field_name: str) -> str:
    text = _require_nonempty_text(value, field_name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise HarnessIOError(f"{field_name} must be a lowercase sha256 hex digest")
    return text


def _stable_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
