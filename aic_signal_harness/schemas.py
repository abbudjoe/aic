"""Typed experiment contracts for the backend-neutral AIC signal harness.

This module intentionally uses only the Python standard library. It should be
safe to import in local CI, bootstrap shells, and offline analysis jobs without
ROS, simulator, or model-framework dependencies.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeVar
from urllib.parse import urlparse


SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class SchemaValidationError(ValueError):
    """Raised when a signal harness schema fails closed validation."""

    def __init__(self, errors: str | list[str] | tuple[str, ...]):
        if isinstance(errors, str):
            error_list = (errors,)
        else:
            error_list = tuple(errors)
        self.errors = error_list
        super().__init__("; ".join(error_list))


class _StrEnum(str, Enum):
    """String-valued enum with strict parse helpers."""

    @classmethod
    def parse(cls: type["_EnumT"], value: str | "_EnumT") -> "_EnumT":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise SchemaValidationError(f"{cls.__name__} must be a string")
        try:
            return cls(value)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise SchemaValidationError(
                f"unknown {cls.__name__} {value!r}; expected one of: {allowed}"
            ) from exc


_EnumT = TypeVar("_EnumT", bound=_StrEnum)


class BackendKind(_StrEnum):
    replay_servo = "replay_servo"
    lewm_world_model = "lewm_world_model"
    isaac_rl = "isaac_rl"
    lerobot_act = "lerobot_act"
    open_vla = "open_vla"
    pi0 = "pi0"
    cosmos_reason_critic = "cosmos_reason_critic"
    classical_servo = "classical_servo"
    custom = "custom"


_LIVE_BACKENDS_REQUIRING_POLICY_ARTIFACT = frozenset(
    {
        BackendKind.isaac_rl,
        BackendKind.lerobot_act,
        BackendKind.lewm_world_model,
        BackendKind.open_vla,
        BackendKind.pi0,
    }
)
_RESERVED_CONFIG_ARTIFACT_KEYS = frozenset(
    {
        "checkpoint",
        "checkpoint_file",
        "checkpoint_path",
        "checkpoint_uri",
        "model_checkpoint",
        "model_path",
        "model_uri",
        "policy_artifact",
        "policy_artifact_path",
        "policy_artifact_uri",
        "policy_checkpoint",
        "policy_path",
        "policy_uri",
        "weights_path",
        "weights_uri",
    }
)
_ARTIFACT_CONTEXT_KEYS = frozenset(
    {
        "artifact",
        "artifacts",
        "checkpoint",
        "model",
        "policy",
        "weights",
    }
)
_ARTIFACT_LOCATION_KEYS = frozenset({"file", "path", "uri", "url"})
_ARTIFACT_LOCATION_SUFFIXES = (
    ".ckpt",
    ".h5",
    ".hdf5",
    ".onnx",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
)
_ARTIFACT_REF_KEYS = frozenset({"kind", "path", "provenance", "sha256", "uri"})
_RUNTIME_BOUNDARY_KEYS = frozenset(
    {
        "deterministic",
        "legal_observation_contract",
        "notes",
        "policy_artifact",
        "uses_online_language_model_control",
    }
)
_POLICY_BACKEND_KEYS = frozenset(
    {
        "backend_kind",
        "config",
        "description",
        "leakage_class",
        "name",
        "provenance",
        "runtime_allowed",
        "runtime_boundary",
        "runtime_role",
        "simulator_sources",
        "training_sources",
    }
)
_EXPERIMENT_SPEC_KEYS = frozenset(
    {
        "backend",
        "created_at_utc",
        "expected_artifacts",
        "experiment_id",
        "hypothesis",
        "schema_version",
        "tags",
    }
)


class SimulatorKind(_StrEnum):
    gazebo = "gazebo"
    isaac_lab = "isaac_lab"
    mujoco = "mujoco"
    offline_replay = "offline_replay"
    unknown = "unknown"


class TrainingSourceKind(_StrEnum):
    official_demo = "official_demo"
    ground_truth_demo = "ground_truth_demo"
    isaac_synthetic = "isaac_synthetic"
    mujoco_synthetic = "mujoco_synthetic"
    teleop = "teleop"
    human_annotation = "human_annotation"
    model_annotation = "model_annotation"
    mixed = "mixed"
    none = "none"
    unknown = "unknown"


class RuntimeRole(_StrEnum):
    live_policy = "live_policy"
    offline_labeler = "offline_labeler"
    offline_planner = "offline_planner"
    offline_critic = "offline_critic"
    training_generator = "training_generator"
    evaluation_only = "evaluation_only"


class LeakageClass(_StrEnum):
    legal_policy_input = "legal_policy_input"
    privileged_training_signal = "privileged_training_signal"
    privileged_eval_signal = "privileged_eval_signal"
    post_hoc_label = "post_hoc_label"
    unknown = "unknown"


def utc_now_iso() -> str:
    """Return a stable UTC timestamp for manifests and experiment specs."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_tuple(values: Any, enum_cls: type[_EnumT], field_name: str) -> tuple[_EnumT, ...]:
    if values is None:
        raise SchemaValidationError(f"{field_name} is required")
    if isinstance(values, (str, bytes, bytearray, enum_cls)) or not isinstance(values, (list, tuple)):
        raise SchemaValidationError(f"{field_name} must be a list or tuple")
    raw_values = tuple(values)
    if not raw_values:
        raise SchemaValidationError(f"{field_name} must not be empty")
    return tuple(enum_cls.parse(value) for value in raw_values)


def _jsonable_copy(value: Any, field_name: str) -> tuple[Any, list[str]]:
    if value is None or isinstance(value, (str, bool, int)):
        return value, []
    if isinstance(value, float):
        if not math.isfinite(value):
            return value, [f"{field_name} must be a finite JSON number"]
        return value, []
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        errors: list[str] = []
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                errors.append(f"{field_name} keys must be nonempty strings")
                continue
            copied_item, item_errors = _jsonable_copy(item, f"{field_name}.{key}")
            copied[key] = copied_item
            errors.extend(item_errors)
        return copied, errors
    if isinstance(value, (list, tuple)):
        copied_list: list[Any] = []
        errors = []
        for index, item in enumerate(value):
            copied_item, item_errors = _jsonable_copy(item, f"{field_name}[{index}]")
            copied_list.append(copied_item)
            errors.extend(item_errors)
        return copied_list, errors
    return value, [f"{field_name} must be JSON serializable"]


def _copy_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{field_name} must be a mapping")
    copied, errors = _jsonable_copy(value, field_name)
    if errors:
        raise SchemaValidationError(errors)
    return _freeze_json(copied)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    if isinstance(value, list):
        return [_thaw_json(item) for item in value]
    return value


def _reject_unknown_keys(value: Mapping[str, Any], allowed_keys: frozenset[str], field_name: str) -> None:
    unknown_keys = sorted(repr(key) for key in value if key not in allowed_keys)
    if unknown_keys:
        raise SchemaValidationError(f"{field_name} has unknown fields: {', '.join(unknown_keys)}")


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"{field_name} must be a nonempty string")
    return value


def _optional_nonempty_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_text(value, field_name)


def _validate_uri_shape(value: str, field_name: str) -> str:
    if any(char.isspace() for char in value):
        raise SchemaValidationError(f"{field_name} must not contain whitespace")
    parsed = urlparse(value)
    if not parsed.scheme:
        raise SchemaValidationError(f"{field_name} must include a URI scheme")
    if parsed.scheme == "file" and not parsed.path:
        raise SchemaValidationError(f"{field_name} must include a file path")
    if parsed.scheme != "file" and not parsed.netloc:
        raise SchemaValidationError(f"{field_name} must include a network location")
    return value


def _as_json_sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if value is None:
        raise SchemaValidationError(f"{field_name} must be a list or tuple")
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise SchemaValidationError(f"{field_name} must be a list or tuple")
    return tuple(value)


def _is_reserved_config_artifact_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    if normalized in _RESERVED_CONFIG_ARTIFACT_KEYS:
        return True
    if normalized == "checkpoint" or normalized.endswith(
        ("_checkpoint", "_checkpoint_file", "_checkpoint_path", "_checkpoint_uri")
    ):
        return True
    if normalized.endswith(("_weights_file", "_weights_path", "_weights_uri")):
        return True
    if "artifact" in normalized and ("policy" in normalized or "model" in normalized):
        return True
    if (normalized.startswith("policy_") or normalized.startswith("model_")) and normalized.endswith(
        ("_path", "_uri", "_file")
    ):
        return True
    return False


def _looks_like_artifact_location(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return (
        "://" in normalized
        or normalized.startswith(("/", "./", "../"))
        or normalized.endswith(_ARTIFACT_LOCATION_SUFFIXES)
    )


def _reserved_config_artifact_key_paths(
    value: Any,
    path: str = "backend.config",
    *,
    artifact_context: bool = False,
) -> list[str]:
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, item in value.items():
            if isinstance(key, str):
                normalized = key.strip().lower().replace("-", "_")
                key_path = f"{path}.{key}"
                if _is_reserved_config_artifact_key(key):
                    paths.append(key_path)
                next_artifact_context = artifact_context or normalized in _ARTIFACT_CONTEXT_KEYS
                if artifact_context and normalized in _ARTIFACT_LOCATION_KEYS:
                    paths.append(key_path)
                if normalized in _ARTIFACT_CONTEXT_KEYS and _looks_like_artifact_location(item):
                    paths.append(key_path)
                paths.extend(
                    _reserved_config_artifact_key_paths(
                        item,
                        key_path,
                        artifact_context=next_artifact_context,
                    )
                )
        return paths
    if isinstance(value, (list, tuple)):
        paths = []
        for index, item in enumerate(value):
            paths.extend(
                _reserved_config_artifact_key_paths(
                    item,
                    f"{path}[{index}]",
                    artifact_context=artifact_context,
                )
            )
        return paths
    return []


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to an expected or produced artifact."""

    kind: str
    path: str | None = None
    uri: str | None = None
    sha256: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(self, "kind", _require_nonempty_text(self.kind, "artifact.kind"))
        except SchemaValidationError as exc:
            errors.extend(exc.errors)

        path: str | None = None
        uri: str | None = None
        try:
            path = _optional_nonempty_text(self.path, "artifact.path")
            object.__setattr__(self, "path", path)
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        try:
            uri = _optional_nonempty_text(self.uri, "artifact.uri")
            if uri is not None:
                uri = _validate_uri_shape(uri, "artifact.uri")
            object.__setattr__(self, "uri", uri)
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        if path is None and uri is None:
            errors.append("artifact must set path or uri")

        if self.sha256 is not None and (
            not isinstance(self.sha256, str) or not _SHA256_RE.match(self.sha256)
        ):
            errors.append("artifact.sha256 must be 64 hexadecimal characters")

        try:
            object.__setattr__(
                self,
                "provenance",
                _copy_json_mapping(self.provenance, "artifact.provenance"),
            )
        except SchemaValidationError as exc:
            errors.extend(exc.errors)

        if errors:
            raise SchemaValidationError(errors)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"kind": self.kind}
        if self.path is not None:
            value["path"] = self.path
        if self.uri is not None:
            value["uri"] = self.uri
        if self.sha256 is not None:
            value["sha256"] = self.sha256
        if self.provenance:
            value["provenance"] = _thaw_json(self.provenance)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        if not isinstance(value, Mapping):
            raise SchemaValidationError("artifact must be a mapping")
        _reject_unknown_keys(value, _ARTIFACT_REF_KEYS, "artifact")
        return cls(
            kind=value.get("kind"),
            path=value.get("path"),
            uri=value.get("uri"),
            sha256=value.get("sha256"),
            provenance=value.get("provenance", {}),
        )


@dataclass(frozen=True)
class RuntimeBoundaryProof:
    """Inspectable proof that a live backend is an offline-compiled policy artifact."""

    deterministic: bool
    uses_online_language_model_control: bool
    legal_observation_contract: str
    policy_artifact: ArtifactRef | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.deterministic) is not bool:
            errors.append("runtime_boundary.deterministic must be a boolean")
        if type(self.uses_online_language_model_control) is not bool:
            errors.append(
                "runtime_boundary.uses_online_language_model_control must be a boolean"
            )
        try:
            object.__setattr__(
                self,
                "legal_observation_contract",
                _require_nonempty_text(
                    self.legal_observation_contract,
                    "runtime_boundary.legal_observation_contract",
                ),
            )
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        if self.policy_artifact is not None and not isinstance(self.policy_artifact, ArtifactRef):
            try:
                object.__setattr__(
                    self,
                    "policy_artifact",
                    ArtifactRef.from_dict(self.policy_artifact),
                )
            except SchemaValidationError as exc:
                errors.extend(exc.errors)
        if not isinstance(self.notes, str):
            errors.append("runtime_boundary.notes must be a string")
        if errors:
            raise SchemaValidationError(errors)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "deterministic": self.deterministic,
            "uses_online_language_model_control": self.uses_online_language_model_control,
            "legal_observation_contract": self.legal_observation_contract,
            "notes": self.notes,
        }
        if self.policy_artifact is not None:
            value["policy_artifact"] = self.policy_artifact.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeBoundaryProof":
        if not isinstance(value, Mapping):
            raise SchemaValidationError("runtime_boundary must be a mapping")
        _reject_unknown_keys(value, _RUNTIME_BOUNDARY_KEYS, "runtime_boundary")
        return cls(
            deterministic=value.get("deterministic"),
            uses_online_language_model_control=value.get("uses_online_language_model_control"),
            legal_observation_contract=value.get("legal_observation_contract"),
            policy_artifact=value.get("policy_artifact"),
            notes=value.get("notes", ""),
        )


@dataclass(frozen=True)
class PolicyBackendSpec:
    """Backend screen contract for live and offline AIC candidates."""

    backend_kind: BackendKind
    name: str
    runtime_role: RuntimeRole
    training_sources: tuple[TrainingSourceKind, ...]
    simulator_sources: tuple[SimulatorKind, ...]
    runtime_allowed: bool
    leakage_class: LeakageClass
    runtime_boundary: RuntimeBoundaryProof | None = None
    description: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(self, "backend_kind", BackendKind.parse(self.backend_kind))
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        try:
            object.__setattr__(self, "runtime_role", RuntimeRole.parse(self.runtime_role))
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        try:
            object.__setattr__(
                self,
                "training_sources",
                _as_tuple(self.training_sources, TrainingSourceKind, "training_sources"),
            )
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        try:
            object.__setattr__(
                self,
                "simulator_sources",
                _as_tuple(self.simulator_sources, SimulatorKind, "simulator_sources"),
            )
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        try:
            object.__setattr__(self, "leakage_class", LeakageClass.parse(self.leakage_class))
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        if self.runtime_boundary is not None and not isinstance(
            self.runtime_boundary, RuntimeBoundaryProof
        ):
            try:
                object.__setattr__(
                    self,
                    "runtime_boundary",
                    RuntimeBoundaryProof.from_dict(self.runtime_boundary),
                )
            except SchemaValidationError as exc:
                errors.extend(exc.errors)
        try:
            object.__setattr__(self, "name", _require_nonempty_text(self.name, "backend.name"))
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        if not isinstance(self.runtime_allowed, bool):
            errors.append("runtime_allowed must be a boolean")
        if not isinstance(self.description, str):
            errors.append("description must be a string")
        try:
            object.__setattr__(self, "config", _copy_json_mapping(self.config, "backend.config"))
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        if isinstance(self.config, Mapping):
            reserved_config_keys = sorted(_reserved_config_artifact_key_paths(self.config))
            if reserved_config_keys:
                errors.append(
                    "backend.config must not define policy artifact keys: "
                    + ", ".join(reserved_config_keys)
                )
        try:
            object.__setattr__(
                self,
                "provenance",
                _copy_json_mapping(self.provenance, "backend.provenance"),
            )
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        if self.backend_kind is BackendKind.custom and not self.provenance:
            errors.append("custom backend must set nonempty provenance")

        if not errors:
            errors.extend(self._runtime_boundary_errors())
        if errors:
            raise SchemaValidationError(errors)

    def _runtime_boundary_errors(self) -> list[str]:
        errors: list[str] = []
        if self.runtime_role is RuntimeRole.live_policy:
            if not self.runtime_allowed:
                errors.append("live_policy backend must set runtime_allowed=True")
            if self.leakage_class is not LeakageClass.legal_policy_input:
                errors.append("live_policy backend must use legal_policy_input leakage_class")
            if self.runtime_boundary is None:
                errors.append("live_policy backend must set runtime_boundary proof")
            else:
                if self.runtime_boundary.deterministic is not True:
                    errors.append("live_policy runtime_boundary must be deterministic")
                if self.runtime_boundary.uses_online_language_model_control is not False:
                    errors.append(
                        "live_policy runtime_boundary must not use online language-model control"
                    )
                if (
                    self.backend_kind in _LIVE_BACKENDS_REQUIRING_POLICY_ARTIFACT
                    and self.runtime_boundary.policy_artifact is None
                ):
                    errors.append(
                        f"{self.backend_kind.value} live_policy must set runtime_boundary.policy_artifact"
                    )
        elif self.runtime_allowed:
            errors.append(f"{self.runtime_role.value} backend must not be runtime_allowed")

        if self.runtime_allowed and self.leakage_class is not LeakageClass.legal_policy_input:
            errors.append("runtime_allowed backend must not use privileged or unknown leakage")
        if self.runtime_allowed and TrainingSourceKind.model_annotation in self.training_sources:
            errors.append("model_annotation sources are not allowed in live runtime specs")
        if self.backend_kind is BackendKind.cosmos_reason_critic and (
            self.runtime_allowed or self.runtime_role is RuntimeRole.live_policy
        ):
            errors.append("cosmos_reason_critic is an offline role unless compiled into another backend")
        return errors

    def to_dict(self) -> dict[str, Any]:
        value = {
            "backend_kind": self.backend_kind.value,
            "name": self.name,
            "runtime_role": self.runtime_role.value,
            "training_sources": [source.value for source in self.training_sources],
            "simulator_sources": [source.value for source in self.simulator_sources],
            "runtime_allowed": self.runtime_allowed,
            "leakage_class": self.leakage_class.value,
            "description": self.description,
            "config": _thaw_json(self.config),
            "provenance": _thaw_json(self.provenance),
        }
        if self.runtime_boundary is not None:
            value["runtime_boundary"] = self.runtime_boundary.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicyBackendSpec":
        if not isinstance(value, Mapping):
            raise SchemaValidationError("backend must be a mapping")
        _reject_unknown_keys(value, _POLICY_BACKEND_KEYS, "backend")
        return cls(
            backend_kind=value.get("backend_kind"),
            name=value.get("name"),
            runtime_role=value.get("runtime_role"),
            training_sources=_as_json_sequence(value.get("training_sources"), "training_sources"),
            simulator_sources=_as_json_sequence(value.get("simulator_sources"), "simulator_sources"),
            runtime_allowed=value.get("runtime_allowed"),
            leakage_class=value.get("leakage_class"),
            runtime_boundary=(
                RuntimeBoundaryProof.from_dict(value["runtime_boundary"])
                if "runtime_boundary" in value
                else None
            ),
            description=value.get("description", ""),
            config=value["config"] if "config" in value else {},
            provenance=value.get("provenance", {}),
        )


@dataclass(frozen=True)
class ExperimentSpec:
    """Top-level experiment proposal contract."""

    experiment_id: str
    hypothesis: str
    backend: PolicyBackendSpec
    expected_artifacts: tuple[ArtifactRef, ...] = field(default_factory=tuple)
    created_at_utc: str = field(default_factory=utc_now_iso)
    tags: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        try:
            object.__setattr__(
                self,
                "experiment_id",
                _require_nonempty_text(self.experiment_id, "experiment_id"),
            )
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        try:
            object.__setattr__(self, "hypothesis", _require_nonempty_text(self.hypothesis, "hypothesis"))
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        if not isinstance(self.backend, PolicyBackendSpec):
            try:
                object.__setattr__(self, "backend", PolicyBackendSpec.from_dict(self.backend))
            except SchemaValidationError as exc:
                errors.extend(exc.errors)
        try:
            expected_artifacts = _as_json_sequence(self.expected_artifacts, "expected_artifacts")
            object.__setattr__(
                self,
                "expected_artifacts",
                tuple(
                    item if isinstance(item, ArtifactRef) else ArtifactRef.from_dict(item)
                    for item in expected_artifacts
                ),
            )
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        if not isinstance(self.created_at_utc, str) or not self.created_at_utc.strip():
            errors.append("created_at_utc must be a nonempty string")
        try:
            tags = _as_json_sequence(self.tags, "tags")
            object.__setattr__(self, "tags", tuple(_require_nonempty_text(tag, "tag") for tag in tags))
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        if errors:
            raise SchemaValidationError(errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "backend": self.backend.to_dict(),
            "expected_artifacts": [artifact.to_dict() for artifact in self.expected_artifacts],
            "created_at_utc": self.created_at_utc,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentSpec":
        if not isinstance(value, Mapping):
            raise SchemaValidationError("experiment spec must be a mapping")
        _reject_unknown_keys(value, _EXPERIMENT_SPEC_KEYS, "experiment spec")
        return cls(
            schema_version=value.get("schema_version"),
            experiment_id=value.get("experiment_id"),
            hypothesis=value.get("hypothesis"),
            backend=PolicyBackendSpec.from_dict(value.get("backend")),
            expected_artifacts=value.get("expected_artifacts", ()),
            created_at_utc=value.get("created_at_utc", ""),
            tags=value.get("tags", ()),
        )


def experiment_spec_to_json(spec: ExperimentSpec) -> str:
    """Serialize an experiment spec to deterministic pretty JSON."""

    if not isinstance(spec, ExperimentSpec):
        raise SchemaValidationError("spec must be an ExperimentSpec")
    return json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n"


def experiment_spec_from_json(payload: str | bytes | bytearray) -> ExperimentSpec:
    """Deserialize and validate an experiment spec from JSON."""

    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"invalid JSON: {exc.msg}") from exc
    return ExperimentSpec.from_dict(value)
