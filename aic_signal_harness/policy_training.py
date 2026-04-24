"""Policy training run contracts for offline AIC policy artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeVar, cast
from urllib.parse import urlparse

from aic_signal_harness.artifacts import HarnessIOError, local_artifact_path, read_json, sha256_file
from aic_signal_harness.schemas import (
    ArtifactRef,
    BackendKind,
    PolicyBackendSpec,
    RuntimeRole,
    SchemaValidationError,
)
from aic_signal_harness.training_dataset import TrainingDatasetReport


POLICY_TRAINING_SCHEMA_VERSION = 1


_TRAINER_INVOCATION_KEYS = frozenset(
    {
        "trainer_id",
        "trainer_name",
        "backend_kind",
        "command",
        "config",
        "environment",
        "working_directory",
        "container_image",
        "seed",
        "offline_only",
        "runtime_allowed",
        "uses_online_language_model_control",
    }
)
_POLICY_TRAINING_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "generated_at_utc",
        "source_dataset_run_id",
        "source_training_dataset_report",
        "trainer",
        "trainer_fingerprint_sha256",
        "status",
        "policy_artifact",
        "candidate_policy",
        "metrics",
        "offline_only",
        "runtime_allowed",
        "autonomous_launch_allowed",
        "ok",
        "errors",
        "notes",
    }
)
_POLICY_ARTIFACT_KIND = "policy_checkpoint"
_TRAINING_DATASET_REPORT_KIND = "training_dataset_report"
_DERIVATION = "record_policy_training_run"
_RESERVED_ARTIFACT_KEY_PREFIXES = (
    "checkpoint",
    "dataset",
    "model",
    "policy",
    "weights",
)
_RESERVED_ARTIFACT_KEY_SUFFIXES = (
    "_checkpoint",
    "_checkpoint_file",
    "_checkpoint_path",
    "_checkpoint_uri",
    "_dataset",
    "_dataset_file",
    "_dataset_path",
    "_dataset_uri",
    "_model_file",
    "_model_path",
    "_model_uri",
    "_policy",
    "_policy_file",
    "_policy_path",
    "_policy_uri",
    "_weights_file",
    "_weights_path",
    "_weights_uri",
)
_ARTIFACT_VALUE_SUFFIXES = (
    ".ckpt",
    ".h5",
    ".hdf5",
    ".onnx",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
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


class PolicyTrainingStatus(_StrEnum):
    """Observed training-run state represented by this first contract slice."""

    completed = "completed"


@dataclass(frozen=True)
class TrainerInvocation:
    """Offline trainer invocation metadata with artifacts kept out of config."""

    trainer_id: str
    trainer_name: str
    backend_kind: BackendKind
    command: tuple[str, ...]
    config: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, str] = field(default_factory=dict)
    working_directory: str | None = None
    container_image: str | None = None
    seed: int | None = None
    offline_only: bool = True
    runtime_allowed: bool = False
    uses_online_language_model_control: bool = False

    def __post_init__(self) -> None:
        errors: list[str] = []
        for field_name in ("trainer_id", "trainer_name"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"trainer invocation.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(self, "backend_kind", BackendKind.parse(self.backend_kind))
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        try:
            object.__setattr__(
                self,
                "command",
                _as_text_sequence(
                    self.command,
                    "trainer invocation.command",
                    allow_empty=False,
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "config",
                _copy_json_mapping(self.config, "trainer invocation.config"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "environment",
                _environment_mapping(self.environment, "trainer invocation.environment"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("working_directory", "container_image"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _optional_nonempty_text(
                        getattr(self, field_name),
                        f"trainer invocation.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "seed",
                _optional_nonnegative_int(self.seed, "trainer invocation.seed"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if self.offline_only is not True:
            errors.append("trainer invocation.offline_only must be true")
        if self.runtime_allowed is not False:
            errors.append("trainer invocation.runtime_allowed must be false")
        if self.uses_online_language_model_control is not False:
            errors.append(
                "trainer invocation.uses_online_language_model_control must be false"
            )
        if isinstance(self.config, Mapping):
            reserved_config_paths = _reserved_artifact_key_paths(
                self.config,
                path="trainer invocation.config",
            )
            if reserved_config_paths:
                errors.append(
                    "trainer invocation.config must not hide artifact references: "
                    + ", ".join(reserved_config_paths)
                )
        if isinstance(self.environment, Mapping):
            reserved_env_keys = _reserved_environment_artifact_keys(self.environment)
            if reserved_env_keys:
                errors.append(
                    "trainer invocation.environment must not hide artifact references: "
                    + ", ".join(reserved_env_keys)
                )
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "trainer_id": self.trainer_id,
            "trainer_name": self.trainer_name,
            "backend_kind": self.backend_kind.value,
            "command": list(self.command),
            "config": _thaw_json(self.config),
            "environment": dict(self.environment),
            "offline_only": self.offline_only,
            "runtime_allowed": self.runtime_allowed,
            "uses_online_language_model_control": self.uses_online_language_model_control,
        }
        if self.working_directory is not None:
            value["working_directory"] = self.working_directory
        if self.container_image is not None:
            value["container_image"] = self.container_image
        if self.seed is not None:
            value["seed"] = self.seed
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainerInvocation":
        if not isinstance(value, Mapping):
            raise HarnessIOError("trainer invocation must be a mapping")
        _reject_unknown_keys(value, _TRAINER_INVOCATION_KEYS, "trainer invocation")
        _require_keys(
            value,
            _TRAINER_INVOCATION_KEYS
            - {
                "config",
                "container_image",
                "environment",
                "seed",
                "working_directory",
            },
            "trainer invocation",
        )
        return cls(
            trainer_id=cast(Any, value.get("trainer_id")),
            trainer_name=cast(Any, value.get("trainer_name")),
            backend_kind=cast(Any, value.get("backend_kind")),
            command=cast(Any, value.get("command")),
            config=cast(Any, value.get("config", {})),
            environment=cast(Any, value.get("environment", {})),
            working_directory=cast(Any, value.get("working_directory")),
            container_image=cast(Any, value.get("container_image")),
            seed=cast(Any, value.get("seed")),
            offline_only=cast(bool, value.get("offline_only")),
            runtime_allowed=cast(bool, value.get("runtime_allowed")),
            uses_online_language_model_control=cast(
                bool,
                value.get("uses_online_language_model_control"),
            ),
        )


@dataclass(frozen=True)
class PolicyTrainingRunReport:
    """Completed offline policy-training provenance for one policy artifact."""

    run_id: str
    generated_at_utc: str
    source_dataset_run_id: str
    source_training_dataset_report: ArtifactRef
    trainer: TrainerInvocation
    trainer_fingerprint_sha256: str
    status: PolicyTrainingStatus
    policy_artifact: ArtifactRef
    candidate_policy: PolicyBackendSpec
    metrics: Mapping[str, float] = field(default_factory=dict)
    offline_only: bool = True
    runtime_allowed: bool = False
    autonomous_launch_allowed: bool = False
    ok: bool = True
    errors: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = POLICY_TRAINING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != POLICY_TRAINING_SCHEMA_VERSION:
            errors.append(f"schema_version must be {POLICY_TRAINING_SCHEMA_VERSION}")
        for field_name in ("run_id", "generated_at_utc", "source_dataset_run_id"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"policy training report.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        if not isinstance(self.source_training_dataset_report, ArtifactRef):
            try:
                object.__setattr__(
                    self,
                    "source_training_dataset_report",
                    ArtifactRef.from_dict(self.source_training_dataset_report),
                )
            except SchemaValidationError as exc:
                errors.extend(exc.errors)
        if not isinstance(self.trainer, TrainerInvocation):
            try:
                object.__setattr__(self, "trainer", TrainerInvocation.from_dict(self.trainer))
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "trainer_fingerprint_sha256",
                _require_sha256(
                    self.trainer_fingerprint_sha256,
                    "policy training report.trainer_fingerprint_sha256",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "status", PolicyTrainingStatus.parse(self.status))
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not isinstance(self.policy_artifact, ArtifactRef):
            try:
                object.__setattr__(
                    self,
                    "policy_artifact",
                    ArtifactRef.from_dict(self.policy_artifact),
                )
            except SchemaValidationError as exc:
                errors.extend(exc.errors)
        if not isinstance(self.candidate_policy, PolicyBackendSpec):
            try:
                object.__setattr__(
                    self,
                    "candidate_policy",
                    PolicyBackendSpec.from_dict(self.candidate_policy),
                )
            except SchemaValidationError as exc:
                errors.extend(exc.errors)
        try:
            object.__setattr__(
                self,
                "metrics",
                _metric_mapping(self.metrics, "policy training report.metrics"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("errors", "notes"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _as_text_sequence(
                        getattr(self, field_name),
                        f"policy training report.{field_name}",
                        allow_empty=True,
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        if self.offline_only is not True:
            errors.append("policy training report.offline_only must be true")
        if self.runtime_allowed is not False:
            errors.append("policy training report.runtime_allowed must be false")
        if self.autonomous_launch_allowed is not False:
            errors.append("policy training report.autonomous_launch_allowed must be false")
        if self.ok is not True:
            errors.append("policy training report.ok must be true for completed reports")
        if self.errors:
            errors.append("policy training report.errors must be empty for completed reports")

        if not errors:
            assert isinstance(self.trainer, TrainerInvocation)
            assert isinstance(self.policy_artifact, ArtifactRef)
            assert isinstance(self.candidate_policy, PolicyBackendSpec)
            errors.extend(_completed_report_consistency_errors(self))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generated_at_utc": self.generated_at_utc,
            "source_dataset_run_id": self.source_dataset_run_id,
            "source_training_dataset_report": self.source_training_dataset_report.to_dict(),
            "trainer": self.trainer.to_dict(),
            "trainer_fingerprint_sha256": self.trainer_fingerprint_sha256,
            "status": self.status.value,
            "policy_artifact": self.policy_artifact.to_dict(),
            "candidate_policy": self.candidate_policy.to_dict(),
            "metrics": dict(self.metrics),
            "offline_only": self.offline_only,
            "runtime_allowed": self.runtime_allowed,
            "autonomous_launch_allowed": self.autonomous_launch_allowed,
            "ok": self.ok,
            "errors": list(self.errors),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicyTrainingRunReport":
        if not isinstance(value, Mapping):
            raise HarnessIOError("policy training report must be a mapping")
        _reject_unknown_keys(value, _POLICY_TRAINING_REPORT_KEYS, "policy training report")
        _require_keys(value, _POLICY_TRAINING_REPORT_KEYS, "policy training report")
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            run_id=cast(Any, value.get("run_id")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            source_dataset_run_id=cast(Any, value.get("source_dataset_run_id")),
            source_training_dataset_report=cast(Any, value.get("source_training_dataset_report")),
            trainer=cast(Any, value.get("trainer")),
            trainer_fingerprint_sha256=cast(Any, value.get("trainer_fingerprint_sha256")),
            status=cast(Any, value.get("status")),
            policy_artifact=cast(Any, value.get("policy_artifact")),
            candidate_policy=cast(Any, value.get("candidate_policy")),
            metrics=cast(Any, value.get("metrics")),
            offline_only=cast(bool, value.get("offline_only")),
            runtime_allowed=cast(bool, value.get("runtime_allowed")),
            autonomous_launch_allowed=cast(bool, value.get("autonomous_launch_allowed")),
            ok=cast(bool, value.get("ok")),
            errors=cast(Any, value.get("errors")),
            notes=cast(Any, value.get("notes")),
        )


def trainer_invocation_fingerprint_sha256(trainer: TrainerInvocation) -> str:
    """Return the stable fingerprint for a trainer invocation."""

    payload = trainer.to_dict()
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _completed_report_consistency_errors(report: PolicyTrainingRunReport) -> list[str]:
    errors: list[str] = []
    if report.status is not PolicyTrainingStatus.completed:
        errors.append("policy training report.status must be completed")
    expected_trainer_fingerprint = trainer_invocation_fingerprint_sha256(report.trainer)
    if report.trainer_fingerprint_sha256 != expected_trainer_fingerprint:
        errors.append("policy training report.trainer_fingerprint_sha256 must match trainer")
    dataset_report = _source_training_dataset_report(
        report.source_training_dataset_report,
        source_dataset_run_id=report.source_dataset_run_id,
    )
    if dataset_report is not None:
        if not dataset_report.ok:
            errors.append("source training dataset report must be ok")
        if dataset_report.example_count < 1:
            errors.append("source training dataset report.example_count must be >= 1")
    if report.trainer.backend_kind is not report.candidate_policy.backend_kind:
        errors.append("policy training report.trainer.backend_kind must match candidate_policy.backend_kind")
    errors.extend(
        _policy_artifact_errors(
            report.policy_artifact,
            run_id=report.run_id,
            source_training_dataset_report_sha256=report.source_training_dataset_report.sha256,
            trainer_fingerprint_sha256=report.trainer_fingerprint_sha256,
        )
    )
    errors.extend(_candidate_policy_errors(report))
    return errors


def _source_training_dataset_report(
    source: ArtifactRef,
    *,
    source_dataset_run_id: str,
) -> TrainingDatasetReport | None:
    errors = _source_training_dataset_report_errors(
        source,
        source_dataset_run_id=source_dataset_run_id,
    )
    if errors:
        raise HarnessIOError("; ".join(errors))
    source_path = local_artifact_path(
        path=source.path,
        uri=source.uri,
        field_name="policy training report.source_training_dataset_report",
    )
    assert source_path is not None
    return TrainingDatasetReport.from_dict(read_json(source_path))


def _source_training_dataset_report_errors(
    source: ArtifactRef,
    *,
    source_dataset_run_id: str,
) -> list[str]:
    errors: list[str] = []
    if source.kind != _TRAINING_DATASET_REPORT_KIND:
        errors.append(
            "policy training report.source_training_dataset_report.kind must be "
            f"{_TRAINING_DATASET_REPORT_KIND!r}"
        )
    if source.sha256 is None:
        errors.append("policy training report.source_training_dataset_report.sha256 must be set")
    if source.provenance.get("run_id") != source_dataset_run_id:
        errors.append(
            "policy training report.source_training_dataset_report provenance run_id "
            "must match source_dataset_run_id"
        )
    try:
        _require_nonempty_text(
            source.provenance.get("producer"),
            "policy training report.source_training_dataset_report provenance producer",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
    if "derivation" not in source.provenance:
        errors.append("policy training report.source_training_dataset_report must set provenance.derivation")
    try:
        source_path = local_artifact_path(
            path=source.path,
            uri=source.uri,
            field_name="policy training report.source_training_dataset_report",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        source_path = None
    if source_path is None:
        errors.append("policy training report.source_training_dataset_report must be local byte-verifiable")
    elif not source_path.exists():
        errors.append("policy training report.source_training_dataset_report.path must exist")
    elif not source_path.is_file():
        errors.append("policy training report.source_training_dataset_report.path must point to a JSON report file")
    elif source.sha256 is not None:
        if sha256_file(source_path) != source.sha256:
            errors.append("policy training report.source_training_dataset_report.sha256 must match path")
        else:
            try:
                source_report = TrainingDatasetReport.from_dict(read_json(source_path))
            except HarnessIOError as exc:
                errors.append(
                    "policy training report.source_training_dataset_report must parse: "
                    + str(exc)
                )
            else:
                if source_report.run_id != source_dataset_run_id:
                    errors.append(
                        "source training dataset report.run_id must match source_dataset_run_id"
                    )
    return errors


def _policy_artifact_errors(
    artifact: ArtifactRef,
    *,
    run_id: str,
    source_training_dataset_report_sha256: str | None,
    trainer_fingerprint_sha256: str,
) -> list[str]:
    errors: list[str] = []
    if artifact.kind != _POLICY_ARTIFACT_KIND:
        errors.append(f"policy training report.policy_artifact.kind must be {_POLICY_ARTIFACT_KIND!r}")
    if artifact.sha256 is None:
        errors.append("policy training report.policy_artifact.sha256 must be set")
    if artifact.provenance.get("run_id") != run_id:
        errors.append("policy training report.policy_artifact provenance run_id must match report.run_id")
    if artifact.provenance.get("derivation") != _DERIVATION:
        errors.append(f"policy training report.policy_artifact provenance derivation must be {_DERIVATION!r}")
    if artifact.provenance.get("source_training_dataset_report_sha256") != source_training_dataset_report_sha256:
        errors.append(
            "policy training report.policy_artifact provenance source_training_dataset_report_sha256 must match source"
        )
    if artifact.provenance.get("trainer_fingerprint_sha256") != trainer_fingerprint_sha256:
        errors.append("policy training report.policy_artifact provenance trainer_fingerprint_sha256 must match trainer")
    try:
        _require_nonempty_text(
            artifact.provenance.get("producer"),
            "policy training report.policy_artifact provenance producer",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
    try:
        artifact_path = local_artifact_path(
            path=artifact.path,
            uri=artifact.uri,
            field_name="policy training report.policy_artifact",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        artifact_path = None
    if artifact_path is None:
        errors.append("policy training report.policy_artifact must be local byte-verifiable")
    elif not artifact_path.exists():
        errors.append("policy training report.policy_artifact.path must exist")
    elif not artifact_path.is_file():
        errors.append("policy training report.policy_artifact.path must point to a policy checkpoint file")
    elif artifact.sha256 is not None and sha256_file(artifact_path) != artifact.sha256:
        errors.append("policy training report.policy_artifact.sha256 must match path")
    return errors


def _candidate_policy_errors(report: PolicyTrainingRunReport) -> list[str]:
    errors: list[str] = []
    policy = report.candidate_policy
    if policy.runtime_role is not RuntimeRole.live_policy:
        errors.append("policy training report.candidate_policy must be a live_policy backend")
    if policy.runtime_boundary is None:
        errors.append("policy training report.candidate_policy must set runtime_boundary")
    else:
        if policy.runtime_boundary.policy_artifact != report.policy_artifact:
            errors.append(
                "policy training report.candidate_policy.runtime_boundary.policy_artifact must match policy_artifact"
            )
    if policy.provenance.get("run_id") != report.run_id:
        errors.append("policy training report.candidate_policy provenance run_id must match report.run_id")
    if policy.provenance.get("source_training_dataset_report_sha256") != report.source_training_dataset_report.sha256:
        errors.append(
            "policy training report.candidate_policy provenance source_training_dataset_report_sha256 must match source"
        )
    if policy.provenance.get("policy_artifact_sha256") != report.policy_artifact.sha256:
        errors.append("policy training report.candidate_policy provenance policy_artifact_sha256 must match policy_artifact")
    if policy.provenance.get("trainer_fingerprint_sha256") != report.trainer_fingerprint_sha256:
        errors.append("policy training report.candidate_policy provenance trainer_fingerprint_sha256 must match trainer")
    return errors


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed_keys: frozenset[str],
    field_name: str,
) -> None:
    unknown_keys = sorted(repr(key) for key in value if key not in allowed_keys)
    if unknown_keys:
        raise HarnessIOError(f"{field_name} has unknown fields: {', '.join(unknown_keys)}")


def _require_keys(
    value: Mapping[str, Any],
    required_keys: frozenset[str],
    field_name: str,
) -> None:
    missing_keys = sorted(repr(key) for key in required_keys if key not in value)
    if missing_keys:
        raise HarnessIOError(f"{field_name} is missing required fields: {', '.join(missing_keys)}")


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessIOError(f"{field_name} must be a nonempty string")
    return value


def _optional_nonempty_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_text(value, field_name)


def _require_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise HarnessIOError(f"{field_name} must be 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise HarnessIOError(f"{field_name} must be 64 hexadecimal characters") from exc
    return value.lower()


def _optional_nonnegative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise HarnessIOError(f"{field_name} must be a nonnegative integer")
    return value


def _as_sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if value is None:
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    return tuple(value)


def _as_text_sequence(value: Any, field_name: str, *, allow_empty: bool) -> tuple[str, ...]:
    items = _as_sequence(value, field_name)
    if not allow_empty and not items:
        raise HarnessIOError(f"{field_name} must not be empty")
    return tuple(_require_nonempty_text(item, field_name) for item in items)


def _environment_mapping(value: Any, field_name: str) -> MappingProxyType[str, str]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    copied: dict[str, str] = {}
    for key, item in value.items():
        key_text = _require_nonempty_text(key, f"{field_name} key")
        item_text = _require_nonempty_text(item, f"{field_name}.{key_text}")
        copied[key_text] = item_text
    return MappingProxyType(copied)


def _metric_mapping(value: Any, field_name: str) -> MappingProxyType[str, float]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    copied: dict[str, float] = {}
    for key, item in value.items():
        key_text = _require_nonempty_text(key, f"{field_name} key")
        if type(item) not in (float, int) or not math.isfinite(float(item)):
            raise HarnessIOError(f"{field_name}.{key_text} must be a finite number")
        copied[key_text] = float(item)
    return MappingProxyType(copied)


def _copy_json_mapping(value: Any, field_name: str) -> MappingProxyType[str, Any]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise HarnessIOError(f"{field_name} keys must be nonempty strings")
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


def _reserved_artifact_key_paths(
    value: Any,
    *,
    path: str,
    artifact_context: bool = False,
) -> list[str]:
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, item in value.items():
            if not isinstance(key, str):
                paths.append(path)
                continue
            normalized = key.strip().lower().replace("-", "_")
            key_path = f"{path}.{key}"
            if _is_reserved_artifact_key(normalized):
                paths.append(key_path)
            if _looks_like_artifact_location(item):
                paths.append(key_path)
            next_artifact_context = artifact_context or normalized in {
                "artifact",
                "artifacts",
                "checkpoint",
                "data",
                "dataset",
                "model",
                "policy",
                "weights",
            }
            if artifact_context and normalized in {"file", "path", "uri", "url"}:
                paths.append(key_path)
            paths.extend(
                _reserved_artifact_key_paths(
                    item,
                    path=key_path,
                    artifact_context=next_artifact_context,
                )
            )
        return sorted(set(paths))
    if isinstance(value, (list, tuple)):
        paths: list[str] = []
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            if _looks_like_artifact_location(item):
                paths.append(item_path)
            paths.extend(
                _reserved_artifact_key_paths(
                    item,
                    path=item_path,
                    artifact_context=artifact_context,
                )
            )
        return sorted(set(paths))
    return []


def _is_reserved_artifact_key(normalized_key: str) -> bool:
    if normalized_key == "gradient_checkpointing":
        return False
    if normalized_key in {
        "artifact",
        "artifacts",
        "checkpoint",
        "checkpoints",
        "data",
        "dataset",
        "datasets",
        "dataset_path",
        "dataset_uri",
        "model",
        "models",
        "model_path",
        "model_uri",
        "policies",
        "policy",
        "policy_artifact",
        "policy_checkpoint",
        "weight",
        "weights",
        "weights_path",
        "weights_uri",
    }:
        return True
    if any(normalized_key.startswith(prefix + "_") for prefix in _RESERVED_ARTIFACT_KEY_PREFIXES):
        if normalized_key.endswith(("_file", "_path", "_uri", "_url")):
            return True
    return normalized_key.endswith(_RESERVED_ARTIFACT_KEY_SUFFIXES)


def _reserved_environment_artifact_keys(value: Mapping[str, str]) -> list[str]:
    reserved: list[str] = []
    for key, item in value.items():
        normalized = key.strip().upper()
        artifact_key = any(
            token in normalized
            for token in (
                "ARTIFACT",
                "ARTIFACTS",
                "CHECKPOINT",
                "DATASET",
                "MODEL",
                "POLICY",
                "WEIGHTS",
            )
        )
        location_key = normalized.endswith(("_FILE", "_PATH", "_URI", "_URL")) or "GCS_URI" in normalized
        if artifact_key or location_key or _looks_like_artifact_location(item):
            reserved.append(key)
    return sorted(reserved)


def _looks_like_artifact_location(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    if not normalized:
        return False
    parsed = urlparse(normalized)
    return (
        bool(parsed.scheme and (parsed.netloc or parsed.scheme == "file"))
        or normalized.startswith(("/", "./", "../", "~"))
        or normalized.endswith(_ARTIFACT_VALUE_SUFFIXES)
    )
