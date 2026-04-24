"""Training dataset contracts derived from offline AIC training signals."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeVar, cast

from aic_signal_harness.artifacts import HarnessIOError, local_artifact_path, read_json, sha256_file
from aic_signal_harness.schemas import ArtifactRef, LeakageClass, SchemaValidationError
from aic_signal_harness.training_signal import (
    TrainingSignal,
    TrainingSignalKind,
    TrainingSignalReport,
)


TRAINING_DATASET_SCHEMA_VERSION = 1


_TRAINING_DATASET_EXAMPLE_KEYS = frozenset(
    {
        "example_id",
        "source_signal_id",
        "kind",
        "target",
        "weight",
        "source",
        "source_event_indices",
        "extraction_method",
        "leakage_class",
        "evidence",
        "trial_id",
        "split",
        "offline_only",
        "runtime_allowed",
        "consumable_by_policy_runtime",
    }
)
_TRAINING_DATASET_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "generated_at_utc",
        "source_training_signal_report",
        "dataset_fingerprint_sha256",
        "example_count",
        "selected_signal_ids",
        "signals_by_kind",
        "signals_by_leakage_class",
        "signals_by_target",
        "examples",
        "ok",
        "errors",
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


class TrainingDatasetSplit(_StrEnum):
    """Explicit split assignment for offline training examples."""

    train = "train"
    validation = "validation"
    test = "test"


@dataclass(frozen=True)
class TrainingDatasetExample:
    """One offline training example selected from a training signal."""

    example_id: str
    source_signal_id: str
    kind: TrainingSignalKind
    target: str
    weight: float
    source: str
    source_event_indices: tuple[int, ...]
    extraction_method: str
    leakage_class: LeakageClass
    evidence: Mapping[str, Any] = field(default_factory=dict)
    trial_id: str | None = None
    split: TrainingDatasetSplit = TrainingDatasetSplit.train
    offline_only: bool = True
    runtime_allowed: bool = False
    consumable_by_policy_runtime: bool = False

    def __post_init__(self) -> None:
        errors: list[str] = []
        for field_name in ("example_id", "source_signal_id", "target", "source", "extraction_method"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"training dataset example.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(self, "kind", TrainingSignalKind.parse(self.kind))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "weight",
                _require_finite_float(self.weight, "training dataset example.weight"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "source_event_indices",
                _source_event_indices(
                    self.source_event_indices,
                    "training dataset example.source_event_indices",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "leakage_class", LeakageClass.parse(self.leakage_class))
        except SchemaValidationError as exc:
            errors.extend(exc.errors)
        try:
            object.__setattr__(self, "split", TrainingDatasetSplit.parse(self.split))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "trial_id",
                _optional_nonempty_text(self.trial_id, "training dataset example.trial_id"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "evidence",
                _copy_json_mapping(self.evidence, "training dataset example.evidence"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if self.offline_only is not True:
            errors.append("training dataset example.offline_only must be true")
        if self.runtime_allowed is not False:
            errors.append("training dataset example.runtime_allowed must be false")
        if self.consumable_by_policy_runtime is not False:
            errors.append("training dataset example.consumable_by_policy_runtime must be false")
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "example_id": self.example_id,
            "source_signal_id": self.source_signal_id,
            "kind": self.kind.value,
            "target": self.target,
            "weight": self.weight,
            "source": self.source,
            "source_event_indices": list(self.source_event_indices),
            "extraction_method": self.extraction_method,
            "leakage_class": self.leakage_class.value,
            "evidence": _thaw_json(self.evidence),
            "split": self.split.value,
            "offline_only": self.offline_only,
            "runtime_allowed": self.runtime_allowed,
            "consumable_by_policy_runtime": self.consumable_by_policy_runtime,
        }
        if self.trial_id is not None:
            value["trial_id"] = self.trial_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingDatasetExample":
        if not isinstance(value, Mapping):
            raise HarnessIOError("training dataset example must be a mapping")
        _reject_unknown_keys(
            value,
            _TRAINING_DATASET_EXAMPLE_KEYS,
            "training dataset example",
        )
        _require_keys(
            value,
            _TRAINING_DATASET_EXAMPLE_KEYS
            - {
                "consumable_by_policy_runtime",
                "offline_only",
                "runtime_allowed",
                "trial_id",
            },
            "training dataset example",
        )
        missing_runtime_fields = sorted(
            field_name
            for field_name in (
                "consumable_by_policy_runtime",
                "offline_only",
                "runtime_allowed",
            )
            if field_name not in value
        )
        if missing_runtime_fields:
            raise HarnessIOError(
                "training dataset example missing required runtime boundary fields: "
                + ", ".join(missing_runtime_fields)
            )
        return cls(
            example_id=cast(Any, value.get("example_id")),
            source_signal_id=cast(Any, value.get("source_signal_id")),
            kind=cast(Any, value.get("kind")),
            target=cast(Any, value.get("target")),
            weight=cast(Any, value.get("weight")),
            source=cast(Any, value.get("source")),
            source_event_indices=cast(Any, value.get("source_event_indices")),
            extraction_method=cast(Any, value.get("extraction_method")),
            leakage_class=cast(Any, value.get("leakage_class")),
            evidence=value.get("evidence", {}),
            trial_id=value.get("trial_id"),
            split=cast(Any, value.get("split")),
            offline_only=cast(bool, value.get("offline_only")),
            runtime_allowed=cast(bool, value.get("runtime_allowed")),
            consumable_by_policy_runtime=cast(
                bool,
                value.get("consumable_by_policy_runtime"),
            ),
        )


@dataclass(frozen=True)
class TrainingDatasetReport:
    """Deterministic offline dataset selected from one training-signal report."""

    run_id: str
    generated_at_utc: str
    source_training_signal_report: ArtifactRef
    dataset_fingerprint_sha256: str
    example_count: int
    selected_signal_ids: tuple[str, ...]
    signals_by_kind: Mapping[str, int]
    signals_by_leakage_class: Mapping[str, int]
    signals_by_target: Mapping[str, int]
    examples: tuple[TrainingDatasetExample, ...] = field(default_factory=tuple)
    ok: bool = False
    errors: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = TRAINING_DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if (
            type(self.schema_version) is not int
            or self.schema_version != TRAINING_DATASET_SCHEMA_VERSION
        ):
            errors.append(f"schema_version must be {TRAINING_DATASET_SCHEMA_VERSION}")
        try:
            object.__setattr__(
                self,
                "run_id",
                _require_nonempty_text(self.run_id, "training dataset report.run_id"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "generated_at_utc",
                _require_nonempty_text(
                    self.generated_at_utc,
                    "training dataset report.generated_at_utc",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not isinstance(self.source_training_signal_report, ArtifactRef):
            try:
                object.__setattr__(
                    self,
                    "source_training_signal_report",
                    ArtifactRef.from_dict(self.source_training_signal_report),
                )
            except SchemaValidationError as exc:
                errors.extend(exc.errors)
        try:
            object.__setattr__(
                self,
                "dataset_fingerprint_sha256",
                _require_sha256(
                    self.dataset_fingerprint_sha256,
                    "training dataset report.dataset_fingerprint_sha256",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "example_count",
                _require_nonnegative_int(
                    self.example_count,
                    "training dataset report.example_count",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            examples = tuple(
                example
                if isinstance(example, TrainingDatasetExample)
                else TrainingDatasetExample.from_dict(example)
                for example in _as_sequence(self.examples, "training dataset report.examples")
            )
            object.__setattr__(self, "examples", examples)
        except HarnessIOError as exc:
            errors.append(str(exc))
            examples = tuple()
        for field_name in (
            "selected_signal_ids",
            "errors",
            "notes",
        ):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _as_text_sequence(
                        getattr(self, field_name),
                        f"training dataset report.{field_name}",
                        allow_empty=True,
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        for field_name in (
            "signals_by_kind",
            "signals_by_leakage_class",
            "signals_by_target",
        ):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _count_mapping(getattr(self, field_name), f"training dataset report.{field_name}"),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        if type(self.ok) is not bool:
            errors.append("training dataset report.ok must be a boolean")

        if not errors:
            errors.extend(_dataset_report_consistency_errors(self))
            errors.extend(_source_training_signal_report_errors(self))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generated_at_utc": self.generated_at_utc,
            "source_training_signal_report": self.source_training_signal_report.to_dict(),
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "example_count": self.example_count,
            "selected_signal_ids": list(self.selected_signal_ids),
            "signals_by_kind": dict(self.signals_by_kind),
            "signals_by_leakage_class": dict(self.signals_by_leakage_class),
            "signals_by_target": dict(self.signals_by_target),
            "examples": [example.to_dict() for example in self.examples],
            "ok": self.ok,
            "errors": list(self.errors),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingDatasetReport":
        if not isinstance(value, Mapping):
            raise HarnessIOError("training dataset report must be a mapping")
        _reject_unknown_keys(
            value,
            _TRAINING_DATASET_REPORT_KEYS,
            "training dataset report",
        )
        _require_keys(value, _TRAINING_DATASET_REPORT_KEYS, "training dataset report")
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            run_id=cast(Any, value.get("run_id")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            source_training_signal_report=cast(Any, value.get("source_training_signal_report")),
            dataset_fingerprint_sha256=cast(Any, value.get("dataset_fingerprint_sha256")),
            example_count=cast(Any, value.get("example_count")),
            selected_signal_ids=cast(Any, value.get("selected_signal_ids", ())),
            signals_by_kind=cast(Any, value.get("signals_by_kind", {})),
            signals_by_leakage_class=cast(Any, value.get("signals_by_leakage_class", {})),
            signals_by_target=cast(Any, value.get("signals_by_target", {})),
            examples=cast(Any, value.get("examples", ())),
            ok=cast(bool, value.get("ok")),
            errors=cast(Any, value.get("errors", ())),
            notes=cast(Any, value.get("notes", ())),
        )


def training_dataset_example_id(
    *,
    source_training_signal_report_sha256: str,
    signal_id: str,
) -> str:
    """Return the stable example id for a selected training signal."""

    source_sha = _require_sha256(
        source_training_signal_report_sha256,
        "source_training_signal_report_sha256",
    )
    source_signal_id = _require_nonempty_text(signal_id, "signal_id")
    payload = {
        "signal_id": source_signal_id,
        "source_training_signal_report_sha256": source_sha,
    }
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return "tdsex_" + hashlib.sha256(encoded).hexdigest()[:20]


def training_dataset_fingerprint_sha256(examples: tuple[TrainingDatasetExample, ...]) -> str:
    """Return a stable content fingerprint for exact selected examples."""

    payload = [example.to_dict() for example in examples]
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def training_signal_to_dataset_example(
    signal: TrainingSignal,
    *,
    source_training_signal_report_sha256: str,
    split: TrainingDatasetSplit = TrainingDatasetSplit.train,
) -> TrainingDatasetExample:
    """Project a validated training signal into an offline dataset example."""

    return TrainingDatasetExample(
        example_id=training_dataset_example_id(
            source_training_signal_report_sha256=source_training_signal_report_sha256,
            signal_id=signal.signal_id,
        ),
        source_signal_id=signal.signal_id,
        kind=signal.kind,
        target=signal.target,
        weight=signal.weight,
        source=signal.source,
        source_event_indices=signal.source_event_indices,
        extraction_method=signal.extraction_method,
        leakage_class=signal.leakage_class,
        evidence=signal.evidence,
        trial_id=signal.trial_id,
        split=split,
        offline_only=True,
        runtime_allowed=False,
        consumable_by_policy_runtime=False,
    )


def _dataset_report_consistency_errors(report: TrainingDatasetReport) -> list[str]:
    errors: list[str] = []
    examples = report.examples
    if report.example_count != len(examples):
        errors.append("training dataset report.example_count must match examples length")
    selected_signal_ids = tuple(example.source_signal_id for example in examples)
    if report.selected_signal_ids != selected_signal_ids:
        errors.append("training dataset report.selected_signal_ids must match examples")
    duplicate_example_ids = _duplicates(example.example_id for example in examples)
    if duplicate_example_ids:
        errors.append(
            "training dataset report.examples must not contain duplicate example_id: "
            + ", ".join(duplicate_example_ids)
        )
    duplicate_signal_ids = _duplicates(example.source_signal_id for example in examples)
    if duplicate_signal_ids:
        errors.append(
            "training dataset report.examples must not contain duplicate source_signal_id: "
            + ", ".join(duplicate_signal_ids)
        )
    expected_fingerprint = training_dataset_fingerprint_sha256(examples)
    if report.dataset_fingerprint_sha256 != expected_fingerprint:
        errors.append("training dataset report.dataset_fingerprint_sha256 must match examples")
    expected_by_kind = _expected_counts(example.kind.value for example in examples)
    if dict(report.signals_by_kind) != expected_by_kind:
        errors.append("training dataset report.signals_by_kind must match examples")
    expected_by_leakage = _expected_counts(example.leakage_class.value for example in examples)
    if dict(report.signals_by_leakage_class) != expected_by_leakage:
        errors.append("training dataset report.signals_by_leakage_class must match examples")
    expected_by_target = _expected_counts(example.target for example in examples)
    if dict(report.signals_by_target) != expected_by_target:
        errors.append("training dataset report.signals_by_target must match examples")
    if report.ok and report.errors:
        errors.append("training dataset report.ok=true requires errors to be empty")
    if not report.ok and not report.errors:
        errors.append("training dataset report.ok=false requires at least one error")
    return errors


def _source_training_signal_report_errors(report: TrainingDatasetReport) -> list[str]:
    errors: list[str] = []
    source = report.source_training_signal_report
    if source.kind != "training_signal_report":
        errors.append(
            "training dataset report.source_training_signal_report.kind must be 'training_signal_report'"
        )
    if source.sha256 is None:
        errors.append("training dataset report.source_training_signal_report.sha256 must be set")
    if source.provenance.get("run_id") != report.run_id:
        errors.append(
            "training dataset report.source_training_signal_report provenance run_id must match report.run_id"
        )
    if "producer" not in source.provenance:
        errors.append("training dataset report.source_training_signal_report must set provenance.producer")
    if "derivation" not in source.provenance:
        errors.append("training dataset report.source_training_signal_report must set provenance.derivation")
    try:
        source_path = local_artifact_path(
            path=source.path,
            uri=source.uri,
            field_name="training dataset report.source_training_signal_report",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        source_path = None
    if source_path is None:
        errors.append(
            "training dataset report.source_training_signal_report must be local byte-verifiable"
        )
    elif not source_path.exists():
        errors.append("training dataset report.source_training_signal_report.path must exist")
    elif not source_path.is_file():
        errors.append("training dataset report.source_training_signal_report.path must point to a JSON report file")
    elif source.sha256 is not None:
        if sha256_file(source_path) != source.sha256:
            errors.append("training dataset report.source_training_signal_report.sha256 must match path")
        else:
            try:
                source_report = TrainingSignalReport.from_dict(read_json(source_path))
            except HarnessIOError as exc:
                errors.append(
                    "training dataset report.source_training_signal_report must parse: "
                    + str(exc)
                )
            else:
                if source_report.run_id != report.run_id:
                    errors.append("source training signal report.run_id must match dataset report.run_id")
                if source_report.generated_at_utc != report.generated_at_utc:
                    errors.append(
                        "training dataset report.generated_at_utc must match source training signal report"
                    )
                if tuple(signal.signal_id for signal in source_report.signals) != report.selected_signal_ids:
                    errors.append(
                        "training dataset report.selected_signal_ids must include every source signal in source order"
                    )
                expected_ok = bool(source_report.signals)
                expected_errors = tuple() if expected_ok else ("no selected training signals",)
                if report.ok is not expected_ok:
                    errors.append("training dataset report.ok must match source signal selection")
                if report.errors != expected_errors:
                    errors.append("training dataset report.errors must match source signal selection")
                errors.extend(
                    _examples_match_source_signal_errors(
                        report.examples,
                        source_report=source_report,
                        source_sha256=source.sha256,
                    )
                )
    return errors


def _examples_match_source_signal_errors(
    examples: tuple[TrainingDatasetExample, ...],
    *,
    source_report: TrainingSignalReport,
    source_sha256: str,
) -> list[str]:
    errors: list[str] = []
    signals_by_id = {signal.signal_id: signal for signal in source_report.signals}
    for example in examples:
        signal = signals_by_id.get(example.source_signal_id)
        if signal is None:
            errors.append(
                f"training dataset example {example.example_id} source_signal_id "
                f"{example.source_signal_id!r} is not present in source report"
            )
            continue
        expected_example = training_signal_to_dataset_example(
            signal,
            source_training_signal_report_sha256=source_sha256,
        )
        if example != expected_example:
            errors.append(
                f"training dataset example {example.example_id} must match source signal "
                f"{example.source_signal_id!r}"
            )
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


def _require_finite_float(value: Any, field_name: str) -> float:
    if type(value) not in (float, int) or not math.isfinite(float(value)):
        raise HarnessIOError(f"{field_name} must be a finite number")
    return float(value)


def _require_nonnegative_int(value: Any, field_name: str) -> int:
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


def _source_event_indices(value: Any, field_name: str) -> tuple[int, ...]:
    items = _as_sequence(value, field_name)
    if not items:
        raise HarnessIOError(f"{field_name} must not be empty")
    indices: list[int] = []
    for item in items:
        if type(item) is not int or item < 0:
            raise HarnessIOError(f"{field_name} must contain nonnegative integers")
        indices.append(item)
    if len(set(indices)) != len(indices):
        raise HarnessIOError(f"{field_name} must not contain duplicates")
    if tuple(indices) != tuple(sorted(indices)):
        raise HarnessIOError(f"{field_name} must be sorted")
    return tuple(indices)


def _count_mapping(value: Any, field_name: str) -> MappingProxyType[str, int]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    copied: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key.strip():
            raise HarnessIOError(f"{field_name} keys must be nonempty strings")
        if type(count) is not int or count < 0:
            raise HarnessIOError(f"{field_name} values must be nonnegative integers")
        copied[key] = count
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


def _expected_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _duplicates(values: Any) -> list[str]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(value for value, count in counts.items() if count > 1)
