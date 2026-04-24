"""Typed append-only ledger contracts for the backend-neutral AIC signal harness."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping, TypeVar
from urllib.parse import urlparse

from aic_signal_harness.artifacts import HarnessIOError
from aic_signal_harness.manifest import RunStatus
from aic_signal_harness.schemas import ArtifactRef, BackendKind, SCHEMA_VERSION


_LEDGER_METRIC_KEYS = frozenset({"name", "goal", "value"})
_LEDGER_DATASET_KEYS = frozenset({"artifact", "episode_count", "step_count"})
_LEDGER_ENTRY_KEYS = frozenset(
    {
        "schema_version",
        "recorded_at_utc",
        "run_id",
        "status",
        "backend_kind",
        "manifest",
        "metric",
        "dataset",
        "experiment_id",
        "notes",
        "promotion",
    }
)
_URI_SOURCE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


class _StrEnum(str, Enum):
    @classmethod
    def parse(cls: type["_EnumT"], value: str | "_EnumT") -> "_EnumT":
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


class LedgerMetricGoal(_StrEnum):
    max = "max"
    min = "min"


@dataclass(frozen=True)
class LedgerMetric:
    """Stable metric snapshot stored with one ledger entry."""

    name: str
    goal: LedgerMetricGoal
    value: float | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(
                self,
                "name",
                _require_nonempty_text(self.name, "ledger metric.name"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "goal", LedgerMetricGoal.parse(self.goal))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "value",
                _optional_finite_float(self.value, "ledger metric.value"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "goal": self.goal.value,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LedgerMetric":
        if not isinstance(value, Mapping):
            raise HarnessIOError("ledger metric must be a mapping")
        _reject_unknown_keys(value, _LEDGER_METRIC_KEYS, "ledger metric")
        return cls(
            name=value.get("name"),
            goal=value.get("goal"),
            value=value.get("value"),
        )


@dataclass(frozen=True)
class LedgerDatasetSummary:
    """Dataset evidence summary linked from one ledger entry."""

    artifact: ArtifactRef
    episode_count: int
    step_count: int

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.artifact, ArtifactRef):
            try:
                object.__setattr__(self, "artifact", ArtifactRef.from_dict(self.artifact))
            except Exception as exc:  # pragma: no cover - ArtifactRef owns detail
                errors.append(str(exc))
        if not errors:
            if self.artifact.kind != "hdf5_dataset":
                errors.append("ledger dataset artifact.kind must be 'hdf5_dataset'")
            if self.artifact.sha256 is None:
                errors.append("ledger dataset artifact.sha256 must be set")
            errors.extend(
                _artifact_source_identity_errors(
                    self.artifact,
                    field_name="ledger dataset artifact",
                )
            )
        for field_name in ("episode_count", "step_count"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonnegative_int(
                        getattr(self, field_name),
                        f"ledger dataset {field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_dict(),
            "episode_count": self.episode_count,
            "step_count": self.step_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LedgerDatasetSummary":
        if not isinstance(value, Mapping):
            raise HarnessIOError("ledger dataset summary must be a mapping")
        _reject_unknown_keys(value, _LEDGER_DATASET_KEYS, "ledger dataset summary")
        return cls(
            artifact=value.get("artifact"),
            episode_count=value.get("episode_count"),
            step_count=value.get("step_count"),
        )


@dataclass(frozen=True)
class LedgerEntry:
    """Typed append-only accounting record for one observed harness run."""

    recorded_at_utc: str
    run_id: str
    status: RunStatus
    backend_kind: BackendKind
    manifest: ArtifactRef
    metric: LedgerMetric
    dataset: LedgerDatasetSummary | None = None
    experiment_id: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    promotion: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        try:
            object.__setattr__(
                self,
                "recorded_at_utc",
                _require_nonempty_text(self.recorded_at_utc, "ledger entry.recorded_at_utc"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "run_id", _require_nonempty_text(self.run_id, "ledger entry.run_id"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "status", RunStatus.parse(self.status))
        except Exception as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "backend_kind", BackendKind.parse(self.backend_kind))
        except Exception as exc:
            errors.append(str(exc))

        if not isinstance(self.manifest, ArtifactRef):
            try:
                object.__setattr__(self, "manifest", ArtifactRef.from_dict(self.manifest))
            except Exception as exc:  # pragma: no cover - ArtifactRef owns detail
                errors.append(str(exc))
        if not errors:
            if self.manifest.kind != "run_manifest":
                errors.append("ledger manifest artifact.kind must be 'run_manifest'")
            if self.manifest.sha256 is None:
                errors.append("ledger manifest artifact.sha256 must be set")
            errors.extend(
                _artifact_source_identity_errors(
                    self.manifest,
                    field_name="ledger manifest artifact",
                )
            )

        if not isinstance(self.metric, LedgerMetric):
            try:
                object.__setattr__(self, "metric", LedgerMetric.from_dict(self.metric))
            except HarnessIOError as exc:
                errors.append(str(exc))

        if self.dataset is not None and not isinstance(self.dataset, LedgerDatasetSummary):
            try:
                object.__setattr__(
                    self,
                    "dataset",
                    LedgerDatasetSummary.from_dict(self.dataset),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))

        try:
            object.__setattr__(
                self,
                "experiment_id",
                _optional_nonempty_text(self.experiment_id, "ledger entry.experiment_id"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))

        try:
            notes = _as_text_sequence(self.notes, "ledger entry.notes")
            object.__setattr__(self, "notes", notes)
        except HarnessIOError as exc:
            errors.append(str(exc))

        try:
            promotion = _copy_json_mapping(self.promotion, "ledger entry.promotion")
            object.__setattr__(self, "promotion", promotion)
        except HarnessIOError as exc:
            errors.append(str(exc))

        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "recorded_at_utc": self.recorded_at_utc,
            "run_id": self.run_id,
            "status": self.status.value,
            "backend_kind": self.backend_kind.value,
            "manifest": self.manifest.to_dict(),
            "metric": self.metric.to_dict(),
            "dataset": None if self.dataset is None else self.dataset.to_dict(),
            "experiment_id": self.experiment_id,
            "notes": list(self.notes),
            "promotion": _thaw_json(self.promotion),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LedgerEntry":
        if not isinstance(value, Mapping):
            raise HarnessIOError("ledger entry must be a mapping")
        _reject_unknown_keys(value, _LEDGER_ENTRY_KEYS, "ledger entry")
        return cls(
            schema_version=value.get("schema_version"),
            recorded_at_utc=value.get("recorded_at_utc"),
            run_id=value.get("run_id"),
            status=value.get("status"),
            backend_kind=value.get("backend_kind"),
            manifest=value.get("manifest"),
            metric=value.get("metric"),
            dataset=value.get("dataset"),
            experiment_id=value.get("experiment_id"),
            notes=value.get("notes", ()),
            promotion=value.get("promotion", {}),
        )


def read_ledger_entries(path: str | Path) -> tuple[LedgerEntry, ...]:
    """Read a JSONL ledger into typed entries."""

    ledger_path = Path(path)
    if not ledger_path.exists():
        return ()
    entries: list[LedgerEntry] = []
    try:
        with ledger_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise HarnessIOError(
                        f"invalid JSONL in {ledger_path}:{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(payload, Mapping):
                    raise HarnessIOError(
                        f"ledger line {line_number} in {ledger_path} must be a JSON object"
                    )
                try:
                    entries.append(LedgerEntry.from_dict(payload))
                except HarnessIOError as exc:
                    raise HarnessIOError(
                        f"invalid ledger entry at {ledger_path}:{line_number}: {exc}"
                    ) from exc
    except OSError as exc:
        raise HarnessIOError(f"failed to read ledger {ledger_path}: {exc}") from exc
    return tuple(entries)


def append_ledger_entry(
    path: str | Path,
    entry: LedgerEntry | Mapping[str, Any],
    *,
    allow_duplicate: bool = False,
) -> LedgerEntry:
    """Append one typed entry to a JSONL ledger."""

    typed_entry = entry if isinstance(entry, LedgerEntry) else LedgerEntry.from_dict(entry)
    ledger_path = Path(path)
    existing_entries = read_ledger_entries(ledger_path)
    if not allow_duplicate and any(
        existing_entry.run_id == typed_entry.run_id for existing_entry in existing_entries
    ):
        raise HarnessIOError(f"ledger already contains run_id={typed_entry.run_id}")

    try:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a", encoding="utf-8") as handle:
            json.dump(typed_entry.to_dict(), handle, allow_nan=False, sort_keys=True)
            handle.write("\n")
    except (TypeError, ValueError) as exc:
        raise HarnessIOError(f"ledger entry is not strict JSON: {exc}") from exc
    except OSError as exc:
        raise HarnessIOError(f"failed to append ledger {ledger_path}: {exc}") from exc
    return typed_entry


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
    return value


def _optional_nonempty_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_text(value, field_name)


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise HarnessIOError(f"{field_name} must be a nonnegative integer")
    return value


def _optional_finite_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise HarnessIOError(f"{field_name} must be a finite number")
    return float(value)


def _as_text_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    return tuple(_require_nonempty_text(item, f"{field_name}[{index}]") for index, item in enumerate(value))


def _artifact_source_identity_errors(
    artifact: ArtifactRef,
    *,
    field_name: str,
) -> list[str]:
    errors: list[str] = []
    for source_field in ("path", "uri"):
        source = getattr(artifact, source_field)
        if source is None:
            continue
        try:
            _validate_source_identity(source, f"{field_name}.{source_field}")
        except HarnessIOError as exc:
            errors.append(str(exc))
    return errors


def _validate_source_identity(value: Any, field_name: str) -> str:
    source = _require_nonempty_text(value, field_name)
    if _is_uri_source(source):
        parsed = urlparse(source)
        if parsed.scheme == "file":
            if not parsed.path:
                raise HarnessIOError(f"{field_name} must include a file path")
        elif not parsed.netloc:
            raise HarnessIOError(f"{field_name} must include a network location")
        if _has_dot_segments(tuple(segment for segment in parsed.path.split("/") if segment)):
            raise HarnessIOError(f"{field_name} must not contain '.' or '..' path segments")
        return source
    if not _is_absolute_local_source(source):
        raise HarnessIOError(f"{field_name} must be an absolute local path or uri")
    if _has_dot_segments(_local_source_path(source).parts):
        raise HarnessIOError(f"{field_name} must not contain '.' or '..' path segments")
    return source


def _is_uri_source(source: str) -> bool:
    return bool(_URI_SOURCE_RE.match(source))


def _is_absolute_local_source(source: str) -> bool:
    return Path(source).is_absolute() or PureWindowsPath(source).is_absolute()


def _local_source_path(source: str) -> PurePosixPath | PureWindowsPath:
    if PureWindowsPath(source).is_absolute():
        return PureWindowsPath(source)
    return PurePosixPath(source)


def _has_dot_segments(parts: tuple[str, ...]) -> bool:
    return any(part in (".", "..") for part in parts)


def _copy_json_mapping(value: Mapping[str, Any] | None, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    copied, errors = _jsonable_copy(value, field_name)
    if errors:
        raise HarnessIOError("; ".join(errors))
    return _freeze_json(copied)


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
        errors: list[str] = []
        for index, item in enumerate(value):
            copied_item, item_errors = _jsonable_copy(item, f"{field_name}[{index}]")
            copied_list.append(copied_item)
            errors.extend(item_errors)
        return copied_list, errors
    return value, [f"{field_name} must be JSON serializable"]


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
    return value
