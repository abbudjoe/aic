"""Typed validation contracts for AIC demonstration HDF5 datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlparse

from aic_signal_harness.artifacts import HarnessIOError
from aic_signal_harness.schemas import SCHEMA_VERSION


_HDF5_DATASET_FIELD_KEYS = frozenset({"shape", "dtype"})
_HDF5_DATASET_THRESHOLDS_KEYS = frozenset({"min_episodes", "min_steps"})
_HDF5_DATASET_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "source",
        "size_bytes",
        "sha256",
        "validated_at_utc",
        "thresholds",
        "required_datasets",
        "missing_datasets",
        "datasets",
        "episode_count",
        "step_count",
        "episode_lengths",
        "episode_offsets",
        "errors",
        "ok",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_URI_SOURCE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")

REQUIRED_HDF5_DATASET_KEYS = (
    "ep_len",
    "ep_offset",
    "ep_idx",
    "episode_idx",
    "step_idx",
    "pixels",
    "left_pixels",
    "right_pixels",
    "proprio",
    "state",
    "action",
    "task_id",
    "plug_type",
    "port_type",
    "target_module_name",
)
PER_STEP_HDF5_DATASET_KEYS = (
    "ep_idx",
    "episode_idx",
    "step_idx",
    "pixels",
    "left_pixels",
    "right_pixels",
    "proprio",
    "state",
    "action",
    "task_id",
    "plug_type",
    "port_type",
    "target_module_name",
)


@dataclass(frozen=True)
class Hdf5DatasetStats:
    """Typed shape and dtype summary for one HDF5 dataset entry."""

    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(
                self,
                "shape",
                _as_nonnegative_int_tuple(self.shape, "hdf5 dataset stats.shape"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "dtype",
                _require_nonempty_text(self.dtype, "hdf5 dataset stats.dtype"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Hdf5DatasetStats":
        if not isinstance(value, Mapping):
            raise HarnessIOError("hdf5 dataset stats must be a mapping")
        _reject_unknown_keys(value, _HDF5_DATASET_FIELD_KEYS, "hdf5 dataset stats")
        return cls(
            shape=value.get("shape"),
            dtype=value.get("dtype"),
        )


@dataclass(frozen=True)
class Hdf5DatasetThresholds:
    """Explicit validation thresholds recorded with each dataset report."""

    min_episodes: int = 1
    min_steps: int = 1

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(
                self,
                "min_episodes",
                _require_nonnegative_int(self.min_episodes, "hdf5 thresholds.min_episodes"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "min_steps",
                _require_nonnegative_int(self.min_steps, "hdf5 thresholds.min_steps"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, int]:
        return {
            "min_episodes": self.min_episodes,
            "min_steps": self.min_steps,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Hdf5DatasetThresholds":
        if not isinstance(value, Mapping):
            raise HarnessIOError("hdf5 thresholds must be a mapping")
        _reject_unknown_keys(value, _HDF5_DATASET_THRESHOLDS_KEYS, "hdf5 thresholds")
        return cls(
            min_episodes=value.get("min_episodes"),
            min_steps=value.get("min_steps"),
        )


@dataclass(frozen=True)
class Hdf5DatasetReport:
    """Typed, auditable validation report for one AIC HDF5 dataset file."""

    source: str
    size_bytes: int
    sha256: str
    validated_at_utc: str
    thresholds: Hdf5DatasetThresholds
    required_datasets: tuple[str, ...] = REQUIRED_HDF5_DATASET_KEYS
    missing_datasets: tuple[str, ...] = field(default_factory=tuple)
    datasets: Mapping[str, Hdf5DatasetStats] = field(default_factory=dict)
    episode_count: int | None = None
    step_count: int | None = None
    episode_lengths: tuple[int, ...] = field(default_factory=tuple)
    episode_offsets: tuple[int, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    ok: bool = False
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        validation_errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            validation_errors.append(f"schema_version must be {SCHEMA_VERSION}")
        try:
            object.__setattr__(
                self,
                "source",
                _validate_source_identity(self.source, "hdf5 report.source"),
            )
        except HarnessIOError as exc:
            validation_errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "validated_at_utc",
                _require_nonempty_text(
                    self.validated_at_utc,
                    "hdf5 report.validated_at_utc",
                ),
            )
        except HarnessIOError as exc:
            validation_errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "size_bytes",
                _require_nonnegative_int(self.size_bytes, "hdf5 report.size_bytes"),
            )
        except HarnessIOError as exc:
            validation_errors.append(str(exc))
        if not isinstance(self.sha256, str) or not _SHA256_RE.match(self.sha256):
            validation_errors.append("hdf5 report.sha256 must be 64 hexadecimal characters")

        try:
            thresholds = (
                self.thresholds
                if isinstance(self.thresholds, Hdf5DatasetThresholds)
                else Hdf5DatasetThresholds.from_dict(self.thresholds)
            )
            object.__setattr__(self, "thresholds", thresholds)
        except HarnessIOError as exc:
            validation_errors.append(str(exc))
            thresholds = None

        try:
            required_datasets = _as_nonempty_text_tuple(
                self.required_datasets,
                "hdf5 report.required_datasets",
                allow_empty=False,
            )
            _reject_duplicates(required_datasets, "hdf5 report.required_datasets")
            object.__setattr__(self, "required_datasets", required_datasets)
        except HarnessIOError as exc:
            validation_errors.append(str(exc))
            required_datasets = tuple()

        try:
            missing_datasets = _as_nonempty_text_tuple(
                self.missing_datasets,
                "hdf5 report.missing_datasets",
                allow_empty=True,
            )
            _reject_duplicates(missing_datasets, "hdf5 report.missing_datasets")
            object.__setattr__(self, "missing_datasets", missing_datasets)
        except HarnessIOError as exc:
            validation_errors.append(str(exc))
            missing_datasets = tuple()

        datasets: dict[str, Hdf5DatasetStats] = {}
        if not isinstance(self.datasets, Mapping):
            validation_errors.append("hdf5 report.datasets must be a mapping")
        else:
            for dataset_name, dataset_stats in self.datasets.items():
                if not isinstance(dataset_name, str) or not dataset_name.strip():
                    validation_errors.append(
                        "hdf5 report.datasets keys must be nonempty strings"
                    )
                    continue
                try:
                    datasets[dataset_name] = (
                        dataset_stats
                        if isinstance(dataset_stats, Hdf5DatasetStats)
                        else Hdf5DatasetStats.from_dict(dataset_stats)
                    )
                except HarnessIOError as exc:
                    validation_errors.append(
                        f"hdf5 report.datasets[{dataset_name!r}]: {exc}"
                    )
        object.__setattr__(self, "datasets", MappingProxyType(datasets))

        try:
            episode_count = _optional_nonnegative_int(
                self.episode_count,
                "hdf5 report.episode_count",
            )
            object.__setattr__(self, "episode_count", episode_count)
        except HarnessIOError as exc:
            validation_errors.append(str(exc))
            episode_count = None

        try:
            step_count = _optional_nonnegative_int(
                self.step_count,
                "hdf5 report.step_count",
            )
            object.__setattr__(self, "step_count", step_count)
        except HarnessIOError as exc:
            validation_errors.append(str(exc))
            step_count = None

        try:
            episode_lengths = _as_nonnegative_int_tuple(
                self.episode_lengths,
                "hdf5 report.episode_lengths",
            )
            object.__setattr__(self, "episode_lengths", episode_lengths)
        except HarnessIOError as exc:
            validation_errors.append(str(exc))
            episode_lengths = tuple()

        try:
            episode_offsets = _as_nonnegative_int_tuple(
                self.episode_offsets,
                "hdf5 report.episode_offsets",
            )
            object.__setattr__(self, "episode_offsets", episode_offsets)
        except HarnessIOError as exc:
            validation_errors.append(str(exc))
            episode_offsets = tuple()

        try:
            report_errors = _as_nonempty_text_tuple(
                self.errors,
                "hdf5 report.errors",
                allow_empty=True,
            )
            object.__setattr__(self, "errors", report_errors)
        except HarnessIOError as exc:
            validation_errors.append(str(exc))
            report_errors = tuple()

        if type(self.ok) is not bool:
            validation_errors.append("hdf5 report.ok must be a boolean")
        if missing_datasets and any(
            dataset_name not in required_datasets for dataset_name in missing_datasets
        ):
            validation_errors.append(
                "hdf5 report.missing_datasets must be a subset of required_datasets"
            )
        derived_missing_datasets = tuple(
            dataset_name
            for dataset_name in required_datasets
            if dataset_name not in datasets
        )
        if missing_datasets != derived_missing_datasets:
            validation_errors.append(
                "hdf5 report.missing_datasets must match required_datasets absent from datasets"
            )
        if (episode_count is None) != (step_count is None):
            validation_errors.append(
                "hdf5 report.episode_count and hdf5 report.step_count must both be set or both be null"
            )
        if episode_count is None:
            if episode_lengths:
                validation_errors.append(
                    "hdf5 report.episode_lengths must be empty when episode_count is null"
                )
            if episode_offsets:
                validation_errors.append(
                    "hdf5 report.episode_offsets must be empty when episode_count is null"
                )
        else:
            if len(episode_lengths) != episode_count:
                validation_errors.append(
                    "hdf5 report.episode_lengths length must match episode_count"
                )
            if len(episode_offsets) != episode_count:
                validation_errors.append(
                    "hdf5 report.episode_offsets length must match episode_count"
                )
            if step_count is not None and sum(episode_lengths) != step_count:
                validation_errors.append(
                    "hdf5 report.step_count must match sum of episode_lengths"
                )
            if episode_offsets != _expected_episode_offsets(episode_lengths):
                validation_errors.append(
                    "hdf5 report.episode_offsets must start at 0 and match cumulative "
                    "episode_lengths"
                )
        if self.ok and report_errors:
            validation_errors.append(
                "hdf5 report.ok must be false when hdf5 report.errors is nonempty"
            )
        if not self.ok and not report_errors:
            validation_errors.append(
                "hdf5 report.errors must be nonempty when hdf5 report.ok is false"
            )
        if self.ok and missing_datasets:
            validation_errors.append(
                "hdf5 report.missing_datasets must be empty when hdf5 report.ok is true"
            )
        if self.ok and episode_count is None:
            validation_errors.append(
                "hdf5 report.episode_count must be set when hdf5 report.ok is true"
            )
        if self.ok and step_count is None:
            validation_errors.append(
                "hdf5 report.step_count must be set when hdf5 report.ok is true"
            )
        if self.ok and thresholds is not None and episode_count is not None:
            if episode_count < thresholds.min_episodes:
                validation_errors.append(
                    "hdf5 report.episode_count must satisfy hdf5 report.thresholds.min_episodes when ok is true"
                )
        if self.ok and thresholds is not None and step_count is not None:
            if step_count < thresholds.min_steps:
                validation_errors.append(
                    "hdf5 report.step_count must satisfy hdf5 report.thresholds.min_steps when ok is true"
                )

        if validation_errors:
            raise HarnessIOError("; ".join(validation_errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "validated_at_utc": self.validated_at_utc,
            "thresholds": self.thresholds.to_dict(),
            "required_datasets": list(self.required_datasets),
            "missing_datasets": list(self.missing_datasets),
            "datasets": {
                dataset_name: dataset_stats.to_dict()
                for dataset_name, dataset_stats in self.datasets.items()
            },
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "episode_lengths": list(self.episode_lengths),
            "episode_offsets": list(self.episode_offsets),
            "errors": list(self.errors),
            "ok": self.ok,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Hdf5DatasetReport":
        if not isinstance(value, Mapping):
            raise HarnessIOError("hdf5 report must be a mapping")
        _reject_unknown_keys(value, _HDF5_DATASET_REPORT_KEYS, "hdf5 report")
        return cls(
            schema_version=value.get("schema_version"),
            source=value.get("source"),
            size_bytes=value.get("size_bytes"),
            sha256=value.get("sha256"),
            validated_at_utc=value.get("validated_at_utc"),
            thresholds=value.get("thresholds"),
            required_datasets=value.get("required_datasets", ()),
            missing_datasets=value.get("missing_datasets", ()),
            datasets=value.get("datasets", {}),
            episode_count=value.get("episode_count"),
            step_count=value.get("step_count"),
            episode_lengths=value.get("episode_lengths", ()),
            episode_offsets=value.get("episode_offsets", ()),
            errors=value.get("errors", ()),
            ok=value.get("ok"),
        )


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


def _validate_source_identity(value: Any, field_name: str) -> str:
    source = _require_nonempty_text(value, field_name)
    if _is_uri_source(source):
        parsed = urlparse(source)
        if parsed.scheme == "file":
            if not parsed.path:
                raise HarnessIOError(f"{field_name} must include a file path")
        elif not parsed.netloc:
            raise HarnessIOError(f"{field_name} must include a network location")
        return source
    if not _is_absolute_local_source(source):
        raise HarnessIOError(f"{field_name} must be an absolute local path or uri")
    return source


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise HarnessIOError(f"{field_name} must be a nonnegative integer")
    return value


def _optional_nonnegative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _require_nonnegative_int(value, field_name)


def _as_nonempty_text_tuple(
    values: Any,
    field_name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    items = tuple(_require_nonempty_text(value, field_name) for value in values)
    if not items and not allow_empty:
        raise HarnessIOError(f"{field_name} must not be empty")
    return items


def _as_nonnegative_int_tuple(values: Any, field_name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    return tuple(_require_nonnegative_int(value, field_name) for value in values)


def _reject_duplicates(values: tuple[str, ...], field_name: str) -> None:
    if len(set(values)) != len(values):
        raise HarnessIOError(f"{field_name} must not contain duplicates")


def _expected_episode_offsets(episode_lengths: tuple[int, ...]) -> tuple[int, ...]:
    offsets: list[int] = []
    running_offset = 0
    for length in episode_lengths:
        offsets.append(running_offset)
        running_offset += length
    return tuple(offsets)


def _is_uri_source(source: str) -> bool:
    return bool(_URI_SOURCE_RE.match(source))


def _is_absolute_local_source(source: str) -> bool:
    return Path(source).is_absolute() or _is_windows_absolute_local_source(source)


def _is_windows_absolute_local_source(source: str) -> bool:
    return PureWindowsPath(source).is_absolute()
