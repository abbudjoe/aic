"""Reducer slice for AIC demonstration HDF5 datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Mapping

from aic_signal_harness.artifacts import HarnessIOError, sha256_file
from aic_signal_harness.hdf5_dataset import (
    Hdf5DatasetReport,
    Hdf5DatasetStats,
    Hdf5DatasetThresholds,
    PER_STEP_HDF5_DATASET_KEYS,
    REQUIRED_HDF5_DATASET_KEYS,
    _is_uri_source,
)
from aic_signal_harness.schemas import ArtifactRef, utc_now_iso


@dataclass(frozen=True)
class Hdf5DatasetReduction:
    """Typed reduction binding an HDF5 dataset report to its artifact identity."""

    artifact: ArtifactRef
    report: Hdf5DatasetReport

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.artifact, ArtifactRef):
            errors.append("hdf5 reduction artifact must be an ArtifactRef")
        if not isinstance(self.report, Hdf5DatasetReport):
            errors.append("hdf5 reduction report must be an Hdf5DatasetReport")
        if errors:
            raise HarnessIOError("; ".join(errors))
        if self.artifact.kind != "hdf5_dataset":
            errors.append("hdf5 reduction artifact.kind must be 'hdf5_dataset'")
        if self.artifact.sha256 is None:
            errors.append("hdf5 reduction artifact.sha256 must be set")
        errors.extend(_identity_contract_errors(self.artifact, self.report.source))
        if self.artifact.sha256 != self.report.sha256:
            errors.append("hdf5 reduction artifact.sha256 must match report.sha256")
        if errors:
            raise HarnessIOError("; ".join(errors))


def validate_hdf5_dataset(
    path: str | Path,
    *,
    min_episodes: int = 1,
    min_steps: int = 1,
    validated_at_utc: str | None = None,
) -> Hdf5DatasetReport:
    """Validate an AIC demonstration HDF5 file and return a typed report."""

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise HarnessIOError("validate_hdf5_dataset requires h5py") from exc

    dataset_path = _resolve_existing_path(path)
    thresholds = Hdf5DatasetThresholds(
        min_episodes=min_episodes,
        min_steps=min_steps,
    )
    validation_started = _stat_snapshot(dataset_path)

    with h5py.File(dataset_path, "r") as handle:
        report = _build_hdf5_dataset_report(
            handle=handle,
            source=str(dataset_path),
            size_bytes=validation_started.st_size,
            sha256=sha256_file(dataset_path),
            validated_at_utc=utc_now_iso() if validated_at_utc is None else validated_at_utc,
            thresholds=thresholds,
        )

    validation_finished = _stat_snapshot(dataset_path)
    _assert_same_file_identity(
        dataset_path,
        validation_started,
        validation_finished,
        "dataset changed during validation",
    )
    return report


def reduce_hdf5_dataset(
    path: str | Path,
    *,
    min_episodes: int = 1,
    min_steps: int = 1,
    uri: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    validated_at_utc: str | None = None,
) -> Hdf5DatasetReduction:
    """Reduce an HDF5 dataset into a typed report and bound artifact reference."""

    report = validate_hdf5_dataset(
        path,
        min_episodes=min_episodes,
        min_steps=min_steps,
        validated_at_utc=validated_at_utc,
    )
    artifact_provenance = _with_declared_uri_provenance(provenance, uri)
    return Hdf5DatasetReduction(
        artifact=ArtifactRef(
            kind="hdf5_dataset",
            path=report.source,
            sha256=report.sha256,
            provenance=artifact_provenance,
        ),
        report=report,
    )


def _build_hdf5_dataset_report(
    *,
    handle: Any,
    source: str,
    size_bytes: int,
    sha256: str,
    validated_at_utc: str,
    thresholds: Hdf5DatasetThresholds,
) -> Hdf5DatasetReport:
    errors: list[str] = []
    present_keys = tuple(sorted(str(key) for key in handle.keys()))
    datasets: dict[str, Hdf5DatasetStats] = {}
    for dataset_name in present_keys:
        entry = handle[dataset_name]
        if not _looks_like_dataset(entry):
            errors.append(
                f"{dataset_name} must be a dataset, got {entry.__class__.__name__}"
            )
            continue
        datasets[dataset_name] = Hdf5DatasetStats(
            shape=tuple(int(dimension) for dimension in entry.shape),
            dtype=str(entry.dtype),
        )
    missing_datasets = tuple(
        dataset_name
        for dataset_name in REQUIRED_HDF5_DATASET_KEYS
        if dataset_name not in datasets
    )
    if missing_datasets:
        errors.append(
            "missing required datasets: " + ", ".join(missing_datasets)
        )

    episode_count: int | None = None
    step_count: int | None = None
    episode_lengths: tuple[int, ...] = ()
    episode_offsets: tuple[int, ...] = ()

    if not missing_datasets:
        ep_len = _read_1d_int_dataset(handle, "ep_len", errors)
        ep_offset = _read_1d_int_dataset(handle, "ep_offset", errors)

        if ep_len is not None and ep_offset is not None:
            episode_count = len(ep_len)
            step_count = sum(ep_len)
            episode_lengths = tuple(ep_len)
            expected_offsets = _expected_episode_offsets(episode_lengths)
            episode_offsets = expected_offsets

            if episode_count < thresholds.min_episodes:
                errors.append(
                    f"episode_count {episode_count} < required {thresholds.min_episodes}"
                )
            if step_count < thresholds.min_steps:
                errors.append(f"step_count {step_count} < required {thresholds.min_steps}")
            if len(ep_offset) != episode_count:
                errors.append(
                    "ep_offset length "
                    f"{len(ep_offset)} does not match ep_len length {episode_count}"
                )
            raw_episode_offsets = tuple(ep_offset)
            if raw_episode_offsets != expected_offsets:
                errors.append(
                    "ep_offset must start at 0 and match cumulative episode starts "
                    f"implied by ep_len; expected {expected_offsets}, got {raw_episode_offsets}"
                )

            for dataset_name in PER_STEP_HDF5_DATASET_KEYS:
                shape = handle[dataset_name].shape
                if len(shape) < 1 or int(shape[0]) != step_count:
                    errors.append(
                        f"{dataset_name} first dimension {shape[:1]} does not match steps {step_count}"
                    )

            for dataset_name in ("pixels", "left_pixels", "right_pixels"):
                shape = handle[dataset_name].shape
                dtype = str(handle[dataset_name].dtype)
                if len(shape) != 4 or int(shape[-1]) != 3:
                    errors.append(
                        f"{dataset_name} must be THWC images with 3 channels, got {shape}"
                    )
                if dtype != "uint8":
                    errors.append(f"{dataset_name} must be uint8, got {dtype}")

            action_shape = handle["action"].shape
            if len(action_shape) != 2 or int(action_shape[1]) != 6:
                errors.append(f"action must have shape (T, 6), got {action_shape}")

            for dataset_name in ("state", "proprio"):
                shape = handle[dataset_name].shape
                if len(shape) != 2 or int(shape[1]) != 32:
                    errors.append(f"{dataset_name} must have shape (T, 32), got {shape}")

    return Hdf5DatasetReport(
        source=source,
        size_bytes=size_bytes,
        sha256=sha256,
        validated_at_utc=validated_at_utc,
        thresholds=thresholds,
        required_datasets=REQUIRED_HDF5_DATASET_KEYS,
        missing_datasets=missing_datasets,
        datasets=datasets,
        episode_count=episode_count,
        step_count=step_count,
        episode_lengths=episode_lengths,
        episode_offsets=episode_offsets,
        errors=tuple(errors),
        ok=not errors,
    )


def _read_1d_int_dataset(
    handle: Any,
    dataset_name: str,
    errors: list[str],
) -> list[int] | None:
    dataset = handle[dataset_name]
    shape = dataset.shape
    dtype = str(dataset.dtype)
    if len(shape) != 1:
        errors.append(f"{dataset_name} must be a 1D integer dataset, got shape {shape}")
        return None
    if "int" not in dtype:
        errors.append(f"{dataset_name} must be an integer dataset, got {dtype}")
        return None

    raw_values = dataset[()]
    if hasattr(raw_values, "tolist"):
        values = raw_values.tolist()
    else:
        values = list(raw_values)
    if not isinstance(values, list):
        values = [values]

    normalized: list[int] = []
    for index, value in enumerate(values):
        try:
            normalized_value = int(value)
        except (TypeError, ValueError) as exc:
            errors.append(
                f"{dataset_name}[{index}] could not be parsed as an integer: {value!r}"
            )
            return None
        if normalized_value < 0:
            errors.append(f"{dataset_name}[{index}] must be nonnegative, got {normalized_value}")
            return None
        normalized.append(normalized_value)
    return normalized


def _resolve_existing_path(path: str | Path) -> Path:
    dataset_path = Path(path).expanduser()
    try:
        return dataset_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HarnessIOError(f"dataset not found: {dataset_path}") from exc
    except OSError as exc:
        raise HarnessIOError(f"failed to resolve dataset path {dataset_path}: {exc}") from exc


def _identity_contract_errors(artifact: ArtifactRef, source: str) -> list[str]:
    if _is_uri_source(source):
        errors: list[str] = []
        if artifact.uri != source:
            errors.append("hdf5 reduction artifact.uri must match report.source when report.source is a uri")
        if artifact.path is not None:
            errors.append("hdf5 reduction artifact.path must be unset when report.source is a uri")
        return errors

    errors = []
    if artifact.path != source:
        errors.append(
            "hdf5 reduction artifact.path must match report.source when report.source is a local path"
        )
    if artifact.uri is not None:
        errors.append(
            "hdf5 reduction artifact.uri must be unset when report.source is a local path"
        )
    return errors


def _with_declared_uri_provenance(
    provenance: Mapping[str, Any] | None,
    uri: str | None,
) -> Mapping[str, Any]:
    merged = {} if provenance is None else dict(provenance)
    if uri is None:
        return merged
    existing = merged.get("declared_uri")
    if existing is not None and existing != uri:
        raise HarnessIOError("hdf5 reduction provenance.declared_uri must match reducer uri")
    merged["declared_uri"] = uri
    return merged


def _looks_like_dataset(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _expected_episode_offsets(episode_lengths: tuple[int, ...]) -> tuple[int, ...]:
    offsets: list[int] = []
    running_offset = 0
    for length in episode_lengths:
        offsets.append(running_offset)
        running_offset += length
    return tuple(offsets)


def _stat_snapshot(path: Path) -> os.stat_result:
    try:
        return path.stat()
    except OSError as exc:
        raise HarnessIOError(f"failed to stat dataset {path}: {exc}") from exc


def _assert_same_file_identity(
    path: Path,
    before: os.stat_result,
    after: os.stat_result,
    message: str,
) -> None:
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise HarnessIOError(f"{message}: {path}")
