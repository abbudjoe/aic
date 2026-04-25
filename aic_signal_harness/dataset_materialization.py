"""Materialize offline training datasets into trainer-consumable artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar, cast

from aic_signal_harness.artifacts import (
    HarnessIOError,
    local_artifact_path,
    read_json,
    sha256_file,
    write_json,
)
from aic_signal_harness.reducers.policy_training import read_training_dataset_report_artifact
from aic_signal_harness.schemas import ArtifactRef, SCHEMA_VERSION, SchemaValidationError
from aic_signal_harness.training_dataset import (
    TrainingDatasetExample,
    TrainingDatasetReport,
    training_dataset_fingerprint_sha256,
)


TRAINING_DATASET_MATERIALIZATION_KIND = "training_dataset_materialization_report"
TRAINER_DATASET_JSONL_KIND = "trainer_dataset_jsonl"

_PRODUCER = "aic_signal_harness.dataset_materialization"
_MATERIALIZE_JSONL_DERIVATION = "materialize_training_dataset_jsonl"
_WRITE_REPORT_DERIVATION = "write_training_dataset_materialization_report"

_MATERIALIZATION_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "generated_at_utc",
        "format",
        "materializer_id",
        "source_training_dataset_report",
        "materialized_dataset",
        "dataset_fingerprint_sha256",
        "example_count",
        "split_counts",
        "leakage_summary",
        "target_counts",
        "offline_only",
        "runtime_allowed",
        "consumable_by_policy_runtime",
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


class TrainingDatasetMaterializationFormat(_StrEnum):
    """Trainer dataset byte format owned by the neutral harness."""

    jsonl = "jsonl"


@dataclass(frozen=True)
class TrainingDatasetMaterializationReport:
    """Typed evidence for one materialized trainer dataset artifact."""

    run_id: str
    generated_at_utc: str
    format: TrainingDatasetMaterializationFormat
    materializer_id: str
    source_training_dataset_report: ArtifactRef
    materialized_dataset: ArtifactRef
    dataset_fingerprint_sha256: str
    example_count: int
    split_counts: Mapping[str, int]
    leakage_summary: Mapping[str, int]
    target_counts: Mapping[str, int]
    offline_only: bool = True
    runtime_allowed: bool = False
    consumable_by_policy_runtime: bool = False
    ok: bool = True
    errors: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        for field_name in ("run_id", "generated_at_utc", "materializer_id"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"training dataset materialization.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "format",
                TrainingDatasetMaterializationFormat.parse(self.format),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("source_training_dataset_report", "materialized_dataset"):
            if not isinstance(getattr(self, field_name), ArtifactRef):
                try:
                    object.__setattr__(
                        self,
                        field_name,
                        ArtifactRef.from_dict(getattr(self, field_name)),
                    )
                except SchemaValidationError as exc:
                    errors.extend(exc.errors)
        try:
            object.__setattr__(
                self,
                "dataset_fingerprint_sha256",
                _require_sha256(
                    self.dataset_fingerprint_sha256,
                    "training dataset materialization.dataset_fingerprint_sha256",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "example_count",
                _require_positive_int(
                    self.example_count,
                    "training dataset materialization.example_count",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("split_counts", "leakage_summary", "target_counts"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _count_mapping(
                        getattr(self, field_name),
                        f"training dataset materialization.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        for field_name in (
            "offline_only",
            "runtime_allowed",
            "consumable_by_policy_runtime",
            "ok",
        ):
            if type(getattr(self, field_name)) is not bool:
                errors.append(f"training dataset materialization.{field_name} must be a boolean")
        try:
            object.__setattr__(
                self,
                "errors",
                _as_text_sequence(self.errors, "training dataset materialization.errors"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "notes",
                _as_text_sequence(self.notes, "training dataset materialization.notes"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not errors:
            errors.extend(_materialization_report_consistency_errors(self))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generated_at_utc": self.generated_at_utc,
            "format": self.format.value,
            "materializer_id": self.materializer_id,
            "source_training_dataset_report": self.source_training_dataset_report.to_dict(),
            "materialized_dataset": self.materialized_dataset.to_dict(),
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "example_count": self.example_count,
            "split_counts": dict(self.split_counts),
            "leakage_summary": dict(self.leakage_summary),
            "target_counts": dict(self.target_counts),
            "offline_only": self.offline_only,
            "runtime_allowed": self.runtime_allowed,
            "consumable_by_policy_runtime": self.consumable_by_policy_runtime,
            "ok": self.ok,
            "errors": list(self.errors),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingDatasetMaterializationReport":
        if not isinstance(value, Mapping):
            raise HarnessIOError("training dataset materialization report must be a mapping")
        _reject_unknown_keys(
            value,
            _MATERIALIZATION_REPORT_KEYS,
            "training dataset materialization report",
        )
        _require_keys(
            value,
            _MATERIALIZATION_REPORT_KEYS - {"notes"},
            "training dataset materialization report",
        )
        return cls(
            schema_version=cast(Any, value.get("schema_version")),
            run_id=cast(Any, value.get("run_id")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            format=cast(Any, value.get("format")),
            materializer_id=cast(Any, value.get("materializer_id")),
            source_training_dataset_report=cast(Any, value.get("source_training_dataset_report")),
            materialized_dataset=cast(Any, value.get("materialized_dataset")),
            dataset_fingerprint_sha256=cast(Any, value.get("dataset_fingerprint_sha256")),
            example_count=cast(Any, value.get("example_count")),
            split_counts=cast(Any, value.get("split_counts")),
            leakage_summary=cast(Any, value.get("leakage_summary")),
            target_counts=cast(Any, value.get("target_counts")),
            offline_only=cast(Any, value.get("offline_only")),
            runtime_allowed=cast(Any, value.get("runtime_allowed")),
            consumable_by_policy_runtime=cast(Any, value.get("consumable_by_policy_runtime")),
            ok=cast(Any, value.get("ok")),
            errors=cast(Any, value.get("errors")),
            notes=cast(Any, value.get("notes", ())),
        )


def materialize_training_dataset_jsonl(
    *,
    source_training_dataset_report: ArtifactRef | Mapping[str, Any],
    output_path: str | Path,
    generated_at_utc: str,
    materializer_id: str = "aic_signal_harness.dataset_materialization.jsonl.v1",
    notes: tuple[str, ...] = (),
    overwrite: bool = False,
) -> TrainingDatasetMaterializationReport:
    """Write canonical JSONL bytes for a validated offline training dataset."""

    source_artifact = (
        source_training_dataset_report
        if isinstance(source_training_dataset_report, ArtifactRef)
        else ArtifactRef.from_dict(source_training_dataset_report)
    )
    _require_nonempty_text(generated_at_utc, "training dataset materialization.generated_at_utc")
    materializer = _require_nonempty_text(
        materializer_id,
        "training dataset materialization.materializer_id",
    )
    source_report = read_training_dataset_report_artifact(source_artifact)
    dataset_path = Path(output_path).expanduser().resolve(strict=False)
    _reject_output_artifact_aliases(
        dataset_path,
        protected_paths={
            "source_training_dataset_report": _local_existing_artifact_file(
                source_artifact,
                "source_training_dataset_report",
            ),
            "source_training_signal_report": _local_existing_artifact_file(
                source_report.source_training_signal_report,
                "source_training_signal_report",
            ),
        },
        field_name="training dataset materialization.output_path",
    )
    _write_bytes_atomic(
        dataset_path,
        _canonical_jsonl_bytes(source_report.examples),
        overwrite=overwrite,
    )
    materialized_dataset = ArtifactRef(
        kind=TRAINER_DATASET_JSONL_KIND,
        path=str(dataset_path),
        sha256=sha256_file(dataset_path),
        provenance={
            "producer": _PRODUCER,
            "derivation": _MATERIALIZE_JSONL_DERIVATION,
            "run_id": source_report.run_id,
            "format": TrainingDatasetMaterializationFormat.jsonl.value,
            "materializer_id": materializer,
            "source_training_dataset_report_sha256": source_artifact.sha256,
            "dataset_fingerprint_sha256": source_report.dataset_fingerprint_sha256,
        },
    )
    return TrainingDatasetMaterializationReport(
        run_id=source_report.run_id,
        generated_at_utc=generated_at_utc,
        format=TrainingDatasetMaterializationFormat.jsonl,
        materializer_id=materializer,
        source_training_dataset_report=source_artifact,
        materialized_dataset=materialized_dataset,
        dataset_fingerprint_sha256=source_report.dataset_fingerprint_sha256,
        example_count=source_report.example_count,
        split_counts=_counts(example.split.value for example in source_report.examples),
        leakage_summary=dict(source_report.signals_by_leakage_class),
        target_counts=dict(source_report.signals_by_target),
        offline_only=True,
        runtime_allowed=False,
        consumable_by_policy_runtime=False,
        ok=True,
        errors=(),
        notes=notes
        + (
            "Materialized dataset is trainer input only; it is not legal live policy runtime input.",
        ),
    )


def write_training_dataset_materialization_report(
    path: str | Path,
    report: TrainingDatasetMaterializationReport | Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> ArtifactRef:
    """Write a dataset materialization report and return its artifact reference."""

    typed_report = (
        report
        if isinstance(report, TrainingDatasetMaterializationReport)
        else TrainingDatasetMaterializationReport.from_dict(report)
    )
    output_path = Path(path).expanduser().resolve(strict=False)
    source_report = read_training_dataset_report_artifact(
        typed_report.source_training_dataset_report
    )
    _reject_output_artifact_aliases(
        output_path,
        protected_paths={
            "source_training_dataset_report": _local_existing_artifact_file(
                typed_report.source_training_dataset_report,
                "source_training_dataset_report",
            ),
            "source_training_signal_report": _local_existing_artifact_file(
                source_report.source_training_signal_report,
                "source_training_signal_report",
            ),
            "materialized_dataset": _local_existing_artifact_file(
                typed_report.materialized_dataset,
                "materialized_dataset",
            ),
        },
        field_name="training dataset materialization report.output_path",
    )
    _revalidate_materialization_report_current_bytes(typed_report)
    write_json(output_path, typed_report.to_dict(), overwrite=overwrite)
    return ArtifactRef(
        kind=TRAINING_DATASET_MATERIALIZATION_KIND,
        path=str(output_path),
        sha256=sha256_file(output_path),
        provenance={
            "producer": _PRODUCER,
            "derivation": _WRITE_REPORT_DERIVATION,
            "run_id": typed_report.run_id,
            "format": typed_report.format.value,
            "materialized_dataset_sha256": typed_report.materialized_dataset.sha256,
            "source_training_dataset_report_sha256": (
                typed_report.source_training_dataset_report.sha256
            ),
        },
    )


def read_training_dataset_materialization_report_artifact(
    artifact: ArtifactRef | Mapping[str, Any],
) -> TrainingDatasetMaterializationReport:
    """Read and validate a local dataset-materialization report artifact."""

    typed_artifact = artifact if isinstance(artifact, ArtifactRef) else ArtifactRef.from_dict(artifact)
    errors: list[str] = []
    if typed_artifact.kind != TRAINING_DATASET_MATERIALIZATION_KIND:
        errors.append(
            "training dataset materialization report artifact.kind must be "
            f"{TRAINING_DATASET_MATERIALIZATION_KIND!r}"
        )
    if typed_artifact.sha256 is None:
        errors.append("training dataset materialization report artifact.sha256 must be set")
    for provenance_key in (
        "producer",
        "derivation",
        "run_id",
        "format",
        "materialized_dataset_sha256",
        "source_training_dataset_report_sha256",
    ):
        if provenance_key not in typed_artifact.provenance:
            errors.append(
                "training dataset materialization report artifact provenance must set "
                f"{provenance_key}"
            )
        else:
            try:
                _require_nonempty_text(
                    typed_artifact.provenance.get(provenance_key),
                    "training dataset materialization report artifact provenance."
                    + provenance_key,
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
    if typed_artifact.provenance.get("producer") != _PRODUCER:
        errors.append(
            "training dataset materialization report artifact provenance producer must be "
            f"{_PRODUCER!r}"
        )
    if typed_artifact.provenance.get("derivation") != _WRITE_REPORT_DERIVATION:
        errors.append(
            "training dataset materialization report artifact provenance derivation must be "
            f"{_WRITE_REPORT_DERIVATION!r}"
        )
    try:
        report_path = local_artifact_path(
            path=typed_artifact.path,
            uri=typed_artifact.uri,
            field_name="training dataset materialization report artifact",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        report_path = None
    if report_path is None:
        errors.append("training dataset materialization report artifact must be local byte-verifiable")
    elif not report_path.exists():
        errors.append("training dataset materialization report artifact.path must exist")
    elif not report_path.is_file():
        errors.append("training dataset materialization report artifact.path must be a file")
    elif typed_artifact.sha256 is not None and sha256_file(report_path) != typed_artifact.sha256:
        errors.append("training dataset materialization report artifact.sha256 must match path")
    if errors:
        raise HarnessIOError("; ".join(errors))
    assert report_path is not None
    report = TrainingDatasetMaterializationReport.from_dict(read_json(report_path))
    if typed_artifact.provenance.get("run_id") != report.run_id:
        raise HarnessIOError(
            "training dataset materialization report artifact provenance run_id must match"
        )
    if typed_artifact.provenance.get("format") != report.format.value:
        raise HarnessIOError(
            "training dataset materialization report artifact provenance format must match"
        )
    if (
        typed_artifact.provenance.get("materialized_dataset_sha256")
        != report.materialized_dataset.sha256
    ):
        raise HarnessIOError(
            "training dataset materialization report artifact provenance "
            "materialized_dataset_sha256 must match"
        )
    if (
        typed_artifact.provenance.get("source_training_dataset_report_sha256")
        != report.source_training_dataset_report.sha256
    ):
        raise HarnessIOError(
            "training dataset materialization report artifact provenance "
            "source_training_dataset_report_sha256 must match"
        )
    return report


def _materialization_report_consistency_errors(
    report: TrainingDatasetMaterializationReport,
) -> list[str]:
    errors: list[str] = []
    if report.offline_only is not True:
        errors.append("training dataset materialization.offline_only must be true")
    if report.runtime_allowed is not False:
        errors.append("training dataset materialization.runtime_allowed must be false")
    if report.consumable_by_policy_runtime is not False:
        errors.append(
            "training dataset materialization.consumable_by_policy_runtime must be false"
        )
    if report.ok is not True:
        errors.append("training dataset materialization.ok must be true")
    if report.errors:
        errors.append("training dataset materialization.errors must be empty when ok=true")
    source_report: TrainingDatasetReport | None
    try:
        source_report = read_training_dataset_report_artifact(report.source_training_dataset_report)
    except HarnessIOError as exc:
        errors.append("source_training_dataset_report must validate: " + str(exc))
        source_report = None
    if source_report is not None:
        errors.extend(_source_report_binding_errors(report, source_report))
        errors.extend(_materialized_dataset_artifact_errors(report, source_report))
    return errors


def _source_report_binding_errors(
    report: TrainingDatasetMaterializationReport,
    source_report: TrainingDatasetReport,
) -> list[str]:
    errors: list[str] = []
    if report.run_id != source_report.run_id:
        errors.append("training dataset materialization.run_id must match source report run_id")
    if report.dataset_fingerprint_sha256 != source_report.dataset_fingerprint_sha256:
        errors.append(
            "training dataset materialization.dataset_fingerprint_sha256 must match source report"
        )
    if report.example_count != source_report.example_count:
        errors.append("training dataset materialization.example_count must match source report")
    expected_splits = _counts(example.split.value for example in source_report.examples)
    if dict(report.split_counts) != expected_splits:
        errors.append("training dataset materialization.split_counts must match source report")
    if dict(report.leakage_summary) != dict(source_report.signals_by_leakage_class):
        errors.append("training dataset materialization.leakage_summary must match source report")
    if dict(report.target_counts) != dict(source_report.signals_by_target):
        errors.append("training dataset materialization.target_counts must match source report")
    return errors


def _materialized_dataset_artifact_errors(
    report: TrainingDatasetMaterializationReport,
    source_report: TrainingDatasetReport,
) -> list[str]:
    errors: list[str] = []
    dataset = report.materialized_dataset
    if dataset.kind != TRAINER_DATASET_JSONL_KIND:
        errors.append(f"materialized_dataset.kind must be {TRAINER_DATASET_JSONL_KIND!r}")
    if dataset.sha256 is None:
        errors.append("materialized_dataset.sha256 must be set")
    for provenance_key in (
        "producer",
        "derivation",
        "run_id",
        "format",
        "materializer_id",
        "source_training_dataset_report_sha256",
        "dataset_fingerprint_sha256",
    ):
        if provenance_key not in dataset.provenance:
            errors.append(f"materialized_dataset provenance must set {provenance_key}")
        else:
            try:
                _require_nonempty_text(
                    dataset.provenance.get(provenance_key),
                    f"materialized_dataset provenance.{provenance_key}",
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
    if dataset.provenance.get("producer") != _PRODUCER:
        errors.append(f"materialized_dataset provenance producer must be {_PRODUCER!r}")
    if dataset.provenance.get("derivation") != _MATERIALIZE_JSONL_DERIVATION:
        errors.append(
            "materialized_dataset provenance derivation must be "
            f"{_MATERIALIZE_JSONL_DERIVATION!r}"
        )
    if dataset.provenance.get("run_id") != source_report.run_id:
        errors.append("materialized_dataset provenance run_id must match source report")
    if dataset.provenance.get("format") != report.format.value:
        errors.append("materialized_dataset provenance format must match report")
    if dataset.provenance.get("materializer_id") != report.materializer_id:
        errors.append("materialized_dataset provenance materializer_id must match report")
    if (
        dataset.provenance.get("source_training_dataset_report_sha256")
        != report.source_training_dataset_report.sha256
    ):
        errors.append(
            "materialized_dataset provenance source_training_dataset_report_sha256 "
            "must match source artifact"
        )
    if dataset.provenance.get("dataset_fingerprint_sha256") != source_report.dataset_fingerprint_sha256:
        errors.append("materialized_dataset provenance dataset_fingerprint_sha256 must match source report")
    try:
        dataset_path = local_artifact_path(
            path=dataset.path,
            uri=dataset.uri,
            field_name="materialized_dataset",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        dataset_path = None
    if dataset_path is None:
        errors.append("materialized_dataset must be local byte-verifiable")
    elif not dataset_path.exists():
        errors.append("materialized_dataset.path must exist")
    elif not dataset_path.is_file():
        errors.append("materialized_dataset.path must be a file")
    elif dataset.sha256 is not None:
        if sha256_file(dataset_path) != dataset.sha256:
            errors.append("materialized_dataset.sha256 must match path")
        else:
            errors.extend(_canonical_jsonl_file_errors(dataset_path, source_report))
    return errors


def _canonical_jsonl_file_errors(
    path: Path,
    source_report: TrainingDatasetReport,
) -> list[str]:
    errors: list[str] = []
    try:
        actual_bytes = path.read_bytes()
    except OSError as exc:
        return [f"materialized_dataset could not be read: {exc}"]
    expected_bytes = _canonical_jsonl_bytes(source_report.examples)
    if actual_bytes != expected_bytes:
        errors.append(
            "materialized_dataset bytes must match canonical JSONL from source_training_dataset_report"
        )
    try:
        examples = _read_jsonl_examples(path)
    except HarnessIOError as exc:
        errors.append(str(exc))
    else:
        if examples != source_report.examples:
            errors.append(
                "materialized_dataset examples must match source_training_dataset_report examples"
            )
        if training_dataset_fingerprint_sha256(examples) != source_report.dataset_fingerprint_sha256:
            errors.append("materialized_dataset fingerprint must match source report")
    return errors


def _read_jsonl_examples(path: Path) -> tuple[TrainingDatasetExample, ...]:
    examples: list[TrainingDatasetExample] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise HarnessIOError(
                        f"materialized_dataset line {line_number} must end with newline"
                    )
                stripped = line.strip()
                if not stripped:
                    raise HarnessIOError(
                        f"materialized_dataset line {line_number} must not be blank"
                    )
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise HarnessIOError(
                        f"materialized_dataset line {line_number} is invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise HarnessIOError(
                        f"materialized_dataset line {line_number} must be a JSON object"
                    )
                examples.append(TrainingDatasetExample.from_dict(value))
    except OSError as exc:
        raise HarnessIOError(f"materialized_dataset could not be read: {exc}") from exc
    return tuple(examples)


def _canonical_jsonl_bytes(examples: tuple[TrainingDatasetExample, ...]) -> bytes:
    lines = [
        json.dumps(
            example.to_dict(),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for example in examples
    ]
    return ("".join(line + "\n" for line in lines)).encode("utf-8")


def _write_bytes_atomic(path: Path, data: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise HarnessIOError(f"{path} already exists; pass overwrite=True to replace it")
    tmp_name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(data)
        if overwrite:
            os.replace(tmp_name, path)
        else:
            try:
                os.link(tmp_name, path)
            except FileExistsError as exc:
                raise HarnessIOError(
                    f"{path} already exists; pass overwrite=True to replace it"
                ) from exc
    except OSError as exc:
        raise HarnessIOError(f"failed to write materialized dataset {path}: {exc}") from exc
    finally:
        if tmp_name is not None:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


def _revalidate_materialization_report_current_bytes(
    report: TrainingDatasetMaterializationReport,
) -> None:
    errors = _materialization_report_consistency_errors(report)
    if errors:
        raise HarnessIOError("; ".join(errors))


def _local_existing_artifact_file(artifact: ArtifactRef, field_name: str) -> Path:
    path = local_artifact_path(
        path=artifact.path,
        uri=artifact.uri,
        field_name=field_name,
    )
    if path is None:
        raise HarnessIOError(f"{field_name} must be local byte-verifiable")
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise HarnessIOError(f"{field_name}.path must exist") from exc
    except OSError as exc:
        raise HarnessIOError(f"{field_name}.path cannot be resolved: {exc}") from exc
    if not resolved.is_file():
        raise HarnessIOError(f"{field_name}.path must be a file")
    return resolved


def _reject_output_artifact_aliases(
    output_path: Path,
    *,
    protected_paths: Mapping[str, Path],
    field_name: str,
) -> None:
    try:
        output_resolved = output_path.expanduser().resolve(strict=False)
    except OSError as exc:
        raise HarnessIOError(f"{field_name} cannot be resolved: {exc}") from exc
    for protected_name, protected_path in protected_paths.items():
        protected_resolved = protected_path.expanduser().resolve(strict=True)
        aliases = output_resolved == protected_resolved
        if not aliases and output_path.exists():
            try:
                aliases = output_path.samefile(protected_resolved)
            except OSError:
                aliases = False
        if aliases:
            raise HarnessIOError(
                f"{field_name} must not alias protected artifact {protected_name}"
            )


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessIOError(f"{field_name} must be a nonempty string")
    return value.strip()


def _require_sha256(value: Any, field_name: str) -> str:
    text = _require_nonempty_text(value, field_name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise HarnessIOError(f"{field_name} must be a lowercase sha256 hex digest")
    return text


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise HarnessIOError(f"{field_name} must be an integer")
    if value < 1:
        raise HarnessIOError(f"{field_name} must be >= 1")
    return value


def _count_mapping(value: Any, field_name: str) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    normalized: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key.strip():
            raise HarnessIOError(f"{field_name} keys must be nonempty strings")
        if isinstance(count, bool) or type(count) is not int:
            raise HarnessIOError(f"{field_name}.{key} must be an integer")
        if count < 0:
            raise HarnessIOError(f"{field_name}.{key} must be >= 0")
        normalized[key] = count
    return dict(sorted(normalized.items()))


def _as_text_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    return tuple(
        _require_nonempty_text(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


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


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))
