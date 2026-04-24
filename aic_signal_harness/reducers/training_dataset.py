"""Derive offline training datasets from typed training-signal reports."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, cast

from aic_signal_harness.artifacts import HarnessIOError, local_artifact_path, read_json, sha256_file
from aic_signal_harness.schemas import ArtifactRef
from aic_signal_harness.training_dataset import (
    TrainingDatasetReport,
    training_dataset_fingerprint_sha256,
    training_signal_to_dataset_example,
)
from aic_signal_harness.training_signal import TrainingSignalReport


def derive_training_dataset_report(
    training_signal_report: TrainingSignalReport | Mapping[str, Any],
    *,
    source_training_signal_report: ArtifactRef,
) -> TrainingDatasetReport:
    """Select offline training examples from a byte-bound training-signal report."""

    typed_report = (
        training_signal_report
        if isinstance(training_signal_report, TrainingSignalReport)
        else TrainingSignalReport.from_dict(training_signal_report)
    )
    source_sha256 = _validate_source_training_signal_report(
        typed_report,
        source_training_signal_report=source_training_signal_report,
    )

    examples = tuple(
        training_signal_to_dataset_example(
            signal,
            source_training_signal_report_sha256=source_sha256,
        )
        for signal in typed_report.signals
    )
    errors = tuple() if examples else ("no selected training signals",)
    return TrainingDatasetReport(
        run_id=typed_report.run_id,
        generated_at_utc=typed_report.generated_at_utc,
        source_training_signal_report=source_training_signal_report,
        dataset_fingerprint_sha256=training_dataset_fingerprint_sha256(examples),
        example_count=len(examples),
        selected_signal_ids=tuple(example.source_signal_id for example in examples),
        signals_by_kind=_counts(example.kind.value for example in examples),
        signals_by_leakage_class=_counts(example.leakage_class.value for example in examples),
        signals_by_target=_counts(example.target for example in examples),
        examples=examples,
        ok=not errors,
        errors=errors,
        notes=(
            "Training dataset examples are offline-only and must not enter the live policy runtime.",
        ),
    )


def _validate_source_training_signal_report(
    training_signal_report: TrainingSignalReport,
    *,
    source_training_signal_report: ArtifactRef,
) -> str:
    errors: list[str] = []
    if source_training_signal_report.kind != "training_signal_report":
        errors.append("source_training_signal_report.kind must be 'training_signal_report'")
    if source_training_signal_report.sha256 is None:
        errors.append("source_training_signal_report.sha256 must be set")
    if source_training_signal_report.provenance.get("run_id") != training_signal_report.run_id:
        errors.append("source_training_signal_report provenance run_id must match report.run_id")
    if "producer" not in source_training_signal_report.provenance:
        errors.append("source_training_signal_report must set provenance.producer")
    if "derivation" not in source_training_signal_report.provenance:
        errors.append("source_training_signal_report must set provenance.derivation")
    try:
        source_path = local_artifact_path(
            path=source_training_signal_report.path,
            uri=source_training_signal_report.uri,
            field_name="source_training_signal_report",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        source_path = None
    if source_path is None:
        errors.append("source_training_signal_report must be local byte-verifiable")
    elif not source_path.exists():
        errors.append("source_training_signal_report.path must exist")
    elif not source_path.is_file():
        errors.append("source_training_signal_report.path must be a file")
    elif source_training_signal_report.sha256 is not None:
        if sha256_file(source_path) != source_training_signal_report.sha256:
            errors.append("source_training_signal_report.sha256 must match path")
        else:
            try:
                source_report = TrainingSignalReport.from_dict(read_json(source_path))
            except HarnessIOError as exc:
                errors.append("source_training_signal_report must parse: " + str(exc))
            else:
                if source_report != training_signal_report:
                    errors.append(
                        "source_training_signal_report does not match supplied training_signal_report"
                    )
    if errors:
        raise HarnessIOError("; ".join(errors))
    return cast(str, source_training_signal_report.sha256)


def _counts(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))
