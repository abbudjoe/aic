from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic_signal_harness import (
    ArtifactRef,
    HarnessIOError,
    LeakageClass,
    TRAINER_DATASET_JSONL_KIND,
    TRAINING_DATASET_MATERIALIZATION_KIND,
    TrainingDatasetMaterializationFormat,
    TrainingDatasetMaterializationReport,
    TrainingSignal,
    TrainingSignalKind,
    TrainingSignalReport,
    derive_training_dataset_report,
    materialize_training_dataset_jsonl,
    read_json,
    read_training_dataset_materialization_report_artifact,
    sha256_file,
    write_json,
    write_training_dataset_materialization_report,
)


def test_dataset_materialization_writes_canonical_jsonl_and_report(tmp_path: Path) -> None:
    source_artifact = _training_dataset_artifact(tmp_path)
    dataset_path = tmp_path / "trainer_dataset.jsonl"

    report = materialize_training_dataset_jsonl(
        source_training_dataset_report=source_artifact,
        output_path=dataset_path,
        generated_at_utc="2026-04-25T03:00:00Z",
        notes=("pytest materialization",),
    )
    report_artifact = write_training_dataset_materialization_report(
        tmp_path / "training_dataset_materialization_report.json",
        report,
    )
    reread = read_training_dataset_materialization_report_artifact(report_artifact)
    source_report = read_json(Path(source_artifact.path or ""))

    assert reread == report
    assert report_artifact.kind == TRAINING_DATASET_MATERIALIZATION_KIND
    assert report.format is TrainingDatasetMaterializationFormat.jsonl
    assert report.materialized_dataset.kind == TRAINER_DATASET_JSONL_KIND
    assert report.materialized_dataset.sha256 == sha256_file(dataset_path)
    assert report.source_training_dataset_report == source_artifact
    assert report.dataset_fingerprint_sha256 == source_report["dataset_fingerprint_sha256"]
    assert report.example_count == 2
    assert dict(report.split_counts) == {"train": 2}
    assert dict(report.leakage_summary) == {
        "legal_policy_action_output": 1,
        "legal_policy_input": 1,
    }
    assert report.offline_only is True
    assert report.runtime_allowed is False
    assert report.consumable_by_policy_runtime is False
    assert report.ok is True
    assert report.errors == ()
    assert [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()] == (
        source_report["examples"]
    )


def test_dataset_materialization_rejects_noncanonical_jsonl_bytes(tmp_path: Path) -> None:
    source_artifact = _training_dataset_artifact(tmp_path)
    dataset_path = tmp_path / "trainer_dataset.jsonl"
    report = materialize_training_dataset_jsonl(
        source_training_dataset_report=source_artifact,
        output_path=dataset_path,
        generated_at_utc="2026-04-25T03:00:00Z",
    )
    source_report = read_json(Path(source_artifact.path or ""))
    pretty_jsonl = "".join(
        json.dumps(example, indent=2, sort_keys=True) + "\n"
        for example in source_report["examples"]
    )
    dataset_path.write_text(pretty_jsonl, encoding="utf-8")
    payload = report.to_dict()
    payload["materialized_dataset"]["sha256"] = sha256_file(dataset_path)

    with pytest.raises(HarnessIOError, match="canonical JSONL"):
        TrainingDatasetMaterializationReport.from_dict(payload)


def test_dataset_materialization_rejects_artifact_provenance_regressions(
    tmp_path: Path,
) -> None:
    source_artifact = _training_dataset_artifact(tmp_path)
    report = materialize_training_dataset_jsonl(
        source_training_dataset_report=source_artifact,
        output_path=tmp_path / "trainer_dataset.jsonl",
        generated_at_utc="2026-04-25T03:00:00Z",
    )

    missing_dataset_provenance = report.to_dict()
    del missing_dataset_provenance["materialized_dataset"]["provenance"]["derivation"]
    with pytest.raises(HarnessIOError, match="provenance must set derivation"):
        TrainingDatasetMaterializationReport.from_dict(missing_dataset_provenance)

    report_artifact = write_training_dataset_materialization_report(
        tmp_path / "training_dataset_materialization_report.json",
        report,
    )
    with pytest.raises(HarnessIOError, match="provenance source_training_dataset_report_sha256"):
        read_training_dataset_materialization_report_artifact(
            ArtifactRef(
                kind=report_artifact.kind,
                path=report_artifact.path,
                sha256=report_artifact.sha256,
                provenance={
                    **dict(report_artifact.provenance),
                    "source_training_dataset_report_sha256": "0" * 64,
                },
            )
        )


def test_dataset_materialization_rejects_contract_contradictions(tmp_path: Path) -> None:
    source_artifact = _training_dataset_artifact(tmp_path)
    report = materialize_training_dataset_jsonl(
        source_training_dataset_report=source_artifact,
        output_path=tmp_path / "trainer_dataset.jsonl",
        generated_at_utc="2026-04-25T03:00:00Z",
    )

    runtime_allowed = report.to_dict()
    runtime_allowed["runtime_allowed"] = True
    with pytest.raises(HarnessIOError, match="runtime_allowed must be false"):
        TrainingDatasetMaterializationReport.from_dict(runtime_allowed)

    wrong_count = report.to_dict()
    wrong_count["example_count"] = 1
    with pytest.raises(HarnessIOError, match="example_count must match source report"):
        TrainingDatasetMaterializationReport.from_dict(wrong_count)

    with pytest.raises(HarnessIOError, match="local byte-verifiable"):
        materialize_training_dataset_jsonl(
            source_training_dataset_report=ArtifactRef(
                kind="training_dataset_report",
                uri="memory://pytest/training_dataset_report.json",
                sha256=source_artifact.sha256,
                provenance=source_artifact.provenance,
            ),
            output_path=tmp_path / "unwritten.jsonl",
            generated_at_utc="2026-04-25T03:00:00Z",
        )


def test_dataset_materialization_rejects_output_aliases_before_corruption(
    tmp_path: Path,
) -> None:
    source_artifact = _training_dataset_artifact(tmp_path)
    source_path = Path(source_artifact.path or "")
    source_before = source_path.read_bytes()
    source_payload = read_json(source_path)
    signal_path = Path(source_payload["source_training_signal_report"]["path"])
    signal_before = signal_path.read_bytes()

    with pytest.raises(HarnessIOError, match="must not alias protected artifact"):
        materialize_training_dataset_jsonl(
            source_training_dataset_report=source_artifact,
            output_path=source_path,
            generated_at_utc="2026-04-25T03:00:00Z",
            overwrite=True,
        )
    assert source_path.read_bytes() == source_before

    with pytest.raises(HarnessIOError, match="must not alias protected artifact"):
        materialize_training_dataset_jsonl(
            source_training_dataset_report=source_artifact,
            output_path=signal_path,
            generated_at_utc="2026-04-25T03:00:00Z",
            overwrite=True,
        )
    assert signal_path.read_bytes() == signal_before


def test_dataset_materialization_report_rejects_output_aliases_before_corruption(
    tmp_path: Path,
) -> None:
    source_artifact = _training_dataset_artifact(tmp_path)
    source_path = Path(source_artifact.path or "")
    source_before = source_path.read_bytes()
    report = materialize_training_dataset_jsonl(
        source_training_dataset_report=source_artifact,
        output_path=tmp_path / "trainer_dataset.jsonl",
        generated_at_utc="2026-04-25T03:00:00Z",
    )
    dataset_path = Path(report.materialized_dataset.path or "")
    dataset_before = dataset_path.read_bytes()

    with pytest.raises(HarnessIOError, match="must not alias protected artifact"):
        write_training_dataset_materialization_report(
            dataset_path,
            report,
            overwrite=True,
        )
    assert dataset_path.read_bytes() == dataset_before

    with pytest.raises(HarnessIOError, match="must not alias protected artifact"):
        write_training_dataset_materialization_report(
            source_path,
            report,
            overwrite=True,
        )
    assert source_path.read_bytes() == source_before


def test_dataset_materialization_report_writer_revalidates_current_bytes(
    tmp_path: Path,
) -> None:
    source_artifact = _training_dataset_artifact(tmp_path)
    report = materialize_training_dataset_jsonl(
        source_training_dataset_report=source_artifact,
        output_path=tmp_path / "trainer_dataset.jsonl",
        generated_at_utc="2026-04-25T03:00:00Z",
    )
    dataset_path = Path(report.materialized_dataset.path or "")
    dataset_path.write_text('{"example_id":"tampered"}\n', encoding="utf-8")

    with pytest.raises(HarnessIOError, match="sha256 must match path"):
        write_training_dataset_materialization_report(
            tmp_path / "training_dataset_materialization_report.json",
            report,
        )
    assert not (tmp_path / "training_dataset_materialization_report.json").exists()


def test_dataset_materialization_refuses_overwrite_and_empty_source(tmp_path: Path) -> None:
    source_artifact = _training_dataset_artifact(tmp_path)
    dataset_path = tmp_path / "trainer_dataset.jsonl"
    materialize_training_dataset_jsonl(
        source_training_dataset_report=source_artifact,
        output_path=dataset_path,
        generated_at_utc="2026-04-25T03:00:00Z",
    )

    with pytest.raises(HarnessIOError, match="already exists"):
        materialize_training_dataset_jsonl(
            source_training_dataset_report=source_artifact,
            output_path=dataset_path,
            generated_at_utc="2026-04-25T03:00:01Z",
        )

    empty_source_artifact = _training_dataset_artifact(tmp_path, signals=(), name="empty")
    with pytest.raises(HarnessIOError, match="ok must be true|example_count must be >= 1"):
        materialize_training_dataset_jsonl(
            source_training_dataset_report=empty_source_artifact,
            output_path=tmp_path / "empty.jsonl",
            generated_at_utc="2026-04-25T03:00:00Z",
        )


def _training_dataset_artifact(
    tmp_path: Path,
    *,
    signals: tuple[TrainingSignal, ...] | None = None,
    name: str = "source",
) -> ArtifactRef:
    signal_report = TrainingSignalReport(
        run_id=f"{name}-run-a",
        generated_at_utc="2026-04-25T02:00:00Z",
        source_trace=ArtifactRef(
            kind="episode_trace",
            uri=f"memory://pytest/{name}/episode_trace.json",
            sha256="a" * 64,
            provenance={
                "producer": "pytest",
                "derivation": "fixture",
                "run_id": f"{name}-run-a",
            },
        ),
        signals=(
            TrainingSignal(
                signal_id="signal-action-1",
                kind=TrainingSignalKind.behavior_clone_action,
                target="policy.action",
                weight=1.0,
                source="episode_trace",
                source_event_indices=(1,),
                extraction_method="episode_trace.v1.behavior_clone_action",
                leakage_class=LeakageClass.legal_policy_action_output,
                evidence={"payload": {"linear": [0.1, 0.0, 0.0]}},
                trial_id="trial-a",
            ),
            TrainingSignal(
                signal_id="signal-observation-1",
                kind=TrainingSignalKind.dataset_episode_summary,
                target="policy.observation",
                weight=0.5,
                source="episode_trace",
                source_event_indices=(2,),
                extraction_method="episode_trace.v1.dataset_episode_summary",
                leakage_class=LeakageClass.legal_policy_input,
                evidence={"payload": {"image": "frame-2"}},
                trial_id="trial-a",
            ),
        )
        if signals is None
        else signals,
    )
    signal_path = tmp_path / f"{name}_training_signal_report.json"
    write_json(signal_path, signal_report.to_dict(), overwrite=True)
    signal_artifact = ArtifactRef(
        kind="training_signal_report",
        path=str(signal_path),
        sha256=sha256_file(signal_path),
        provenance={
            "producer": "pytest",
            "derivation": "derive_training_signal_report",
            "run_id": signal_report.run_id,
        },
    )
    dataset_report = derive_training_dataset_report(
        signal_report,
        source_training_signal_report=signal_artifact,
    )
    dataset_path = tmp_path / f"{name}_training_dataset_report.json"
    write_json(dataset_path, dataset_report.to_dict(), overwrite=True)
    return ArtifactRef(
        kind="training_dataset_report",
        path=str(dataset_path),
        sha256=sha256_file(dataset_path),
        provenance={
            "producer": "pytest",
            "derivation": "derive_training_dataset_report",
            "run_id": dataset_report.run_id,
        },
    )
