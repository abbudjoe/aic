from __future__ import annotations

from pathlib import Path

import pytest

from aic_signal_harness import (
    ArtifactRef,
    HarnessIOError,
    LeakageClass,
    TrainingDatasetExample,
    TrainingDatasetReport,
    TrainingSignal,
    TrainingSignalKind,
    TrainingSignalReport,
    derive_training_dataset_report,
    sha256_file,
    training_dataset_example_id,
    training_dataset_fingerprint_sha256,
    write_json,
)


def test_training_dataset_report_round_trip_and_deterministic_derivation(tmp_path: Path) -> None:
    signal_report = _training_signal_report()
    source_artifact = _write_training_signal_report(tmp_path, signal_report)

    report = derive_training_dataset_report(
        signal_report,
        source_training_signal_report=source_artifact,
    )
    repeated = derive_training_dataset_report(
        signal_report,
        source_training_signal_report=source_artifact,
    )

    assert repeated == report
    assert TrainingDatasetReport.from_dict(report.to_dict()) == report
    assert report.ok is True
    assert report.errors == ()
    assert report.generated_at_utc == signal_report.generated_at_utc
    assert report.example_count == 2
    assert report.selected_signal_ids == ("tsig_action_1", "tsig_safety_1")
    assert dict(report.signals_by_kind) == {
        "behavior_clone_action": 1,
        "safety_guard_avoidance": 1,
    }
    assert dict(report.signals_by_leakage_class) == {
        "legal_policy_action_output": 1,
        "privileged_training_signal": 1,
    }
    assert dict(report.signals_by_target) == {
        "policy.action": 1,
        "safety.guard": 1,
    }
    assert report.dataset_fingerprint_sha256 == training_dataset_fingerprint_sha256(
        report.examples
    )
    assert report.examples[0].example_id == training_dataset_example_id(
        source_training_signal_report_sha256=source_artifact.sha256 or "",
        signal_id="tsig_action_1",
    )
    assert all(example.offline_only is True for example in report.examples)
    assert all(example.runtime_allowed is False for example in report.examples)
    assert all(
        example.consumable_by_policy_runtime is False for example in report.examples
    )


def test_training_dataset_report_binds_source_training_signal_report(tmp_path: Path) -> None:
    signal_report = _training_signal_report()
    source_artifact = _write_training_signal_report(tmp_path, signal_report)

    forged_source = _training_signal_report(
        signals=(
            _training_signal(
                signal_id="tsig_action_1",
                target="forged.action",
            ),
            _training_signal(
                signal_id="tsig_safety_1",
                kind=TrainingSignalKind.safety_guard_avoidance,
                target="safety.guard",
                weight=0.5,
                leakage_class=LeakageClass.privileged_training_signal,
                source_event_indices=(2,),
                extraction_method="episode_trace.v1.safety_guard",
            ),
        )
    )
    forged_path = tmp_path / "forged_training_signal_report.json"
    write_json(forged_path, forged_source.to_dict())
    forged_artifact = ArtifactRef(
        kind="training_signal_report",
        path=str(forged_path),
        sha256=sha256_file(forged_path),
        provenance={
            "producer": "pytest",
            "run_id": signal_report.run_id,
            "derivation": "derive_training_signal_report",
        },
    )

    with pytest.raises(HarnessIOError, match="does not match supplied training_signal_report"):
        derive_training_dataset_report(
            signal_report,
            source_training_signal_report=forged_artifact,
        )

    uri_only_artifact = ArtifactRef(
        kind="training_signal_report",
        uri="memory://pytest/training_signal_report.json",
        sha256=source_artifact.sha256,
        provenance={
            "producer": "pytest",
            "run_id": signal_report.run_id,
            "derivation": "derive_training_signal_report",
        },
    )
    with pytest.raises(HarnessIOError, match="local byte-verifiable"):
        derive_training_dataset_report(
            signal_report,
            source_training_signal_report=uri_only_artifact,
        )

    wrong_kind_artifact = ArtifactRef(
        kind="episode_trace",
        path=source_artifact.path,
        sha256=source_artifact.sha256,
        provenance={
            "producer": "pytest",
            "run_id": signal_report.run_id,
            "derivation": "derive_training_signal_report",
        },
    )
    with pytest.raises(HarnessIOError, match="kind must be 'training_signal_report'"):
        derive_training_dataset_report(
            signal_report,
            source_training_signal_report=wrong_kind_artifact,
        )


def test_training_dataset_rejects_source_report_missing_required_signals(tmp_path: Path) -> None:
    signal_report = _training_signal_report()
    malformed_payload = signal_report.to_dict()
    del malformed_payload["signals"]
    malformed_path = tmp_path / "malformed_training_signal_report.json"
    write_json(malformed_path, malformed_payload)
    malformed_artifact = ArtifactRef(
        kind="training_signal_report",
        path=str(malformed_path),
        sha256=sha256_file(malformed_path),
        provenance={
            "producer": "pytest",
            "run_id": signal_report.run_id,
            "derivation": "derive_training_signal_report",
        },
    )

    with pytest.raises(HarnessIOError, match="missing required fields"):
        TrainingSignalReport.from_dict(malformed_payload)
    with pytest.raises(HarnessIOError, match="missing required fields"):
        derive_training_dataset_report(
            signal_report,
            source_training_signal_report=malformed_artifact,
        )


def test_training_dataset_report_rejects_forged_examples(tmp_path: Path) -> None:
    signal_report = _training_signal_report()
    source_artifact = _write_training_signal_report(tmp_path, signal_report)
    report = derive_training_dataset_report(
        signal_report,
        source_training_signal_report=source_artifact,
    )

    payload = report.to_dict()
    payload["examples"][0]["target"] = "forged.action"
    payload["signals_by_target"] = {
        "forged.action": 1,
        "safety.guard": 1,
    }
    examples = tuple(
        TrainingDatasetExample.from_dict(example)
        for example in payload["examples"]
    )
    payload["dataset_fingerprint_sha256"] = training_dataset_fingerprint_sha256(examples)

    with pytest.raises(HarnessIOError, match="must match source signal"):
        TrainingDatasetReport.from_dict(payload)


def test_training_dataset_report_rejects_omitted_source_signals(tmp_path: Path) -> None:
    signal_report = _training_signal_report()
    source_artifact = _write_training_signal_report(tmp_path, signal_report)
    report = derive_training_dataset_report(
        signal_report,
        source_training_signal_report=source_artifact,
    )

    payload = report.to_dict()
    payload["examples"] = payload["examples"][:1]
    payload["example_count"] = 1
    payload["selected_signal_ids"] = ["tsig_action_1"]
    payload["signals_by_kind"] = {"behavior_clone_action": 1}
    payload["signals_by_leakage_class"] = {"legal_policy_action_output": 1}
    payload["signals_by_target"] = {"policy.action": 1}
    payload["dataset_fingerprint_sha256"] = training_dataset_fingerprint_sha256(
        (TrainingDatasetExample.from_dict(payload["examples"][0]),)
    )

    with pytest.raises(HarnessIOError, match="include every source signal"):
        TrainingDatasetReport.from_dict(payload)


def test_training_dataset_report_binds_source_timestamp_split_and_empty_status(tmp_path: Path) -> None:
    signal_report = _training_signal_report()
    source_artifact = _write_training_signal_report(tmp_path, signal_report)
    report = derive_training_dataset_report(
        signal_report,
        source_training_signal_report=source_artifact,
    )

    forged_timestamp = report.to_dict()
    forged_timestamp["generated_at_utc"] = "2026-04-24T00:00:07Z"
    with pytest.raises(HarnessIOError, match="generated_at_utc must match source"):
        TrainingDatasetReport.from_dict(forged_timestamp)

    forged_split = report.to_dict()
    forged_split["examples"][0]["split"] = "validation"
    examples = tuple(
        TrainingDatasetExample.from_dict(example)
        for example in forged_split["examples"]
    )
    forged_split["dataset_fingerprint_sha256"] = training_dataset_fingerprint_sha256(examples)
    with pytest.raises(HarnessIOError, match="must match source signal"):
        TrainingDatasetReport.from_dict(forged_split)

    empty_signal_report = _training_signal_report(signals=())
    empty_source_artifact = _write_training_signal_report(
        tmp_path,
        empty_signal_report,
        path_name="empty_training_signal_report.json",
    )
    empty_report = derive_training_dataset_report(
        empty_signal_report,
        source_training_signal_report=empty_source_artifact,
    )
    forged_empty = empty_report.to_dict()
    forged_empty["ok"] = True
    forged_empty["errors"] = []
    with pytest.raises(HarnessIOError, match="ok must match source signal selection"):
        TrainingDatasetReport.from_dict(forged_empty)


def test_training_dataset_report_rejects_runtime_boundary_regressions(tmp_path: Path) -> None:
    signal_report = _training_signal_report()
    source_artifact = _write_training_signal_report(tmp_path, signal_report)
    report = derive_training_dataset_report(
        signal_report,
        source_training_signal_report=source_artifact,
    )

    missing_runtime = report.to_dict()
    del missing_runtime["examples"][0]["runtime_allowed"]
    with pytest.raises(HarnessIOError, match="missing required runtime boundary fields"):
        TrainingDatasetReport.from_dict(missing_runtime)

    runtime_allowed = report.to_dict()
    runtime_allowed["examples"][0]["runtime_allowed"] = True
    with pytest.raises(HarnessIOError, match="runtime_allowed must be false"):
        TrainingDatasetReport.from_dict(runtime_allowed)


def test_training_dataset_report_keeps_empty_selection_explicit(tmp_path: Path) -> None:
    signal_report = _training_signal_report(signals=())
    source_artifact = _write_training_signal_report(tmp_path, signal_report)

    report = derive_training_dataset_report(
        signal_report,
        source_training_signal_report=source_artifact,
    )

    assert TrainingDatasetReport.from_dict(report.to_dict()) == report
    assert report.ok is False
    assert report.errors == ("no selected training signals",)
    assert report.example_count == 0
    assert report.selected_signal_ids == ()


def test_training_dataset_report_rejects_unknown_fields(tmp_path: Path) -> None:
    signal_report = _training_signal_report()
    source_artifact = _write_training_signal_report(tmp_path, signal_report)
    report = derive_training_dataset_report(
        signal_report,
        source_training_signal_report=source_artifact,
    )
    payload = report.to_dict()
    payload["surprise"] = True

    with pytest.raises(HarnessIOError, match="unknown fields"):
        TrainingDatasetReport.from_dict(payload)


def test_training_dataset_report_rejects_missing_required_fields(tmp_path: Path) -> None:
    signal_report = _training_signal_report(signals=())
    source_artifact = _write_training_signal_report(tmp_path, signal_report)
    report = derive_training_dataset_report(
        signal_report,
        source_training_signal_report=source_artifact,
    )
    payload = report.to_dict()
    del payload["signals_by_kind"]

    with pytest.raises(HarnessIOError, match="missing required fields"):
        TrainingDatasetReport.from_dict(payload)


def _training_signal_report(
    *,
    signals: tuple[TrainingSignal, ...] | None = None,
) -> TrainingSignalReport:
    return TrainingSignalReport(
        run_id="run-a",
        generated_at_utc="2026-04-24T00:00:06Z",
        source_trace=ArtifactRef(
            kind="episode_trace",
            uri="memory://pytest/episode_trace.json",
            sha256="a" * 64,
            provenance={
                "producer": "pytest",
                "run_id": "run-a",
                "derivation": "derive_episode_trace",
            },
        ),
        signals=(
            _training_signal(
                signal_id="tsig_action_1",
                target="policy.action",
            ),
            _training_signal(
                signal_id="tsig_safety_1",
                kind=TrainingSignalKind.safety_guard_avoidance,
                target="safety.guard",
                weight=0.5,
                leakage_class=LeakageClass.privileged_training_signal,
                source_event_indices=(2,),
                extraction_method="episode_trace.v1.safety_guard",
            ),
        )
        if signals is None
        else signals,
        notes=("Signals are offline-only.",),
    )


def _training_signal(
    *,
    signal_id: str,
    target: str,
    kind: TrainingSignalKind = TrainingSignalKind.behavior_clone_action,
    weight: float = 1.0,
    leakage_class: LeakageClass = LeakageClass.legal_policy_action_output,
    source_event_indices: tuple[int, ...] = (1,),
    extraction_method: str = "episode_trace.v1.behavior_clone_action",
) -> TrainingSignal:
    return TrainingSignal(
        signal_id=signal_id,
        kind=kind,
        target=target,
        weight=weight,
        source="episode_trace",
        source_event_indices=source_event_indices,
        extraction_method=extraction_method,
        leakage_class=leakage_class,
        evidence={"payload": {"linear": [0.1, 0.0, 0.0]}},
        trial_id="trial-a",
    )


def _write_training_signal_report(
    tmp_path: Path,
    report: TrainingSignalReport,
    *,
    path_name: str = "training_signal_report.json",
) -> ArtifactRef:
    path = tmp_path / path_name
    write_json(path, report.to_dict())
    return ArtifactRef(
        kind="training_signal_report",
        path=str(path),
        sha256=sha256_file(path),
        provenance={
            "producer": "pytest",
            "run_id": report.run_id,
            "derivation": "derive_training_signal_report",
        },
    )
