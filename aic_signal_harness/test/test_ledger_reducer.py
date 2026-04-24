import pytest

from aic_signal_harness import (
    ArtifactRef,
    BackendKind,
    HarnessIOError,
    Hdf5DatasetReduction,
    Hdf5DatasetReport,
    Hdf5DatasetStats,
    Hdf5DatasetThresholds,
    LeakageClass,
    PolicyBackendSpec,
    RunManifest,
    RunStatus,
    RuntimeBoundaryProof,
    RuntimeRole,
    ScoreReport,
    SimulatorKind,
    TrainingSourceKind,
    TrialScore,
    build_ledger_entry,
)


def _live_backend(**overrides):
    values = {
        "backend_kind": BackendKind.replay_servo,
        "name": "replay-servo-baseline",
        "runtime_role": RuntimeRole.live_policy,
        "training_sources": (TrainingSourceKind.official_demo,),
        "simulator_sources": (SimulatorKind.offline_replay,),
        "runtime_allowed": True,
        "leakage_class": LeakageClass.legal_policy_input,
        "runtime_boundary": RuntimeBoundaryProof(
            deterministic=True,
            uses_online_language_model_control=False,
            legal_observation_contract="official aic_model observations only",
        ),
        "description": "Replay a screened demonstration through the official runtime path.",
        "config": {"control_hz": 10},
        "provenance": {"producer": "pytest"},
    }
    values.update(overrides)
    return PolicyBackendSpec(**values)


def _score_report() -> ScoreReport:
    return ScoreReport(
        source="/tmp/eval/scoring.yaml",
        parsed_at_utc="2026-04-23T00:00:00Z",
        total=7.5,
        trials={"trial_1": TrialScore(total=7.5, tier_1=1.0, tier_2=2.5, tier_3=4.0)},
    )


def _dataset_reduction() -> Hdf5DatasetReduction:
    report = Hdf5DatasetReport(
        source="/tmp/data/demo.h5",
        size_bytes=123,
        sha256="c" * 64,
        validated_at_utc="2026-04-23T12:00:00Z",
        thresholds=Hdf5DatasetThresholds(min_episodes=1, min_steps=1),
        required_datasets=("ep_len", "ep_offset"),
        missing_datasets=(),
        datasets={
            "ep_len": Hdf5DatasetStats(shape=(3,), dtype="int32"),
            "ep_offset": Hdf5DatasetStats(shape=(3,), dtype="int64"),
        },
        episode_count=3,
        step_count=50,
        episode_lengths=(10, 20, 20),
        episode_offsets=(0, 10, 30),
        errors=(),
        ok=True,
    )
    return Hdf5DatasetReduction(
        artifact=ArtifactRef(
            kind="hdf5_dataset",
            path=report.source,
            sha256=report.sha256,
            provenance={"declared_uri": "gs://bucket/data/demo.h5"},
        ),
        report=report,
    )


def _manifest(**overrides) -> RunManifest:
    values = {
        "run_id": "gate2-run-a",
        "status": RunStatus.completed,
        "backend": _live_backend(),
        "created_at_utc": "2026-04-23T00:00:00Z",
        "updated_at_utc": "2026-04-23T00:10:00Z",
        "artifacts": (
            ArtifactRef(kind="scoring_yaml", path="/tmp/eval/scoring.yaml", sha256="b" * 64),
        ),
        "score": _score_report(),
        "experiment_id": "gate2",
        "notes": ("kept as current replay baseline",),
    }
    values.update(overrides)
    return RunManifest(**values)


def test_build_ledger_entry_from_completed_manifest_uses_score_and_dataset_reduction() -> None:
    manifest = _manifest(
        artifacts=(
            ArtifactRef(kind="scoring_yaml", path="/tmp/eval/scoring.yaml", sha256="b" * 64),
            ArtifactRef(kind="hdf5_dataset", uri="gs://bucket/data/demo.h5", sha256="c" * 64),
        ),
    )

    entry = build_ledger_entry(
        manifest,
        manifest_artifact=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-run-a/manifest.json",
            sha256="a" * 64,
        ),
        dataset_reduction=_dataset_reduction(),
        promotion={"decision": "accepted", "eligible_for_submission": True},
        recorded_at_utc="2026-04-23T12:00:00Z",
    )

    assert entry.metric.value == 7.5
    assert entry.dataset is not None
    assert entry.dataset.episode_count == 3
    assert entry.dataset.step_count == 50
    assert entry.dataset.artifact.path == "/tmp/data/demo.h5"
    assert entry.dataset.artifact.uri == "gs://bucket/data/demo.h5"
    assert entry.backend_kind is BackendKind.replay_servo
    assert entry.manifest.sha256 == "a" * 64


def test_build_ledger_entry_from_running_manifest_without_score_leaves_metric_empty() -> None:
    manifest = _manifest(
        status=RunStatus.running,
        artifacts=(),
        score=None,
    )

    entry = build_ledger_entry(
        manifest,
        manifest_artifact=ArtifactRef(
            kind="run_manifest",
            path="/tmp/runs/gate2-run-a/manifest.json",
            sha256="a" * 64,
        ),
        recorded_at_utc="2026-04-23T12:00:00Z",
    )

    assert entry.metric.value is None
    assert entry.dataset is None


def test_build_ledger_entry_rejects_manifest_artifact_without_sha256() -> None:
    with pytest.raises(HarnessIOError, match="ledger manifest artifact.sha256 must be set"):
        build_ledger_entry(
            _manifest(),
            manifest_artifact=ArtifactRef(
                kind="run_manifest",
                path="/tmp/runs/gate2-run-a/manifest.json",
            ),
        )


def test_build_ledger_entry_rejects_manifest_artifact_with_wrong_kind() -> None:
    with pytest.raises(HarnessIOError, match="ledger manifest artifact.kind must be 'run_manifest'"):
        build_ledger_entry(
            _manifest(),
            manifest_artifact=ArtifactRef(
                kind="log",
                path="/tmp/runs/gate2-run-a/manifest.json",
                sha256="a" * 64,
            ),
        )


def test_build_ledger_entry_rejects_duplicate_manifest_dataset_artifact_state() -> None:
    manifest = _manifest(
        artifacts=(
            ArtifactRef(kind="scoring_yaml", path="/tmp/eval/scoring.yaml", sha256="b" * 64),
            ArtifactRef(kind="hdf5_dataset", path="/tmp/data/demo.h5", sha256="c" * 64),
            ArtifactRef(kind="hdf5_dataset", uri="gs://bucket/data/demo.h5", sha256="c" * 64),
        ),
    )

    with pytest.raises(HarnessIOError, match="duplicate hdf5_dataset artifact state"):
        build_ledger_entry(
            manifest,
            manifest_artifact=ArtifactRef(
                kind="run_manifest",
                path="/tmp/runs/gate2-run-a/manifest.json",
                sha256="a" * 64,
            ),
            dataset_reduction=_dataset_reduction(),
        )


def test_build_ledger_entry_rejects_conflicting_dataset_artifact_state() -> None:
    manifest = _manifest(
        artifacts=(
            ArtifactRef(kind="scoring_yaml", path="/tmp/eval/scoring.yaml", sha256="b" * 64),
            ArtifactRef(kind="hdf5_dataset", path="/tmp/data/other.h5", sha256="d" * 64),
        ),
    )

    with pytest.raises(HarnessIOError, match="conflicting hdf5_dataset artifact state: path"):
        build_ledger_entry(
            manifest,
            manifest_artifact=ArtifactRef(
                kind="run_manifest",
                path="/tmp/runs/gate2-run-a/manifest.json",
                sha256="a" * 64,
            ),
            dataset_reduction=_dataset_reduction(),
        )
