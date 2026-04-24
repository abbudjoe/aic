import hashlib
from pathlib import Path

import pytest

from aic_signal_harness import (
    ArtifactRef,
    HarnessIOError,
    Hdf5DatasetReduction,
    Hdf5DatasetReport,
    Hdf5DatasetStats,
    Hdf5DatasetThresholds,
    reduce_hdf5_dataset,
    sha256_file,
    validate_hdf5_dataset,
)


def _write_valid_dataset(path: Path, *, episode_lengths: tuple[int, ...] = (2,)) -> None:
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")

    path.parent.mkdir(parents=True, exist_ok=True)
    step_count = sum(episode_lengths)
    episode_offsets: list[int] = []
    running_offset = 0
    for length in episode_lengths:
        episode_offsets.append(running_offset)
        running_offset += length

    episode_idx: list[int] = []
    step_idx: list[int] = []
    for episode_number, length in enumerate(episode_lengths):
        for local_step in range(length):
            episode_idx.append(episode_number)
            step_idx.append(local_step)

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ep_len", data=np.asarray(episode_lengths, dtype=np.int32))
        handle.create_dataset("ep_offset", data=np.asarray(episode_offsets, dtype=np.int64))
        handle.create_dataset("ep_idx", data=np.asarray(episode_idx, dtype=np.int32))
        handle.create_dataset("episode_idx", data=np.asarray(episode_idx, dtype=np.int32))
        handle.create_dataset("step_idx", data=np.asarray(step_idx, dtype=np.int32))
        handle.create_dataset("pixels", data=np.zeros((step_count, 4, 5, 3), dtype=np.uint8))
        handle.create_dataset("left_pixels", data=np.zeros((step_count, 4, 5, 3), dtype=np.uint8))
        handle.create_dataset("right_pixels", data=np.zeros((step_count, 4, 5, 3), dtype=np.uint8))
        handle.create_dataset("proprio", data=np.zeros((step_count, 32), dtype=np.float32))
        handle.create_dataset("state", data=np.zeros((step_count, 32), dtype=np.float32))
        handle.create_dataset("action", data=np.zeros((step_count, 6), dtype=np.float32))
        for dataset_name in ("task_id", "plug_type", "port_type", "target_module_name"):
            handle.create_dataset(
                dataset_name,
                data=np.asarray(["a"] * step_count, dtype=object),
                dtype=string_dtype,
            )


def test_validate_hdf5_dataset_accepts_expected_aic_schema(tmp_path: Path) -> None:
    dataset = tmp_path / "rollout.hdf5"
    _write_valid_dataset(dataset, episode_lengths=(2, 3))

    report = validate_hdf5_dataset(
        dataset,
        min_episodes=2,
        min_steps=5,
        validated_at_utc="2026-04-23T12:00:00Z",
    )

    assert report == Hdf5DatasetReport(
        source=str(dataset),
        size_bytes=dataset.stat().st_size,
        sha256=sha256_file(dataset),
        validated_at_utc="2026-04-23T12:00:00Z",
        thresholds=Hdf5DatasetThresholds(min_episodes=2, min_steps=5),
        required_datasets=report.required_datasets,
        missing_datasets=(),
        datasets=report.datasets,
        episode_count=2,
        step_count=5,
        episode_lengths=(2, 3),
        episode_offsets=(0, 2),
        errors=(),
        ok=True,
    )


def test_validate_hdf5_dataset_reports_missing_required_datasets(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    dataset = tmp_path / "broken.hdf5"
    with h5py.File(dataset, "w") as handle:
        handle.create_dataset("ep_len", data=[1])

    report = validate_hdf5_dataset(dataset, validated_at_utc="2026-04-23T12:00:00Z")

    assert report.ok is False
    assert report.episode_count is None
    assert "ep_offset" in report.missing_datasets
    assert any("missing required datasets" in error for error in report.errors)


def test_validate_hdf5_dataset_reports_shape_and_dtype_mismatches(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    dataset = tmp_path / "broken.hdf5"
    _write_valid_dataset(dataset)

    with h5py.File(dataset, "a") as handle:
        del handle["pixels"]
        del handle["action"]
        handle.create_dataset("pixels", data=[[[0.0]]])
        handle.create_dataset("action", data=[[0.0, 1.0, 2.0]])

    report = validate_hdf5_dataset(dataset, validated_at_utc="2026-04-23T12:00:00Z")

    assert report.ok is False
    assert "pixels must be THWC images with 3 channels" in " ".join(report.errors)
    assert "pixels must be uint8" in " ".join(report.errors)
    assert "action must have shape (T, 6)" in " ".join(report.errors)


def test_validate_hdf5_dataset_reports_non_dataset_top_level_objects(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    dataset = tmp_path / "broken.hdf5"
    with h5py.File(dataset, "w") as handle:
        handle.create_group("ep_len")

    report = validate_hdf5_dataset(dataset, validated_at_utc="2026-04-23T12:00:00Z")

    assert report.ok is False
    assert "ep_len" in report.missing_datasets
    assert any("ep_len must be a dataset, got Group" == error for error in report.errors)


def test_validate_hdf5_dataset_reports_threshold_failures(tmp_path: Path) -> None:
    dataset = tmp_path / "rollout.hdf5"
    _write_valid_dataset(dataset, episode_lengths=(2,))

    report = validate_hdf5_dataset(
        dataset,
        min_episodes=2,
        min_steps=3,
        validated_at_utc="2026-04-23T12:00:00Z",
    )

    assert report.ok is False
    assert "episode_count 1 < required 2" in report.errors
    assert "step_count 2 < required 3" in report.errors
    assert report.thresholds == Hdf5DatasetThresholds(min_episodes=2, min_steps=3)


def test_validate_hdf5_dataset_reports_invalid_episode_offsets(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    dataset = tmp_path / "rollout.hdf5"
    _write_valid_dataset(dataset, episode_lengths=(2, 3))

    with h5py.File(dataset, "a") as handle:
        del handle["ep_offset"]
        handle.create_dataset("ep_offset", data=np.asarray([99, 99], dtype=np.int64))

    report = validate_hdf5_dataset(dataset, validated_at_utc="2026-04-23T12:00:00Z")

    assert report.ok is False
    assert any(
        "ep_offset must start at 0 and match cumulative episode starts implied by ep_len"
        in error
        for error in report.errors
    )


def test_reduce_hdf5_dataset_canonicalizes_local_path_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "datasets" / "rollout.hdf5"
    _write_valid_dataset(dataset)
    monkeypatch.chdir(tmp_path)

    relative_reduction = reduce_hdf5_dataset(
        Path("datasets/rollout.hdf5"),
        validated_at_utc="2026-04-23T12:00:00Z",
    )
    absolute_reduction = reduce_hdf5_dataset(
        dataset,
        validated_at_utc="2026-04-23T12:00:00Z",
    )

    assert relative_reduction == absolute_reduction
    assert relative_reduction.artifact.path == str(dataset)
    assert relative_reduction.report.source == str(dataset)


def test_reduce_hdf5_dataset_binds_artifact_to_report(tmp_path: Path) -> None:
    dataset = tmp_path / "rollout.hdf5"
    _write_valid_dataset(dataset)

    reduction = reduce_hdf5_dataset(
        dataset,
        uri="gs://bucket/datasets/rollout.hdf5",
        provenance={"producer": "pytest"},
        validated_at_utc="2026-04-23T12:00:00Z",
    )

    assert reduction.artifact.kind == "hdf5_dataset"
    assert reduction.artifact.path == str(dataset)
    assert reduction.artifact.uri is None
    assert reduction.artifact.sha256 == reduction.report.sha256 == sha256_file(dataset)
    assert reduction.artifact.provenance == {
        "producer": "pytest",
        "declared_uri": "gs://bucket/datasets/rollout.hdf5",
    }


def test_hdf5_dataset_reduction_rejects_malformed_evidence() -> None:
    with pytest.raises(
        HarnessIOError,
        match="artifact.path must match report.source when report.source is a local path",
    ):
        Hdf5DatasetReduction(
            artifact=ArtifactRef(
                kind="hdf5_dataset",
                path="/tmp/other.hdf5",
                sha256="a" * 64,
            ),
            report=Hdf5DatasetReport(
                source="/tmp/demo.hdf5",
                size_bytes=123,
                sha256="a" * 64,
                validated_at_utc="2026-04-23T12:00:00Z",
                thresholds=Hdf5DatasetThresholds(min_episodes=1, min_steps=1),
                required_datasets=("ep_len", "ep_offset"),
                missing_datasets=("ep_offset",),
                datasets={"ep_len": Hdf5DatasetStats(shape=(1,), dtype="int32")},
                errors=("missing required datasets: ep_offset",),
                ok=False,
            ),
        )


def test_hdf5_dataset_reduction_rejects_conflicting_second_identity_for_local_source() -> None:
    with pytest.raises(
        HarnessIOError,
        match="artifact.uri must be unset when report.source is a local path",
    ):
        Hdf5DatasetReduction(
            artifact=ArtifactRef(
                kind="hdf5_dataset",
                path="/tmp/demo.hdf5",
                uri="gs://bucket/demo.hdf5",
                sha256="a" * 64,
            ),
            report=Hdf5DatasetReport(
                source="/tmp/demo.hdf5",
                size_bytes=123,
                sha256="a" * 64,
                validated_at_utc="2026-04-23T12:00:00Z",
                thresholds=Hdf5DatasetThresholds(min_episodes=1, min_steps=1),
                required_datasets=("ep_len",),
                missing_datasets=("ep_len",),
                datasets={},
                errors=("missing required datasets: ep_len",),
                ok=False,
            ),
        )


def test_hdf5_dataset_reduction_rejects_path_when_report_source_is_uri() -> None:
    with pytest.raises(
        HarnessIOError,
        match="artifact.path must be unset when report.source is a uri",
    ):
        Hdf5DatasetReduction(
            artifact=ArtifactRef(
                kind="hdf5_dataset",
                path="/tmp/other.hdf5",
                uri="gs://bucket/demo.hdf5",
                sha256="a" * 64,
            ),
            report=Hdf5DatasetReport(
                source="gs://bucket/demo.hdf5",
                size_bytes=123,
                sha256="a" * 64,
                validated_at_utc="2026-04-23T12:00:00Z",
                thresholds=Hdf5DatasetThresholds(min_episodes=1, min_steps=1),
                required_datasets=("ep_len",),
                missing_datasets=("ep_len",),
                datasets={},
                errors=("missing required datasets: ep_len",),
                ok=False,
            ),
        )
