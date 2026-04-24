from pathlib import Path

import pytest

from aic_signal_harness import (
    HarnessIOError,
    Hdf5DatasetReport,
    Hdf5DatasetStats,
    Hdf5DatasetThresholds,
)


def test_hdf5_dataset_report_round_trips_strictly() -> None:
    report = Hdf5DatasetReport(
        source="/tmp/demo.hdf5",
        size_bytes=123,
        sha256="a" * 64,
        validated_at_utc="2026-04-23T12:00:00Z",
        thresholds=Hdf5DatasetThresholds(min_episodes=3, min_steps=50),
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

    encoded = report.to_dict()

    assert Hdf5DatasetReport.from_dict(encoded) == report


def test_hdf5_dataset_report_rejects_unknown_fields() -> None:
    report_dict = {
        "schema_version": 1,
        "source": "/tmp/demo.hdf5",
        "size_bytes": 123,
        "sha256": "a" * 64,
        "validated_at_utc": "2026-04-23T12:00:00Z",
        "thresholds": {"min_episodes": 1, "min_steps": 1},
        "required_datasets": ["ep_len"],
        "missing_datasets": [],
        "datasets": {"ep_len": {"shape": [1], "dtype": "int32"}},
        "episode_count": 1,
        "step_count": 1,
        "episode_lengths": [1],
        "episode_offsets": [0],
        "errors": [],
        "ok": True,
        "unexpected": "boom",
    }

    with pytest.raises(HarnessIOError, match="unknown fields"):
        Hdf5DatasetReport.from_dict(report_dict)


def test_hdf5_dataset_report_rejects_relative_local_source_on_reload() -> None:
    report_dict = {
        "schema_version": 1,
        "source": "relative/demo.hdf5",
        "size_bytes": 123,
        "sha256": "a" * 64,
        "validated_at_utc": "2026-04-23T12:00:00Z",
        "thresholds": {"min_episodes": 1, "min_steps": 1},
        "required_datasets": ["ep_len"],
        "missing_datasets": [],
        "datasets": {"ep_len": {"shape": [1], "dtype": "int32"}},
        "episode_count": 1,
        "step_count": 1,
        "episode_lengths": [1],
        "episode_offsets": [0],
        "errors": [],
        "ok": True,
    }

    with pytest.raises(HarnessIOError, match="absolute local path or uri"):
        Hdf5DatasetReport.from_dict(report_dict)


def test_hdf5_dataset_report_rejects_incoherent_episode_offsets() -> None:
    with pytest.raises(HarnessIOError, match="must start at 0 and match cumulative episode_lengths"):
        Hdf5DatasetReport(
            source="/tmp/demo.hdf5",
            size_bytes=123,
            sha256="a" * 64,
            validated_at_utc="2026-04-23T12:00:00Z",
            thresholds=Hdf5DatasetThresholds(min_episodes=1, min_steps=1),
            required_datasets=("ep_len",),
            missing_datasets=(),
            datasets={"ep_len": Hdf5DatasetStats(shape=(2,), dtype="int32")},
            episode_count=2,
            step_count=3,
            episode_lengths=(1, 2),
            episode_offsets=(0, 99),
            errors=(),
            ok=True,
        )


def test_hdf5_dataset_report_rejects_missing_dataset_claims_that_do_not_match_dataset_map() -> None:
    with pytest.raises(
        HarnessIOError,
        match="missing_datasets must match required_datasets absent from datasets",
    ):
        Hdf5DatasetReport(
            source="/tmp/demo.hdf5",
            size_bytes=123,
            sha256="a" * 64,
            validated_at_utc="2026-04-23T12:00:00Z",
            thresholds=Hdf5DatasetThresholds(min_episodes=1, min_steps=1),
            required_datasets=("ep_len", "ep_offset"),
            missing_datasets=(),
            datasets={"ep_len": Hdf5DatasetStats(shape=(1,), dtype="int32")},
            episode_count=None,
            step_count=None,
            episode_lengths=(),
            episode_offsets=(),
            errors=("missing required datasets: ep_offset",),
            ok=False,
        )


def test_hdf5_dataset_report_from_dict_rejects_missing_dataset_claim_mismatch() -> None:
    with pytest.raises(
        HarnessIOError,
        match="missing_datasets must match required_datasets absent from datasets",
    ):
        Hdf5DatasetReport.from_dict(
            {
                "schema_version": 1,
                "source": "/tmp/demo.hdf5",
                "size_bytes": 123,
                "sha256": "a" * 64,
                "validated_at_utc": "2026-04-23T12:00:00Z",
                "thresholds": {"min_episodes": 1, "min_steps": 1},
                "required_datasets": ["ep_len", "ep_offset"],
                "missing_datasets": [],
                "datasets": {"ep_len": {"shape": [1], "dtype": "int32"}},
                "episode_count": None,
                "step_count": None,
                "episode_lengths": [],
                "episode_offsets": [],
                "errors": ["missing required datasets: ep_offset"],
                "ok": False,
            }
        )


def test_hdf5_dataset_report_from_dict_rejects_incoherent_episode_offsets() -> None:
    with pytest.raises(HarnessIOError, match="must start at 0 and match cumulative episode_lengths"):
        Hdf5DatasetReport.from_dict(
            {
                "schema_version": 1,
                "source": "/tmp/demo.hdf5",
                "size_bytes": 123,
                "sha256": "a" * 64,
                "validated_at_utc": "2026-04-23T12:00:00Z",
                "thresholds": {"min_episodes": 1, "min_steps": 1},
                "required_datasets": ["ep_len"],
                "missing_datasets": [],
                "datasets": {"ep_len": {"shape": [2], "dtype": "int32"}},
                "episode_count": 2,
                "step_count": 3,
                "episode_lengths": [1, 2],
                "episode_offsets": [0, 99],
                "errors": ["some other validation error"],
                "ok": False,
            }
        )


def test_hdf5_dataset_report_rejects_inconsistent_ok_state() -> None:
    with pytest.raises(HarnessIOError, match="errors must be nonempty"):
        Hdf5DatasetReport(
            source="/tmp/demo.hdf5",
            size_bytes=123,
            sha256="a" * 64,
            validated_at_utc="2026-04-23T12:00:00Z",
            thresholds=Hdf5DatasetThresholds(min_episodes=1, min_steps=1),
            required_datasets=("ep_len",),
            missing_datasets=(),
            datasets={"ep_len": Hdf5DatasetStats(shape=(1,), dtype="int32")},
            episode_count=1,
            step_count=1,
            episode_lengths=(1,),
            episode_offsets=(0,),
            errors=(),
            ok=False,
        )
