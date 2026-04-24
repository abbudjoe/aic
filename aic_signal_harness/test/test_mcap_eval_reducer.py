import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aic_signal_harness import (
    ArtifactRef,
    HarnessIOError,
    McapEvalBundleReport,
    McapEvalTrialReport,
    McapEvalReduction,
    reduce_mcap_eval_bundle,
    analyze_mcap_eval_bundle,
)
from aic_signal_harness.reducers.mcap_eval import _bundle_sha256_from_report
import aic_signal_harness.reducers.mcap_eval as mcap_eval_reducer


def _report_sha256(mapping: dict[str, Any]) -> str:
    payload = (
        json.dumps(mapping, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stamp(sec: int, nanosec: int = 0) -> SimpleNamespace:
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def _point(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def _controller_state_message(
    *,
    sec: int,
    nanosec: int,
    position: tuple[float, float, float],
    tcp_error: tuple[float, float, float],
) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(sec, nanosec)),
        tcp_pose=SimpleNamespace(position=_point(*position)),
        tcp_error=tcp_error,
    )


def _pose_command_message(
    *,
    sec: int,
    nanosec: int,
    linear: tuple[float, float, float],
    angular: tuple[float, float, float],
) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(sec, nanosec)),
        velocity=SimpleNamespace(
            linear=_point(*linear),
            angular=_point(*angular),
        ),
    )


def _off_limit_contact_message(collision1: str, collision2: str) -> SimpleNamespace:
    return SimpleNamespace(
        contacts=[
            SimpleNamespace(
                collision1=SimpleNamespace(name=collision1),
                collision2=SimpleNamespace(name=collision2),
            )
        ]
    )


def _off_limit_contact_message_many(
    *pairs: tuple[str, str],
) -> SimpleNamespace:
    return SimpleNamespace(
        contacts=[
            SimpleNamespace(
                collision1=SimpleNamespace(name=collision1),
                collision2=SimpleNamespace(name=collision2),
            )
            for collision1, collision2 in pairs
        ]
    )


def _bag_paths(eval_dir: Path, trial_indexes: tuple[int, ...] = (1, 2, 3)) -> tuple[Path, ...]:
    paths = []
    for trial_index in trial_indexes:
        bag_dir = eval_dir / f"bag_trial_{trial_index}_20260423_120000_000"
        bag_dir.mkdir(parents=True, exist_ok=True)
        mcap_path = bag_dir / f"bag_trial_{trial_index}_20260423_120000_000_0.mcap"
        mcap_path.write_bytes(f"trial-{trial_index}".encode("utf-8"))
        paths.append(mcap_path)
    return tuple(paths)


def _fake_reader(entries: dict[tuple[str, str], tuple[SimpleNamespace, ...]]):
    def read_ros2_messages(mcap_path: str | Path, *, topics: list[str]):
        assert len(topics) == 1
        key = (str(Path(mcap_path).resolve()), topics[0])
        return entries.get(key, ())

    return read_ros2_messages


def test_analyze_mcap_eval_bundle_extracts_official_trial_evidence(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    trial_1, trial_2, trial_3 = _bag_paths(eval_dir)
    reader = _fake_reader(
        {
            (str(trial_1.resolve()), "/aic_controller/controller_state"): (
                SimpleNamespace(
                    log_time_ns=1_000_000_000,
                    ros_msg=_controller_state_message(
                        sec=10,
                        nanosec=0,
                        position=(1.0, 1.1, 1.2),
                        tcp_error=(0.1, 0.2, 0.3),
                    ),
                ),
                SimpleNamespace(
                    log_time_ns=1_500_000_000,
                    ros_msg=_controller_state_message(
                        sec=10,
                        nanosec=500_000_000,
                        position=(1.3, 1.4, 1.5),
                        tcp_error=(0.4, 0.5, 0.6),
                    ),
                ),
            ),
            (str(trial_1.resolve()), "/aic_controller/pose_commands"): (
                SimpleNamespace(
                    log_time_ns=1_250_000_000,
                    ros_msg=_pose_command_message(
                        sec=10,
                        nanosec=250_000_000,
                        linear=(0.01, 0.02, 0.03),
                        angular=(0.04, 0.05, 0.06),
                    ),
                ),
            ),
            (str(trial_3.resolve()), "/aic_controller/controller_state"): (
                SimpleNamespace(
                    log_time_ns=2_000_000_000,
                    ros_msg=_controller_state_message(
                        sec=20,
                        nanosec=0,
                        position=(2.0, 2.1, 2.2),
                        tcp_error=(0.7, 0.8, 0.9),
                    ),
                ),
                SimpleNamespace(
                    log_time_ns=2_400_000_000,
                    ros_msg=_controller_state_message(
                        sec=20,
                        nanosec=400_000_000,
                        position=(2.3, 2.4, 2.5),
                        tcp_error=(1.0, 1.1, 1.2),
                    ),
                ),
            ),
            (str(trial_3.resolve()), "/aic_controller/pose_commands"): (
                SimpleNamespace(
                    log_time_ns=2_150_000_000,
                    ros_msg=_pose_command_message(
                        sec=20,
                        nanosec=150_000_000,
                        linear=(0.11, 0.12, 0.13),
                        angular=(0.14, 0.15, 0.16),
                    ),
                ),
            ),
            (str(trial_3.resolve()), "/aic/gazebo/contacts/off_limit"): (
                SimpleNamespace(
                    log_time_ns=2_300_000_000,
                    ros_msg=_off_limit_contact_message("plug", "enclosure"),
                ),
                SimpleNamespace(
                    log_time_ns=2_350_000_000,
                    ros_msg=_off_limit_contact_message("plug", "enclosure"),
                ),
            ),
        }
    )

    report = analyze_mcap_eval_bundle(
        eval_dir,
        task_hints_by_trial={3: {"port_type": "sc", "task_id": "task_3"}},
        analyzed_at_utc="2026-04-23T12:00:00Z",
        read_ros2_messages=reader,
    )

    assert report.source == str(eval_dir.resolve())
    assert tuple(trial.trial_id for trial in report.trials) == ("trial_1", "trial_2", "trial_3")
    assert report.trials[0].controller_state_count == 2
    assert report.trials[0].controller_duration_sec == pytest.approx(0.5)
    assert report.trials[1].controller_state_count == 0
    assert report.trials[2].off_limit_contact_count == 2
    assert report.trials[2].first_off_limit_contact is not None
    assert report.trials[2].first_off_limit_contact.log_elapsed_sec == pytest.approx(0.3)
    assert report.trials[2].first_off_limit_contact.nearest_controller_elapsed_sec == pytest.approx(0.4)
    assert report.trials[2].first_off_limit_contact.nearest_command_elapsed_sec == pytest.approx(0.0)
    assert report.trials[2].first_off_limit_contact.recommended_stop_sec == pytest.approx(0.15)
    assert report.recommended_env == {"AIC_LEWM_SC_REPLAY_STOP_SEC": "0.15"}
    assert report.trials[2].task_hints is not None
    assert report.trials[2].task_hints.port_type == "sc"


def test_analyze_mcap_eval_bundle_counts_multiple_contacts_from_one_message(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    trial_1, trial_2, trial_3 = _bag_paths(eval_dir)
    reader = _fake_reader(
        {
            (str(trial_1.resolve()), "/aic_controller/controller_state"): (
                SimpleNamespace(
                    log_time_ns=1_000_000_000,
                    ros_msg=_controller_state_message(
                        sec=10,
                        nanosec=0,
                        position=(1.0, 1.1, 1.2),
                        tcp_error=(0.1, 0.2, 0.3),
                    ),
                ),
            ),
            (str(trial_1.resolve()), "/aic/gazebo/contacts/off_limit"): (
                SimpleNamespace(
                    log_time_ns=1_100_000_000,
                    ros_msg=_off_limit_contact_message_many(
                        ("plug", "enclosure"),
                        ("plug", "fixture"),
                    ),
                ),
            ),
        }
    )

    report = analyze_mcap_eval_bundle(
        eval_dir,
        analyzed_at_utc="2026-04-23T12:00:00Z",
        read_ros2_messages=reader,
    )

    assert report.trials[0].off_limit_contact_count == 2
    assert report.trials[0].first_off_limit_contact is not None
    assert report.trials[0].first_off_limit_contact.collision2 == "enclosure"


def test_reduce_mcap_eval_bundle_canonicalizes_local_path_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_dir = tmp_path / "eval"
    _bag_paths(eval_dir)
    reader = _fake_reader({})
    monkeypatch.chdir(tmp_path)

    relative_reduction = reduce_mcap_eval_bundle(
        Path("eval"),
        analyzed_at_utc="2026-04-23T12:00:00Z",
        read_ros2_messages=reader,
    )
    absolute_reduction = reduce_mcap_eval_bundle(
        eval_dir,
        analyzed_at_utc="2026-04-23T12:00:00Z",
        read_ros2_messages=reader,
    )

    assert relative_reduction == absolute_reduction
    assert relative_reduction.artifact.path == str(eval_dir.resolve())
    assert all(trial.source.startswith(str(eval_dir.resolve())) for trial in relative_reduction.report.trials)


def test_analyze_mcap_eval_bundle_rejects_file_changes_during_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_dir = tmp_path / "eval"
    trial_1, trial_2, trial_3 = _bag_paths(eval_dir)
    original_sha256_file = mcap_eval_reducer.sha256_file

    def mutating_sha256(path: str | Path) -> str:
        candidate = Path(path)
        if candidate.resolve() == trial_1.resolve():
            candidate.write_bytes(b"mutated-after-read")
        return original_sha256_file(candidate)

    monkeypatch.setattr(mcap_eval_reducer, "sha256_file", mutating_sha256)

    with pytest.raises(HarnessIOError, match="mcap file changed during analysis"):
        analyze_mcap_eval_bundle(
            eval_dir,
            analyzed_at_utc="2026-04-23T12:00:00Z",
            read_ros2_messages=_fake_reader({}),
        )


def test_analyze_mcap_eval_bundle_rejects_ambiguous_bag_directory_layout(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    bag_dir = eval_dir / "bag_trial_1_20260423_120000_000"
    bag_dir.mkdir(parents=True, exist_ok=True)
    (bag_dir / "bag_trial_1_a.mcap").write_bytes(b"a")
    (bag_dir / "bag_trial_1_b.mcap").write_bytes(b"b")
    _bag_paths(eval_dir, trial_indexes=(2, 3))

    with pytest.raises(HarnessIOError, match="exactly one .mcap file"):
        analyze_mcap_eval_bundle(
            eval_dir,
            analyzed_at_utc="2026-04-23T12:00:00Z",
            read_ros2_messages=_fake_reader({}),
        )


def test_analyze_mcap_eval_bundle_rejects_non_official_trial_count(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    _bag_paths(eval_dir, trial_indexes=(1,))

    with pytest.raises(
        HarnessIOError,
        match="official eval bundle must contain exactly trial_1, trial_2, and trial_3",
    ):
        analyze_mcap_eval_bundle(
            eval_dir,
            analyzed_at_utc="2026-04-23T12:00:00Z",
            read_ros2_messages=_fake_reader({}),
        )


def test_reduce_mcap_eval_bundle_binds_artifact_identity_and_provenance(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    _bag_paths(eval_dir)
    reduction = reduce_mcap_eval_bundle(
        eval_dir,
        analyzed_at_utc="2026-04-23T12:00:00Z",
        uri="gs://bucket/eval",
        provenance={"producer": "pytest"},
        read_ros2_messages=_fake_reader({}),
    )

    assert reduction.artifact.kind == "mcap_eval_bundle"
    assert reduction.artifact.path == str(eval_dir.resolve())
    assert reduction.artifact.uri is None
    assert reduction.artifact.provenance["producer"] == "pytest"
    assert reduction.artifact.provenance["declared_uri"] == "gs://bucket/eval"
    assert isinstance(reduction.artifact.provenance["report_sha256"], str)
    assert len(reduction.artifact.provenance["report_sha256"]) == 64
    assert reduction.artifact.sha256 is not None
    assert len(reduction.artifact.sha256) == 64


def test_reduce_mcap_eval_bundle_rejects_conflicting_declared_uri_provenance(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    _bag_paths(eval_dir)

    with pytest.raises(
        HarnessIOError,
        match="provenance.declared_uri must match reducer uri",
    ):
        reduce_mcap_eval_bundle(
            eval_dir,
            analyzed_at_utc="2026-04-23T12:00:00Z",
            uri="gs://bucket/eval",
            provenance={"declared_uri": "gs://bucket/other"},
            read_ros2_messages=_fake_reader({}),
        )


def test_reduce_mcap_eval_bundle_rejects_conflicting_report_sha256_provenance(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    _bag_paths(eval_dir)

    with pytest.raises(
        HarnessIOError,
        match="provenance.report_sha256 must match reducer report",
    ):
        reduce_mcap_eval_bundle(
            eval_dir,
            analyzed_at_utc="2026-04-23T12:00:00Z",
            provenance={"report_sha256": "0" * 64},
            read_ros2_messages=_fake_reader({}),
        )


def test_mcap_eval_reduction_accepts_uri_bundle_identity() -> None:
    report = McapEvalBundleReport(
        source="gs://bucket/eval",
        analyzed_at_utc="2026-04-23T12:00:00Z",
        contact_margin_sec=0.25,
        stop_step_sec=0.05,
        recommended_env={},
        trials=tuple(
            McapEvalTrialReport(
                trial_id=f"trial_{trial_index}",
                source=f"gs://bucket/eval/bag_trial_{trial_index}_20260423_120000_000/bag_trial_{trial_index}_0.mcap",
                size_bytes=120 + trial_index,
                sha256=(hex(trial_index)[2:] * 64)[:64],
                controller_state_count=0,
                pose_command_count=0,
                off_limit_contact_count=0,
            )
            for trial_index in (1, 2, 3)
        ),
    )

    reduction = McapEvalReduction(
        artifact=ArtifactRef(
            kind="mcap_eval_bundle",
            uri=report.source,
            sha256=_bundle_sha256_from_report(report),
            provenance={"report_sha256": _report_sha256(report.to_dict())},
        ),
        report=report,
    )

    assert reduction.artifact.uri == report.source
    assert reduction.artifact.path is None

    with pytest.raises(HarnessIOError, match="provenance.report_sha256"):
        McapEvalReduction(
            artifact=ArtifactRef(
                kind="mcap_eval_bundle",
                uri=report.source,
                sha256=_bundle_sha256_from_report(report),
                provenance={"report_sha256": "0" * 64},
            ),
            report=report,
        )


def test_mcap_eval_reduction_rejects_conflicting_second_identity_for_local_source() -> None:
    report = McapEvalBundleReport(
        source="/tmp/eval",
        analyzed_at_utc="2026-04-23T12:00:00Z",
        contact_margin_sec=0.25,
        stop_step_sec=0.05,
        recommended_env={},
        trials=tuple(
            McapEvalTrialReport(
                trial_id=f"trial_{trial_index}",
                source=f"/tmp/eval/bag_trial_{trial_index}_20260423_120000_000/bag_trial_{trial_index}_0.mcap",
                size_bytes=120 + trial_index,
                sha256=(hex(trial_index)[2:] * 64)[:64],
                controller_state_count=0,
                pose_command_count=0,
                off_limit_contact_count=0,
            )
            for trial_index in (1, 2, 3)
        ),
    )

    with pytest.raises(
        HarnessIOError,
        match="artifact.uri must be unset when report.source is a local path",
    ):
        McapEvalReduction(
            artifact=ArtifactRef(
                kind="mcap_eval_bundle",
                path=report.source,
                uri="gs://bucket/eval",
                sha256="a" * 64,
            ),
            report=report,
        )
