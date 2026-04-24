"""Reducer slice for official AIC MCAP evaluation bundles."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Callable, Iterable, Mapping

from aic_signal_harness.artifacts import HarnessIOError, sha256_file
from aic_signal_harness.mcap_eval import (
    McapEvalBundleReport,
    McapEvalFirstContact,
    McapEvalTaskHints,
    McapEvalTrialReport,
    _is_uri_source,
    _same,
)
from aic_signal_harness.schemas import ArtifactRef, utc_now_iso


_BAG_DIR_RE = r"^bag_trial_([1-9][0-9]*)(?:_|$)"
_OFFICIAL_TRIAL_INDEXES = (1, 2, 3)


@dataclass(frozen=True)
class McapEvalReduction:
    """Typed reduction binding an MCAP eval bundle report to its artifact identity."""

    artifact: ArtifactRef
    report: McapEvalBundleReport

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.artifact, ArtifactRef):
            errors.append("mcap reduction artifact must be an ArtifactRef")
        if not isinstance(self.report, McapEvalBundleReport):
            errors.append("mcap reduction report must be an McapEvalBundleReport")
        if errors:
            raise HarnessIOError("; ".join(errors))
        if self.artifact.kind != "mcap_eval_bundle":
            errors.append("mcap reduction artifact.kind must be 'mcap_eval_bundle'")
        if self.artifact.sha256 is None:
            errors.append("mcap reduction artifact.sha256 must be set")
        errors.extend(_identity_contract_errors(self.artifact, self.report.source))
        expected_sha256 = _bundle_sha256_from_report(self.report)
        if self.artifact.sha256 != expected_sha256:
            errors.append("mcap reduction artifact.sha256 must match report-derived bundle sha256")
        if errors:
            raise HarnessIOError("; ".join(errors))


def analyze_mcap_eval_bundle(
    eval_dir: str | Path,
    *,
    task_hints_by_trial: Mapping[int | str, McapEvalTaskHints | Mapping[str, Any]] | None = None,
    contact_margin_sec: float = 0.25,
    stop_step_sec: float = 0.05,
    analyzed_at_utc: str | None = None,
    read_ros2_messages: Callable[..., Iterable[Any]] | None = None,
) -> McapEvalBundleReport:
    """Analyze an official AIC eval-bag directory into a typed report."""

    reader = read_ros2_messages
    if reader is None:
        try:
            from mcap_ros2.reader import read_ros2_messages as reader
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise HarnessIOError("analyze_mcap_eval_bundle requires mcap_ros2") from exc

    bundle_root = _resolve_existing_dir(eval_dir)
    discovered_trials = _discover_trial_mcap_paths(bundle_root)
    normalized_hints = _normalize_task_hints_by_trial(
        task_hints_by_trial,
        expected_trial_indexes=tuple(index for index, _ in discovered_trials),
    )

    trial_reports = tuple(
        _analyze_trial(
            trial_index=trial_index,
            mcap_path=mcap_path,
            read_ros2_messages=reader,
            task_hints=normalized_hints.get(trial_index),
            contact_margin_sec=contact_margin_sec,
            stop_step_sec=stop_step_sec,
        )
        for trial_index, mcap_path in discovered_trials
    )
    recommended_env = _recommended_env_from_trials(trial_reports)
    return McapEvalBundleReport(
        source=str(bundle_root),
        analyzed_at_utc=utc_now_iso() if analyzed_at_utc is None else analyzed_at_utc,
        contact_margin_sec=contact_margin_sec,
        stop_step_sec=stop_step_sec,
        recommended_env=recommended_env,
        trials=trial_reports,
    )


def reduce_mcap_eval_bundle(
    eval_dir: str | Path,
    *,
    task_hints_by_trial: Mapping[int | str, McapEvalTaskHints | Mapping[str, Any]] | None = None,
    contact_margin_sec: float = 0.25,
    stop_step_sec: float = 0.05,
    uri: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    analyzed_at_utc: str | None = None,
    read_ros2_messages: Callable[..., Iterable[Any]] | None = None,
) -> McapEvalReduction:
    """Reduce an official AIC MCAP eval bundle into typed evidence and artifact identity."""

    report = analyze_mcap_eval_bundle(
        eval_dir,
        task_hints_by_trial=task_hints_by_trial,
        contact_margin_sec=contact_margin_sec,
        stop_step_sec=stop_step_sec,
        analyzed_at_utc=analyzed_at_utc,
        read_ros2_messages=read_ros2_messages,
    )
    return McapEvalReduction(
        artifact=ArtifactRef(
            kind="mcap_eval_bundle",
            path=report.source,
            sha256=_bundle_sha256_from_report(report),
            provenance=_with_declared_uri_provenance(provenance, uri),
        ),
        report=report,
    )


def _analyze_trial(
    *,
    trial_index: int,
    mcap_path: Path,
    read_ros2_messages: Callable[..., Iterable[Any]],
    task_hints: McapEvalTaskHints | None,
    contact_margin_sec: float,
    stop_step_sec: float,
) -> McapEvalTrialReport:
    before = _stat_snapshot(mcap_path)
    controller_states = _controller_states(read_ros2_messages, mcap_path)
    pose_commands = _pose_commands(read_ros2_messages, mcap_path)
    off_limit_contacts = _off_limit_contacts(read_ros2_messages, mcap_path)
    sha256 = sha256_file(mcap_path)
    after = _stat_snapshot(mcap_path)
    _assert_same_file_identity(
        mcap_path,
        before,
        after,
        "mcap file changed during analysis",
    )

    first_contact = _first_contact_evidence(
        controller_states=controller_states,
        pose_commands=pose_commands,
        off_limit_contacts=off_limit_contacts,
        task_hints=task_hints,
        contact_margin_sec=contact_margin_sec,
        stop_step_sec=stop_step_sec,
    )
    size_bytes = before.st_size
    return McapEvalTrialReport(
        trial_id=f"trial_{trial_index}",
        source=str(mcap_path),
        size_bytes=size_bytes,
        sha256=sha256,
        controller_state_count=len(controller_states),
        pose_command_count=len(pose_commands),
        off_limit_contact_count=len(off_limit_contacts),
        controller_stamp_start_sec=(
            controller_states[0]["stamp_sec"] if controller_states else None
        ),
        controller_stamp_end_sec=(
            controller_states[-1]["stamp_sec"] if controller_states else None
        ),
        controller_duration_sec=(
            controller_states[-1]["stamp_sec"] - controller_states[0]["stamp_sec"]
            if controller_states
            else None
        ),
        final_tcp_position=(
            controller_states[-1]["tcp_position"] if controller_states else None
        ),
        final_tcp_error=(
            controller_states[-1]["tcp_error"] if controller_states else None
        ),
        task_hints=task_hints,
        first_off_limit_contact=first_contact,
    )


def _controller_states(
    read_ros2_messages: Callable[..., Iterable[Any]],
    mcap_path: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "log_time_ns": _require_log_time_ns(item, mcap_path, "controller_state", index),
            "stamp_sec": _stamp_sec(item.ros_msg, mcap_path, "controller_state", index),
            "tcp_position": _point(
                item.ros_msg.tcp_pose.position,
                mcap_path,
                "controller_state.tcp_pose.position",
                index,
            ),
            "tcp_error": _first_three(
                item.ros_msg.tcp_error,
                mcap_path,
                "controller_state.tcp_error",
                index,
            ),
        }
        for index, item in enumerate(
            _read_topic_messages(
                read_ros2_messages,
                mcap_path,
                topic="/aic_controller/controller_state",
                label="controller_state",
            )
        )
    ]


def _pose_commands(
    read_ros2_messages: Callable[..., Iterable[Any]],
    mcap_path: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "log_time_ns": _require_log_time_ns(item, mcap_path, "pose_commands", index),
            "stamp_sec": _stamp_sec(item.ros_msg, mcap_path, "pose_commands", index),
            "linear": _point(
                item.ros_msg.velocity.linear,
                mcap_path,
                "pose_commands.velocity.linear",
                index,
            ),
            "angular": _point(
                item.ros_msg.velocity.angular,
                mcap_path,
                "pose_commands.velocity.angular",
                index,
            ),
        }
        for index, item in enumerate(
            _read_topic_messages(
                read_ros2_messages,
                mcap_path,
                topic="/aic_controller/pose_commands",
                label="pose_commands",
            )
        )
    ]


def _off_limit_contacts(
    read_ros2_messages: Callable[..., Iterable[Any]],
    mcap_path: Path,
) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    for index, item in enumerate(
        _read_topic_messages(
            read_ros2_messages,
            mcap_path,
            topic="/aic/gazebo/contacts/off_limit",
            label="off_limit_contacts",
        )
    ):
        msg = getattr(item, "ros_msg", None)
        if msg is None:
            raise HarnessIOError(
                f"malformed off_limit_contacts message at {mcap_path}: item {index} is missing ros_msg"
            )
        raw_contacts = getattr(msg, "contacts", None)
        if raw_contacts is None:
            raise HarnessIOError(
                f"malformed off_limit_contacts message at {mcap_path}: item {index} is missing contacts"
            )
        try:
            if not raw_contacts:
                continue
            log_time_ns = _require_log_time_ns(item, mcap_path, "off_limit_contacts", index)
            for contact_index, raw_contact in enumerate(raw_contacts):
                collision1 = str(raw_contact.collision1.name)
                collision2 = str(raw_contact.collision2.name)
                contacts.append(
                    {
                        "log_time_ns": log_time_ns,
                        "collision1": collision1,
                        "collision2": collision2,
                    }
                )
        except (AttributeError, IndexError, TypeError) as exc:
            raise HarnessIOError(
                "malformed off_limit_contacts message at "
                f"{mcap_path}: item {index} has invalid contact structure"
            ) from exc
    return contacts


def _first_contact_evidence(
    *,
    controller_states: list[dict[str, Any]],
    pose_commands: list[dict[str, Any]],
    off_limit_contacts: list[dict[str, Any]],
    task_hints: McapEvalTaskHints | None,
    contact_margin_sec: float,
    stop_step_sec: float,
) -> McapEvalFirstContact | None:
    if not off_limit_contacts:
        return None

    first_contact = off_limit_contacts[0]
    nearest_state = _nearest_by_log_time(controller_states, first_contact["log_time_ns"])
    nearest_command = _nearest_by_log_time(pose_commands, first_contact["log_time_ns"])
    recommended_stop_sec: float | None = None
    nearest_controller_elapsed_sec: float | None = None
    nearest_tcp_position: tuple[float, float, float] | None = None
    nearest_tcp_error: tuple[float, float, float] | None = None
    nearest_command_elapsed_sec: float | None = None
    nearest_command_linear: tuple[float, float, float] | None = None
    nearest_command_angular: tuple[float, float, float] | None = None

    if nearest_state is not None and controller_states:
        nearest_controller_elapsed_sec = (
            nearest_state["stamp_sec"] - controller_states[0]["stamp_sec"]
        )
        nearest_tcp_position = nearest_state["tcp_position"]
        nearest_tcp_error = nearest_state["tcp_error"]
        if _same(task_hints.port_type if task_hints is not None else None, "sc"):
            recommended_stop_sec = _floor_step(
                max(stop_step_sec, nearest_controller_elapsed_sec - contact_margin_sec),
                stop_step_sec,
            )

    if nearest_command is not None and pose_commands:
        nearest_command_elapsed_sec = (
            nearest_command["stamp_sec"] - pose_commands[0]["stamp_sec"]
        )
        nearest_command_linear = nearest_command["linear"]
        nearest_command_angular = nearest_command["angular"]

    return McapEvalFirstContact(
        log_time_ns=first_contact["log_time_ns"],
        collision1=first_contact["collision1"],
        collision2=first_contact["collision2"],
        log_elapsed_sec=_elapsed_from_first(
            controller_states,
            first_contact["log_time_ns"],
        ),
        nearest_controller_elapsed_sec=nearest_controller_elapsed_sec,
        nearest_tcp_position=nearest_tcp_position,
        nearest_tcp_error=nearest_tcp_error,
        nearest_command_elapsed_sec=nearest_command_elapsed_sec,
        nearest_command_linear=nearest_command_linear,
        nearest_command_angular=nearest_command_angular,
        recommended_stop_sec=recommended_stop_sec,
    )


def _read_topic_messages(
    read_ros2_messages: Callable[..., Iterable[Any]],
    mcap_path: Path,
    *,
    topic: str,
    label: str,
) -> list[Any]:
    try:
        return list(read_ros2_messages(mcap_path, topics=[topic]))
    except HarnessIOError:
        raise
    except Exception as exc:
        raise HarnessIOError(
            f"failed to read {label} from {mcap_path}: {exc}"
        ) from exc


def _stamp_sec(msg: Any, mcap_path: Path, label: str, index: int) -> float:
    try:
        stamp = msg.header.stamp
        seconds = float(stamp.sec)
        nanoseconds = float(stamp.nanosec)
    except (AttributeError, TypeError, ValueError) as exc:
        raise HarnessIOError(
            f"malformed {label} message at {mcap_path}: item {index} is missing a valid header stamp"
        ) from exc
    if seconds < 0.0 or nanoseconds < 0.0:
        raise HarnessIOError(
            f"malformed {label} message at {mcap_path}: item {index} has a negative header stamp"
        )
    return seconds + nanoseconds / 1_000_000_000.0


def _point(value: Any, mcap_path: Path, label: str, index: int) -> tuple[float, float, float]:
    try:
        coordinates = (float(value.x), float(value.y), float(value.z))
    except (AttributeError, TypeError, ValueError) as exc:
        raise HarnessIOError(
            f"malformed {label} message at {mcap_path}: item {index} is missing a valid xyz point"
        ) from exc
    return coordinates


def _first_three(
    value: Any,
    mcap_path: Path,
    label: str,
    index: int,
) -> tuple[float, float, float]:
    try:
        coordinates = (float(value[0]), float(value[1]), float(value[2]))
    except (IndexError, TypeError, ValueError) as exc:
        raise HarnessIOError(
            f"malformed {label} message at {mcap_path}: item {index} must expose at least three numeric values"
        ) from exc
    return coordinates


def _require_log_time_ns(item: Any, mcap_path: Path, label: str, index: int) -> int:
    raw_value = getattr(item, "log_time_ns", None)
    if type(raw_value) is not int or raw_value < 0:
        raise HarnessIOError(
            f"malformed {label} message at {mcap_path}: item {index} must set a nonnegative integer log_time_ns"
        )
    return raw_value


def _nearest_by_log_time(
    items: list[dict[str, Any]],
    log_time_ns: int,
) -> dict[str, Any] | None:
    if not items:
        return None
    return min(items, key=lambda item: abs(int(item["log_time_ns"]) - log_time_ns))


def _elapsed_from_first(items: list[dict[str, Any]], log_time_ns: int) -> float | None:
    if not items:
        return None
    return (log_time_ns - int(items[0]["log_time_ns"])) / 1_000_000_000.0


def _floor_step(value: float, step: float) -> float:
    if step <= 0.0:
        return value
    if value <= 0.0:
        return 0.0
    return math.floor((value / step) + 1e-9) * step


def _resolve_existing_dir(path: str | Path) -> Path:
    bundle_path = Path(path).expanduser()
    try:
        resolved = bundle_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HarnessIOError(f"eval bundle directory not found: {bundle_path}") from exc
    if not resolved.is_dir():
        raise HarnessIOError(f"eval bundle path must be a directory: {resolved}")
    return resolved


def _discover_trial_mcap_paths(eval_dir: Path) -> tuple[tuple[int, Path], ...]:
    import re

    bag_dirs: dict[int, Path] = {}
    top_level_mcaps: list[Path] = []
    for entry in eval_dir.iterdir():
        if entry.is_dir():
            match = re.match(_BAG_DIR_RE, entry.name)
            if match is None:
                continue
            trial_index = int(match.group(1))
            if trial_index in bag_dirs:
                raise HarnessIOError(
                    f"eval bundle contains duplicate bag_trial directory index {trial_index}"
                )
            bag_dirs[trial_index] = entry.resolve()
            continue
        if entry.is_file() and entry.suffix == ".mcap":
            top_level_mcaps.append(entry.resolve())

    if bag_dirs and top_level_mcaps:
        raise HarnessIOError(
            "eval bundle layout is ambiguous: found both bag_trial_* directories and top-level .mcap files"
        )
    if bag_dirs:
        _require_official_trial_indexes(tuple(bag_dirs))
        discovered: list[tuple[int, Path]] = []
        for trial_index, bag_dir in sorted(bag_dirs.items()):
            mcaps = sorted(
                entry.resolve()
                for entry in bag_dir.iterdir()
                if entry.is_file() and entry.suffix == ".mcap"
            )
            if len(mcaps) != 1:
                raise HarnessIOError(
                    f"{bag_dir} must contain exactly one .mcap file, found {len(mcaps)}"
                )
            discovered.append((trial_index, mcaps[0]))
        return tuple(discovered)

    if not top_level_mcaps:
        raise HarnessIOError(f"no .mcap files found under eval bundle {eval_dir}")
    parsed_trials: dict[int, Path] = {}
    for mcap_path in top_level_mcaps:
        trial_index = _top_level_trial_index(mcap_path.name)
        if trial_index is None:
            raise HarnessIOError(
                "top-level .mcap files must encode an explicit trial index when more than one file is present"
            )
        if trial_index in parsed_trials:
            raise HarnessIOError(
                f"eval bundle contains duplicate top-level mcap trial index {trial_index}"
            )
        parsed_trials[trial_index] = mcap_path
    _require_official_trial_indexes(tuple(parsed_trials))
    return tuple(sorted(parsed_trials.items()))


def _top_level_trial_index(filename: str) -> int | None:
    import re

    for pattern in (
        r"^bag_trial_([1-9][0-9]*)(?:_|\.|$)",
        r"^trial_([1-9][0-9]*)(?:_|\.|$)",
    ):
        match = re.match(pattern, filename)
        if match is not None:
            return int(match.group(1))
    return None


def _require_official_trial_indexes(trial_indexes: tuple[int, ...]) -> None:
    if tuple(sorted(trial_indexes)) != _OFFICIAL_TRIAL_INDEXES:
        raise HarnessIOError(
            "official eval bundle must contain exactly trial_1, trial_2, and trial_3"
        )


def _normalize_task_hints_by_trial(
    task_hints_by_trial: Mapping[int | str, McapEvalTaskHints | Mapping[str, Any]] | None,
    *,
    expected_trial_indexes: tuple[int, ...],
) -> dict[int, McapEvalTaskHints]:
    if task_hints_by_trial is None:
        return {}
    if not isinstance(task_hints_by_trial, Mapping):
        raise HarnessIOError("task_hints_by_trial must be a mapping")

    normalized: dict[int, McapEvalTaskHints] = {}
    for key, value in task_hints_by_trial.items():
        trial_index = _normalize_trial_hint_key(key)
        if trial_index in normalized:
            raise HarnessIOError(
                f"task_hints_by_trial contains duplicate hints for trial_{trial_index}"
            )
        normalized[trial_index] = (
            value
            if isinstance(value, McapEvalTaskHints)
            else McapEvalTaskHints.from_dict(value)
        )
    unknown_trials = sorted(index for index in normalized if index not in expected_trial_indexes)
    if unknown_trials:
        raise HarnessIOError(
            "task_hints_by_trial contains hints for unknown trials: "
            + ", ".join(f"trial_{index}" for index in unknown_trials)
        )
    return normalized


def _normalize_trial_hint_key(key: int | str) -> int:
    if type(key) is int and key > 0:
        return key
    if isinstance(key, str):
        stripped = key.strip()
        if stripped.isdigit() and int(stripped) > 0:
            return int(stripped)
        if stripped.startswith("trial_") and stripped[6:].isdigit() and int(stripped[6:]) > 0:
            return int(stripped[6:])
    raise HarnessIOError(
        "task_hints_by_trial keys must be positive integers, digit strings, or 'trial_<n>' strings"
    )


def _recommended_env_from_trials(
    trials: tuple[McapEvalTrialReport, ...],
) -> dict[str, str]:
    recommended_values = {
        f"{trial.first_off_limit_contact.recommended_stop_sec:.2f}"
        for trial in trials
        if trial.first_off_limit_contact is not None
        and trial.first_off_limit_contact.recommended_stop_sec is not None
    }
    if not recommended_values:
        return {}
    if len(recommended_values) != 1:
        raise HarnessIOError("conflicting sc stop recommendations across eval bundle trials")
    return {"AIC_LEWM_SC_REPLAY_STOP_SEC": next(iter(recommended_values))}


def _bundle_sha256_from_report(report: McapEvalBundleReport) -> str:
    digest = hashlib.sha256()
    bundle_source = report.source
    bundle_root = Path(bundle_source) if not _is_uri_source(bundle_source) else None
    for trial in report.trials:
        if bundle_root is not None:
            trial_identity = Path(trial.source).relative_to(bundle_root).as_posix()
        else:
            trial_identity = trial.source
        digest.update(trial.trial_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(trial_identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(trial.size_bytes).encode("utf-8"))
        digest.update(b"\0")
        digest.update(trial.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _identity_contract_errors(artifact: ArtifactRef, report_source: str) -> list[str]:
    errors: list[str] = []
    if _is_uri_source(report_source):
        if artifact.path is not None:
            errors.append("mcap reduction artifact.path must be unset when report.source is a uri")
        if artifact.uri != report_source:
            errors.append("mcap reduction artifact.uri must match report.source when report.source is a uri")
        return errors

    if artifact.uri is not None:
        errors.append("mcap reduction artifact.uri must be unset when report.source is a local path")
    if artifact.path != report_source:
        errors.append("mcap reduction artifact.path must match report.source when report.source is a local path")
    return errors


def _with_declared_uri_provenance(
    provenance: Mapping[str, Any] | None,
    uri: str | None,
) -> dict[str, Any]:
    merged = {} if provenance is None else dict(provenance)
    if uri is None:
        return merged
    existing = merged.get("declared_uri")
    if existing is not None and existing != uri:
        raise HarnessIOError("mcap reduction provenance.declared_uri must match reducer uri")
    merged["declared_uri"] = uri
    return merged


def _stat_snapshot(path: Path) -> os.stat_result:
    try:
        return path.stat()
    except FileNotFoundError as exc:
        raise HarnessIOError(f"file not found: {path}") from exc
    except OSError as exc:
        raise HarnessIOError(f"failed to stat {path}: {exc}") from exc


def _assert_same_file_identity(
    path: Path,
    before: os.stat_result,
    after: os.stat_result,
    message: str,
) -> None:
    if (
        before.st_ino != after.st_ino
        or before.st_dev != after.st_dev
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise HarnessIOError(f"{message}: {path}")
