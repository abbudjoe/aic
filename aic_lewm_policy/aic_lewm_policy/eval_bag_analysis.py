"""Offline MCAP analysis for official AIC eval bundles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def analyze_eval_bundle(
    eval_dir: str | Path,
    *,
    replay_check_path: str | Path | None = None,
    contact_margin_sec: float = 0.25,
    stop_step_sec: float = 0.05,
) -> dict[str, Any]:
    try:
        from mcap_ros2.reader import read_ros2_messages
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("eval bag analysis requires mcap_ros2") from exc

    eval_dir = Path(eval_dir).expanduser()
    replay_tasks = _load_replay_tasks(replay_check_path)
    trials = []
    recommendations: dict[str, str] = {}

    for trial_index, mcap_path in enumerate(_mcap_paths(eval_dir), start=1):
        trial_name = f"trial_{trial_index}"
        task = replay_tasks.get(trial_index - 1, {})
        states = _controller_states(read_ros2_messages, mcap_path)
        commands = _pose_commands(read_ros2_messages, mcap_path)
        contacts = _contacts(read_ros2_messages, mcap_path)
        first_contact = contacts[0] if contacts else None
        trial_report: dict[str, Any] = {
            "trial": trial_name,
            "path": str(mcap_path),
            "task": task,
            "controller_state_count": len(states),
            "pose_command_count": len(commands),
            "off_limit_contact_count": len(contacts),
        }

        if states:
            trial_report["controller_stamp_start_sec"] = states[0]["stamp_sec"]
            trial_report["controller_stamp_end_sec"] = states[-1]["stamp_sec"]
            trial_report["controller_duration_sec"] = (
                states[-1]["stamp_sec"] - states[0]["stamp_sec"]
            )
            trial_report["final_tcp_position"] = states[-1]["tcp_position"]
            trial_report["final_tcp_error"] = states[-1]["tcp_error"]

        if first_contact is not None:
            nearest_state = _nearest_by_log_time(states, first_contact["log_time_ns"])
            nearest_command = _nearest_by_log_time(commands, first_contact["log_time_ns"])
            first_contact_report = {
                **first_contact,
                "log_elapsed_sec": _elapsed_from_first(states, first_contact["log_time_ns"]),
            }
            if nearest_state is not None and states:
                state_elapsed = nearest_state["stamp_sec"] - states[0]["stamp_sec"]
                first_contact_report["nearest_controller_elapsed_sec"] = state_elapsed
                first_contact_report["nearest_tcp_position"] = nearest_state["tcp_position"]
                first_contact_report["nearest_tcp_error"] = nearest_state["tcp_error"]
                if _same(task.get("port_type"), "sc"):
                    stop_sec = _floor_step(
                        max(stop_step_sec, state_elapsed - contact_margin_sec),
                        stop_step_sec,
                    )
                    first_contact_report["recommended_stop_sec"] = stop_sec
                    recommendations["AIC_LEWM_SC_REPLAY_STOP_SEC"] = f"{stop_sec:.2f}"
            if nearest_command is not None and commands:
                first_contact_report["nearest_command_elapsed_sec"] = (
                    nearest_command["stamp_sec"] - commands[0]["stamp_sec"]
                )
                first_contact_report["nearest_command_linear"] = nearest_command["linear"]
                first_contact_report["nearest_command_angular"] = nearest_command["angular"]
            trial_report["first_off_limit_contact"] = first_contact_report
        trials.append(trial_report)

    return {
        "schema_version": 1,
        "eval_dir": str(eval_dir),
        "contact_margin_sec": contact_margin_sec,
        "stop_step_sec": stop_step_sec,
        "recommended_env": recommendations,
        "trials": trials,
    }


def _mcap_paths(eval_dir: Path) -> list[Path]:
    paths = sorted(eval_dir.glob("bag_trial_*/*.mcap"))
    if paths:
        return paths
    return sorted(eval_dir.glob("*.mcap"))


def _load_replay_tasks(path: str | Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = {}
    for check in report.get("checks", []):
        expected = check.get("expected_episode")
        if expected is None:
            continue
        tasks[int(expected)] = dict(check.get("task") or {})
    return tasks


def _controller_states(read_ros2_messages, mcap_path: Path) -> list[dict[str, Any]]:
    states = []
    for item in read_ros2_messages(mcap_path, topics=["/aic_controller/controller_state"]):
        msg = item.ros_msg
        states.append(
            {
                "log_time_ns": item.log_time_ns,
                "stamp_sec": _stamp_sec(msg),
                "tcp_position": _point(msg.tcp_pose.position),
                "tcp_error": [float(value) for value in msg.tcp_error[:3]],
            }
        )
    return states


def _pose_commands(read_ros2_messages, mcap_path: Path) -> list[dict[str, Any]]:
    commands = []
    for item in read_ros2_messages(mcap_path, topics=["/aic_controller/pose_commands"]):
        msg = item.ros_msg
        commands.append(
            {
                "log_time_ns": item.log_time_ns,
                "stamp_sec": _stamp_sec(msg),
                "linear": _point(msg.velocity.linear),
                "angular": _point(msg.velocity.angular),
            }
        )
    return commands


def _contacts(read_ros2_messages, mcap_path: Path) -> list[dict[str, Any]]:
    contacts = []
    for item in read_ros2_messages(mcap_path, topics=["/aic/gazebo/contacts/off_limit"]):
        msg = item.ros_msg
        if not msg.contacts:
            continue
        contact = msg.contacts[0]
        contacts.append(
            {
                "log_time_ns": item.log_time_ns,
                "collision1": str(contact.collision1.name),
                "collision2": str(contact.collision2.name),
            }
        )
    return contacts


def _nearest_by_log_time(items: list[dict[str, Any]], log_time_ns: int) -> dict[str, Any] | None:
    if not items:
        return None
    return min(items, key=lambda item: abs(int(item["log_time_ns"]) - log_time_ns))


def _elapsed_from_first(items: list[dict[str, Any]], log_time_ns: int) -> float | None:
    if not items:
        return None
    return (log_time_ns - int(items[0]["log_time_ns"])) / 1_000_000_000.0


def _stamp_sec(msg) -> float:
    stamp = msg.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def _point(value) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _floor_step(value: float, step: float) -> float:
    if step <= 0.0:
        return value
    return math.floor(value / step) * step


def _same(left: Any, right: str) -> bool:
    if left is None:
        return False
    return str(left).strip().lower() == right


def _write_json(path: str | Path, value: dict[str, Any], *, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze official AIC eval MCAP bags")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--replay-check")
    parser.add_argument("--contact-margin-sec", type=float, default=0.25)
    parser.add_argument("--stop-step-sec", type=float, default=0.05)
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_eval_bundle(
        args.eval_dir,
        replay_check_path=args.replay_check,
        contact_margin_sec=args.contact_margin_sec,
        stop_step_sec=args.stop_step_sec,
    )
    if args.output:
        _write_json(args.output, report, overwrite=args.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
