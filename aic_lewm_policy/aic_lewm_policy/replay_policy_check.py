"""Offline checks for the demonstration replay planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .replay_planner import (
    ReplayDataset,
    final_servo_profile_for_episode,
    final_servo_window_sec_for_episode,
    replay_duration_sec_for_episode,
    replay_stop_sec_for_episode,
)
from .schemas import PolicyRuntimeConfig, TaskSpec


def build_replay_report(dataset_path: str | Path) -> dict[str, Any]:
    dataset_path = Path(dataset_path).expanduser()
    dataset = ReplayDataset.from_hdf5(dataset_path)
    config = PolicyRuntimeConfig.from_env()
    replay_hz = config.replay_hz
    checks = []
    for episode in dataset.episodes:
        task = TaskSpec(
            task_id=episode.task_id,
            cable_type="",
            cable_name="",
            plug_type=episode.plug_type,
            plug_name="",
            port_type=episode.port_type,
            port_name="",
            target_module_name=episode.target_module_name,
            time_limit_sec=30.0,
        )
        selected = dataset.select(task)
        actions = episode.actions[:, :6]
        linear_norm = np.linalg.norm(actions[:, :3], axis=1)
        angular_norm = np.linalg.norm(actions[:, 3:], axis=1)
        duration_sec = replay_duration_sec_for_episode(episode, config)
        stop_sec = replay_stop_sec_for_episode(episode, config)
        servo_window = final_servo_window_sec_for_episode(episode, config)
        servo_profile = final_servo_profile_for_episode(episode, config)
        checks.append(
            {
                "task": {
                    "task_id": task.task_id,
                    "plug_type": task.plug_type,
                    "port_type": task.port_type,
                    "target_module_name": task.target_module_name,
                },
                "selected_episode": int(selected.episode_index),
                "expected_episode": int(episode.episode_index),
                "exact_episode_match": selected.episode_index == episode.episode_index,
                "step_count": int(actions.shape[0]),
                "duration_sec_at_replay_hz": duration_sec,
                "replay_stop_sec": stop_sec,
                "effective_duration_sec": min(duration_sec, stop_sec)
                if stop_sec is not None
                else duration_sec,
                "final_servo": {
                    "enabled": config.final_servo.enabled,
                    "window_sec": list(servo_window) if servo_window is not None else None,
                    "duration_sec": config.final_servo.duration_sec
                    if config.final_servo.enabled
                    else None,
                    "force_guard_n": config.final_servo.force_guard_n
                    if config.final_servo.enabled
                    else None,
                    "force_guard_mode": config.final_servo.force_guard_mode
                    if config.final_servo.enabled
                    else None,
                    "linear": list(servo_profile.linear),
                    "angular": list(servo_profile.angular),
                },
                "linear_norm_max": float(linear_norm.max(initial=0.0)),
                "angular_norm_max": float(angular_norm.max(initial=0.0)),
            }
        )

    exact_count = sum(1 for check in checks if check["exact_episode_match"])
    return {
        "schema_version": 1,
        "dataset": str(dataset_path),
        "episode_count": len(dataset.episodes),
        "replay_hz": replay_hz,
        "checks": checks,
        "score": {
            "exact_selection_fraction": exact_count / max(len(checks), 1),
            "exact_selection_count": exact_count,
        },
    }


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
    parser = argparse.ArgumentParser(description="Offline check for AIC replay planner datasets")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_replay_report(args.dataset)
    if args.output:
        _write_json(args.output, report, overwrite=args.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
