"""Offline learned-policy smoke check against an AIC LEWM HDF5 dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .planners import make_planner
from .schemas import AicObservation, PolicyRuntimeConfig, TaskSpec


class PrintLogger:
    def info(self, message: str) -> None:
        print(f"INFO: {message}")

    def warn(self, message: str) -> None:
        print(f"WARN: {message}")

    def error(self, message: str) -> None:
        print(f"ERROR: {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ["AIC_LEWM_CHECKPOINT"] = str(Path(args.checkpoint).expanduser())
    os.environ["AIC_LEWM_GOAL_DATASET"] = str(Path(args.dataset).expanduser())
    os.environ["AIC_LEWM_DEVICE"] = args.device
    os.environ["AIC_LEWM_REQUIRE_CHECKPOINT"] = "1"
    os.environ["AIC_LEWM_NUM_ACTION_CANDIDATES"] = str(args.candidates)
    os.environ["AIC_LEWM_PLANNING_HORIZON"] = str(args.horizon)

    dataset = Path(args.dataset).expanduser()
    observations, task = _load_observations(dataset, max(args.steps, 1))

    planner = make_planner(PolicyRuntimeConfig.from_env(), PrintLogger())
    planner.reset(task)
    actions = []
    for index, observation in enumerate(observations):
        action = planner.select_action(task, observation, elapsed_sec=index * 0.1)
        actions.append(
            {
                "linear": list(action.linear),
                "angular": list(action.angular),
                "frame_id": action.frame_id,
            }
        )

    print(
        json.dumps(
            {
                "ok": True,
                "checkpoint": os.environ["AIC_LEWM_CHECKPOINT"],
                "dataset": str(dataset),
                "planner": type(planner).__name__,
                "steps": len(observations),
                "actions": actions,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_observations(dataset: Path, steps: int) -> tuple[list[AicObservation], TaskSpec]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency checked elsewhere
        raise RuntimeError("offline policy check requires h5py") from exc

    with h5py.File(dataset, "r") as handle:
        count = min(steps, int(handle["pixels"].shape[0]))
        if count <= 0:
            raise ValueError(f"dataset has no rows: {dataset}")

        task = TaskSpec(
            task_id=_decode(handle["task_id"][0]),
            cable_type="",
            cable_name="",
            plug_type=_decode(handle["plug_type"][0]),
            plug_name="",
            port_type=_decode(handle["port_type"][0]),
            port_name="",
            target_module_name=_decode(handle["target_module_name"][0]),
            time_limit_sec=8.0,
        )
        observations = [
            AicObservation(
                timestamp_sec=float(index),
                images={
                    "center": np.asarray(handle["pixels"][index], dtype=np.uint8),
                    "left": np.asarray(handle["left_pixels"][index], dtype=np.uint8),
                    "right": np.asarray(handle["right_pixels"][index], dtype=np.uint8),
                },
                state=np.asarray(handle["state"][index], dtype=np.float32),
            )
            for index in range(count)
        ]

    return observations, task


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
