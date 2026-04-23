from __future__ import annotations

from pathlib import Path

import numpy as np

from aic_lewm_policy.lewm_mpc import (
    ActionNormalizer,
    GoalLibrary,
    LearnedLewmMpcPlanner,
    image_to_model_tensor,
)
from aic_lewm_policy.schemas import (
    AicObservation,
    FinalInsertionServoConfig,
    PolicyRuntimeConfig,
    TaskSpec,
)


def _write_dataset(path: Path) -> None:
    h5py = __import__("h5py")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ep_len", data=np.asarray([3, 3], dtype=np.int32))
        handle.create_dataset("ep_offset", data=np.asarray([0, 3], dtype=np.int64))
        handle.create_dataset("pixels", data=np.zeros((6, 8, 8, 3), dtype=np.uint8))
        handle.create_dataset("left_pixels", data=np.zeros((6, 8, 8, 3), dtype=np.uint8))
        handle.create_dataset("right_pixels", data=np.zeros((6, 8, 8, 3), dtype=np.uint8))
        handle.create_dataset("state", data=np.zeros((6, 32), dtype=np.float32))
        handle.create_dataset(
            "action",
            data=np.asarray(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -0.01, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -0.02, 0.0, 0.0, 0.0],
                    [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.03, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        handle.create_dataset("task_id", data=np.asarray(["task_a"] * 3 + ["task_b"] * 3, dtype=object), dtype=string_dtype)
        handle.create_dataset("plug_type", data=np.asarray(["sfp"] * 3 + ["sc"] * 3, dtype=object), dtype=string_dtype)
        handle.create_dataset("port_type", data=np.asarray(["sfp"] * 3 + ["sc"] * 3, dtype=object), dtype=string_dtype)
        handle.create_dataset("target_module_name", data=np.asarray(["mod_a"] * 3 + ["mod_b"] * 3, dtype=object), dtype=string_dtype)


def test_goal_library_selects_matching_terminal_example(tmp_path: Path) -> None:
    dataset = tmp_path / "aic.h5"
    _write_dataset(dataset)
    library = GoalLibrary.from_hdf5(dataset)

    selected = library.select(
        TaskSpec(
            task_id="current",
            cable_type="",
            cable_name="",
            plug_type="sc",
            plug_name="",
            port_type="sc",
            port_name="",
            target_module_name="mod_b",
            time_limit_sec=8.0,
        )
    )

    assert selected.task_id == "task_b"
    assert selected.plug_type == "sc"


def test_action_normalizer_uses_frameskip_token_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "aic.h5"
    _write_dataset(dataset)
    normalizer = ActionNormalizer.from_hdf5(dataset, frameskip=2)

    raw = normalizer.repeated_token(np.asarray([0.0, 0.0, -0.01, 0.0, 0.0, 0.0]))
    normalized = normalizer.normalize_token(raw)

    assert raw.shape == (12,)
    assert normalized.shape == (12,)
    assert np.isfinite(normalized).all()


def test_image_to_model_tensor_normalizes_chw() -> None:
    tensor = image_to_model_tensor(np.zeros((8, 10, 3), dtype=np.uint8), image_size=16)

    assert tuple(tensor.shape) == (3, 16, 16)
    assert float(tensor.mean()) < 0.0


def test_learned_planner_returns_clamped_action(tmp_path: Path) -> None:
    import torch

    dataset = tmp_path / "aic.h5"
    _write_dataset(dataset)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(()))

        def get_cost(self, info, action_candidates):
            _ = info
            return action_candidates[:, :, -1, 2].abs()

    class Logger:
        def info(self, message):
            _ = message

        def warn(self, message):
            _ = message

        def error(self, message):
            raise AssertionError(message)

    config = PolicyRuntimeConfig(
        planner_mode="lewm_mpc",
        control_hz=10.0,
        max_runtime_sec=8.0,
        frame_id="gripper/tcp",
        linear_velocity_limit=0.025,
        angular_velocity_limit=0.2,
        feedback_period_sec=1.0,
        require_checkpoint=True,
        goal_dataset_path=str(dataset),
        image_size=16,
        history_size=3,
        frameskip=2,
        planning_horizon=2,
        num_action_candidates=4,
        replay_dataset_path=None,
        replay_hz=20.0,
        replay_time_scale=1.0,
        replay_action_gain=1.0,
        replay_sc_stop_sec=None,
        final_servo=FinalInsertionServoConfig.disabled(),
    )
    planner = LearnedLewmMpcPlanner(
        model=DummyModel(),
        config=config,
        goal_library=GoalLibrary.from_hdf5(dataset),
        action_normalizer=ActionNormalizer.from_hdf5(dataset, frameskip=2),
        logger=Logger(),
    )
    task = TaskSpec("current", "", "", "sfp", "", "sfp", "", "mod_a", 8.0)
    planner.reset(task)
    observation = AicObservation(
        timestamp_sec=0.0,
        images={
            "center": np.zeros((8, 8, 3), dtype=np.uint8),
            "left": np.zeros((8, 8, 3), dtype=np.uint8),
            "right": np.zeros((8, 8, 3), dtype=np.uint8),
        },
        state=np.zeros(32, dtype=np.float32),
    )

    actions = [planner.select_action(task, observation, float(index)) for index in range(3)]

    assert actions[0].linear == (0.0, 0.0, 0.0)
    assert abs(actions[-1].linear[2]) <= config.linear_velocity_limit
