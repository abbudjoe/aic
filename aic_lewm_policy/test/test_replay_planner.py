from pathlib import Path

import h5py
import numpy as np
import pytest

from aic_lewm_policy.replay_planner import DemonstrationReplayPlanner, ReplayDataset
from aic_lewm_policy.schemas import (
    AicObservation,
    FinalInsertionServoConfig,
    FinalServoProfile,
    PolicyRuntimeConfig,
    TaskSpec,
)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warn(self, message):
        self.messages.append(("warn", message))


def _config(dataset: Path) -> PolicyRuntimeConfig:
    return PolicyRuntimeConfig(
        planner_mode="replay",
        control_hz=20.0,
        max_runtime_sec=30.0,
        frame_id="base_link",
        linear_velocity_limit=0.25,
        angular_velocity_limit=2.0,
        feedback_period_sec=1.0,
        require_checkpoint=False,
        goal_dataset_path=None,
        image_size=224,
        history_size=1,
        frameskip=1,
        planning_horizon=1,
        num_action_candidates=1,
        replay_dataset_path=str(dataset),
        replay_hz=20.0,
        replay_time_scale=1.0,
        replay_action_gain=1.0,
        replay_sc_stop_sec=None,
        final_servo=FinalInsertionServoConfig.disabled(),
    )


def _task(**overrides) -> TaskSpec:
    values = {
        "task_id": "task_1",
        "cable_type": "sfp_sc",
        "cable_name": "sfp_sc",
        "plug_type": "sfp",
        "plug_name": "sfp_module",
        "port_type": "sfp",
        "port_name": "sfp_port_0",
        "target_module_name": "nic_card_mount_1",
        "time_limit_sec": 30.0,
    }
    values.update(overrides)
    return TaskSpec(**values)


def _write_dataset(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ep_len", data=np.asarray([2, 3], dtype=np.int32))
        handle.create_dataset("ep_offset", data=np.asarray([0, 2], dtype=np.int64))
        handle.create_dataset(
            "action",
            data=np.asarray(
                [
                    [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.4, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset("task_id", data=np.asarray(["task_1"] * 5, dtype=dtype))
        handle.create_dataset("plug_type", data=np.asarray(["sfp"] * 5, dtype=dtype))
        handle.create_dataset("port_type", data=np.asarray(["sfp"] * 5, dtype=dtype))
        handle.create_dataset(
            "target_module_name",
            data=np.asarray(
                [
                    "nic_card_mount_0",
                    "nic_card_mount_0",
                    "nic_card_mount_1",
                    "nic_card_mount_1",
                    "nic_card_mount_1",
                ],
                dtype=dtype,
            ),
        )


def _write_sc_dataset(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ep_len", data=np.asarray([3], dtype=np.int32))
        handle.create_dataset("ep_offset", data=np.asarray([0], dtype=np.int64))
        handle.create_dataset(
            "action",
            data=np.asarray(
                [
                    [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset("task_id", data=np.asarray(["task_1"] * 3, dtype=dtype))
        handle.create_dataset("plug_type", data=np.asarray(["sc"] * 3, dtype=dtype))
        handle.create_dataset("port_type", data=np.asarray(["sc"] * 3, dtype=dtype))
        handle.create_dataset(
            "target_module_name",
            data=np.asarray(["sc_port_1"] * 3, dtype=dtype),
        )


def test_replay_selects_exact_module_match(tmp_path):
    dataset_path = tmp_path / "demo.h5"
    _write_dataset(dataset_path)
    planner = DemonstrationReplayPlanner(
        dataset=ReplayDataset.from_hdf5(dataset_path),
        config=_config(dataset_path),
        logger=_Logger(),
    )

    planner.reset(_task(target_module_name="nic_card_mount_1"))

    assert planner.selected_episode is not None
    assert planner.selected_episode.episode_index == 1
    action = planner.select_action(task=_task(), observation=None, elapsed_sec=0.051)
    assert action.frame_id == "base_link"
    assert action.linear == pytest.approx((0.0, 0.4, 0.0))


def test_replay_falls_back_to_zero_after_episode(tmp_path):
    dataset_path = tmp_path / "demo.h5"
    _write_dataset(dataset_path)
    planner = DemonstrationReplayPlanner(
        dataset=ReplayDataset.from_hdf5(dataset_path),
        config=_config(dataset_path),
        logger=_Logger(),
    )

    planner.reset(_task(target_module_name="nic_card_mount_0"))
    action = planner.select_action(task=_task(), observation=None, elapsed_sec=10.0)

    assert action.linear == (0.0, 0.0, 0.0)
    assert action.angular == (0.0, 0.0, 0.0)


def test_sfp_final_servo_engages_after_replay_when_force_is_safe(tmp_path):
    dataset_path = tmp_path / "demo.h5"
    _write_dataset(dataset_path)
    config = _config(dataset_path)
    config = PolicyRuntimeConfig(
        **{
            **config.__dict__,
            "replay_hz": 10.0,
            "final_servo": FinalInsertionServoConfig(
                enabled=True,
                duration_sec=1.25,
                force_guard_n=18.0,
                force_guard_mode="absolute",
                sfp=FinalServoProfile(linear=(0.0, 0.0, -0.02), angular=(0.0, 0.0, 0.0)),
                sc=FinalServoProfile.zero(),
            ),
        }
    )
    planner = DemonstrationReplayPlanner(
        dataset=ReplayDataset.from_hdf5(dataset_path),
        config=config,
        logger=_Logger(),
    )

    planner.reset(_task(target_module_name="nic_card_mount_0"))
    action = planner.select_action(
        task=_task(),
        observation=_observation_with_force((0.0, 0.0, 2.0)),
        elapsed_sec=0.21,
    )

    assert action.linear == pytest.approx((0.0, 0.0, -0.02))
    assert action.angular == (0.0, 0.0, 0.0)


def test_final_servo_holds_when_force_guard_trips(tmp_path):
    dataset_path = tmp_path / "demo.h5"
    _write_dataset(dataset_path)
    config = _config(dataset_path)
    config = PolicyRuntimeConfig(
        **{
            **config.__dict__,
            "replay_hz": 10.0,
            "final_servo": FinalInsertionServoConfig(
                enabled=True,
                duration_sec=1.25,
                force_guard_n=18.0,
                force_guard_mode="absolute",
                sfp=FinalServoProfile(linear=(0.0, 0.0, -0.02), angular=(0.0, 0.0, 0.0)),
                sc=FinalServoProfile.zero(),
            ),
        }
    )
    planner = DemonstrationReplayPlanner(
        dataset=ReplayDataset.from_hdf5(dataset_path),
        config=config,
        logger=_Logger(),
    )

    planner.reset(_task(target_module_name="nic_card_mount_0"))
    action = planner.select_action(
        task=_task(),
        observation=_observation_with_force((0.0, 0.0, 20.0)),
        elapsed_sec=0.21,
    )

    assert action.linear == (0.0, 0.0, 0.0)
    assert action.angular == (0.0, 0.0, 0.0)


def test_delta_force_guard_allows_static_load_and_trips_on_change(tmp_path):
    dataset_path = tmp_path / "demo.h5"
    _write_dataset(dataset_path)
    config = _config(dataset_path)
    config = PolicyRuntimeConfig(
        **{
            **config.__dict__,
            "replay_hz": 10.0,
            "final_servo": FinalInsertionServoConfig(
                enabled=True,
                duration_sec=1.25,
                force_guard_n=4.0,
                force_guard_mode="delta",
                sfp=FinalServoProfile(linear=(0.0, 0.0, -0.02), angular=(0.0, 0.0, 0.0)),
                sc=FinalServoProfile.zero(),
            ),
        }
    )
    logger = _Logger()
    planner = DemonstrationReplayPlanner(
        dataset=ReplayDataset.from_hdf5(dataset_path),
        config=config,
        logger=logger,
    )

    planner.reset(_task(target_module_name="nic_card_mount_0"))
    static_load = planner.select_action(
        task=_task(),
        observation=_observation_with_force((0.0, -10.0, 18.0)),
        elapsed_sec=0.21,
    )
    increased_load = planner.select_action(
        task=_task(),
        observation=_observation_with_force((0.0, -10.0, 23.0)),
        elapsed_sec=0.22,
    )

    assert static_load.linear == pytest.approx((0.0, 0.0, -0.02))
    assert increased_load.linear == (0.0, 0.0, 0.0)
    assert any("force baseline captured" in message for _, message in logger.messages)
    assert any("mode=delta" in message for _, message in logger.messages)


def test_sc_replay_safety_stop_holds_after_configured_time(tmp_path):
    dataset_path = tmp_path / "sc_demo.h5"
    _write_sc_dataset(dataset_path)
    config = _config(dataset_path)
    config = PolicyRuntimeConfig(
        **{
            **config.__dict__,
            "replay_hz": 10.0,
            "replay_sc_stop_sec": 0.15,
        }
    )
    planner = DemonstrationReplayPlanner(
        dataset=ReplayDataset.from_hdf5(dataset_path),
        config=config,
        logger=_Logger(),
    )

    planner.reset(
        _task(
            plug_type="sc",
            port_type="sc",
            target_module_name="sc_port_1",
        )
    )

    before_stop = planner.select_action(task=_task(), observation=None, elapsed_sec=0.10)
    after_stop = planner.select_action(task=_task(), observation=None, elapsed_sec=0.15)

    assert before_stop.linear == pytest.approx((0.0, 0.2, 0.0))
    assert after_stop.linear == (0.0, 0.0, 0.0)
    assert after_stop.angular == (0.0, 0.0, 0.0)


def test_replay_mode_env_defaults_to_base_frame_and_controller_limits(monkeypatch):
    monkeypatch.setenv("AIC_LEWM_PLANNER_MODE", "replay")
    monkeypatch.setenv("AIC_LEWM_GOAL_DATASET", "/tmp/demo.h5")
    monkeypatch.setenv("AIC_LEWM_COMMAND_FRAME", "")

    config = PolicyRuntimeConfig.from_env()

    assert config.planner_mode == "replay"
    assert config.frame_id == "base_link"
    assert config.linear_velocity_limit == 0.25
    assert config.angular_velocity_limit == 2.0
    assert config.replay_dataset_path == "/tmp/demo.h5"
    assert config.replay_hz == 10.0
    assert config.replay_sc_stop_sec is None


def test_replay_mode_reads_sc_stop_env(monkeypatch):
    monkeypatch.setenv("AIC_LEWM_PLANNER_MODE", "replay")
    monkeypatch.setenv("AIC_LEWM_GOAL_DATASET", "/tmp/demo.h5")
    monkeypatch.setenv("AIC_LEWM_SC_REPLAY_STOP_SEC", "2.75")

    config = PolicyRuntimeConfig.from_env()

    assert config.replay_sc_stop_sec == 2.75


def test_replay_mode_reads_final_servo_env(monkeypatch):
    monkeypatch.setenv("AIC_LEWM_PLANNER_MODE", "replay")
    monkeypatch.setenv("AIC_LEWM_GOAL_DATASET", "/tmp/demo.h5")
    monkeypatch.setenv("AIC_LEWM_FINAL_SERVO_ENABLED", "1")
    monkeypatch.setenv("AIC_LEWM_FINAL_SERVO_DURATION_SEC", "1.5")
    monkeypatch.setenv("AIC_LEWM_FINAL_SERVO_FORCE_GUARD_N", "21")
    monkeypatch.setenv("AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE", "delta")
    monkeypatch.setenv("AIC_LEWM_SFP_FINAL_SERVO_LINEAR", "0.001,0.002,-0.03")

    config = PolicyRuntimeConfig.from_env()

    assert config.final_servo.enabled is True
    assert config.final_servo.duration_sec == 1.5
    assert config.final_servo.force_guard_n == 21.0
    assert config.final_servo.force_guard_mode == "delta"
    assert config.final_servo.sfp.linear == (0.001, 0.002, -0.03)


def _observation_with_force(force_xyz) -> AicObservation:
    state = np.zeros((32,), dtype=np.float32)
    state[19:22] = np.asarray(force_xyz, dtype=np.float32)
    return AicObservation(
        timestamp_sec=0.0,
        images={name: np.zeros((1, 1, 3), dtype=np.uint8) for name in ("left", "center", "right")},
        state=state,
    )
