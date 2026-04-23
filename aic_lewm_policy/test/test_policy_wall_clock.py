from __future__ import annotations

from types import SimpleNamespace

import pytest

from aic_lewm_policy.actions import CartesianVelocityAction
from aic_lewm_policy.LewmMpcPolicy import LewmMpcPolicy
from aic_lewm_policy.schemas import FinalInsertionServoConfig, PolicyRuntimeConfig


class _Logger:
    def info(self, message):
        _ = message

    def warn(self, message):
        _ = message

    def error(self, message):
        raise AssertionError(message)


def test_insert_cable_uses_observation_time_and_wall_throttle(monkeypatch) -> None:
    module = __import__("aic_lewm_policy.LewmMpcPolicy", fromlist=["time"])
    clock = {"now": 0.0}
    sleeps: list[float] = []
    elapsed_values: list[float] = []
    published: list[CartesianVelocityAction] = []

    def fake_sleep(duration_sec: float) -> None:
        sleeps.append(duration_sec)
        clock["now"] += duration_sec

    monkeypatch.setattr(module.time, "sleep", fake_sleep)

    class Planner:
        def reset(self, task):
            _ = task

        def select_action(self, task, observation, elapsed_sec):
            _ = task, observation
            elapsed_values.append(elapsed_sec)
            clock["now"] += 0.22
            return CartesianVelocityAction(
                linear=(0.0, 0.0, -0.01),
                angular=(0.0, 0.0, 0.0),
                frame_id="gripper/tcp",
            )

    policy = object.__new__(LewmMpcPolicy)
    policy._config = PolicyRuntimeConfig(
        planner_mode="lewm_mpc",
        control_hz=10.0,
        max_runtime_sec=0.55,
        frame_id="gripper/tcp",
        linear_velocity_limit=0.025,
        angular_velocity_limit=0.2,
        feedback_period_sec=1.0,
        require_checkpoint=False,
        goal_dataset_path=None,
        image_size=16,
        history_size=3,
        frameskip=2,
        planning_horizon=1,
        num_action_candidates=4,
        replay_dataset_path=None,
        replay_hz=20.0,
        replay_time_scale=1.0,
        replay_action_gain=1.0,
        replay_sc_stop_sec=None,
        final_servo=FinalInsertionServoConfig.disabled(),
    )
    policy._planner = Planner()
    policy.get_logger = lambda: _Logger()
    policy._get_preprocessed_observation = lambda get_observation: SimpleNamespace(
        timestamp_sec=clock["now"]
    )
    policy._publish_action = lambda action, move_robot: published.append(action)
    policy.sleep_for = lambda duration_sec: (_ for _ in ()).throw(
        AssertionError(f"sim-clock sleep was used: {duration_sec}")
    )
    policy.time_now = lambda: clock["now"]

    task = SimpleNamespace(
        id="task_1",
        cable_type="sfp_sc",
        cable_name="cable_0",
        plug_type="sfp",
        plug_name="sfp_tip",
        port_type="sfp",
        port_name="sfp_port_0",
        target_module_name="nic_card_mount_0",
        time_limit=180.0,
    )

    assert policy.insert_cable(task, lambda: object(), lambda **_: None, lambda _: None)

    assert elapsed_values == [0.0, 0.32]
    assert sleeps == pytest.approx([0.1, 0.01])
    assert published[-1].linear == (0.0, 0.0, 0.0)
