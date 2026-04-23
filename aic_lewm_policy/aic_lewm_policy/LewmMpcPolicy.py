"""Lifecycle-compatible AIC policy entry point for LEWM-based planning."""

from __future__ import annotations

import time

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

from .actions import CartesianVelocityAction
from .planners import make_planner
from .preprocessing import ObservationPreprocessor
from .schemas import PolicyRuntimeConfig, TaskSpec


class LewmMpcPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._config = PolicyRuntimeConfig.from_env()
        self._preprocessor = ObservationPreprocessor()
        self._planner = make_planner(config=self._config, logger=self.get_logger())
        self.get_logger().info(
            "LewmMpcPolicy configured: "
            f"control_hz={self._config.control_hz}, "
            f"max_runtime_sec={self._config.max_runtime_sec}, "
            f"frame_id={self._config.frame_id}"
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        task_spec = TaskSpec.from_msg(task)
        self._planner.reset(task_spec)
        self.get_logger().info(f"LewmMpcPolicy.insert_cable() task: {task_spec}")

        runtime_sec = min(self._config.max_runtime_sec, max(task_spec.time_limit_sec, 0.0))
        if runtime_sec <= 0.0:
            self.get_logger().warn("Task has non-positive time limit; publishing one hold command.")
            self._publish_action(CartesianVelocityAction.zero(self._config.frame_id), move_robot)
            return True

        start_time_sec: float | None = None
        next_feedback_sec = 0.0

        while True:
            observation = self._get_preprocessed_observation(get_observation)
            current_time_sec = _observation_or_clock_seconds(observation, self.time_now())
            if start_time_sec is None:
                start_time_sec = current_time_sec
            elapsed_sec = max(0.0, current_time_sec - start_time_sec)
            if elapsed_sec >= runtime_sec:
                break

            action = self._planner.select_action(
                task=task_spec,
                observation=observation,
                elapsed_sec=elapsed_sec,
            )
            self._publish_action(action, move_robot)

            if elapsed_sec >= next_feedback_sec:
                send_feedback(
                    f"lewm policy running: elapsed={elapsed_sec:.1f}s task={task_spec.plug_type}->{task_spec.port_type}"
                )
                next_feedback_sec = elapsed_sec + self._config.feedback_period_sec

            post_action_elapsed_sec = max(
                elapsed_sec,
                _time_seconds(self.time_now()) - start_time_sec,
            )
            remaining_sec = runtime_sec - post_action_elapsed_sec
            if remaining_sec <= 0.0:
                break
            time.sleep(min(self._config.control_period_sec, remaining_sec))

        self._publish_action(CartesianVelocityAction.zero(self._config.frame_id), move_robot)
        self.get_logger().info("LewmMpcPolicy.insert_cable() exiting.")
        return True

    def _get_preprocessed_observation(self, get_observation: GetObservationCallback):
        observation_msg = get_observation()
        if observation_msg is None:
            self.get_logger().warn("No observation received; holding command.")
            return None
        try:
            return self._preprocessor.from_msg(observation_msg)
        except Exception as exc:
            self.get_logger().error(f"Observation preprocessing failed: {exc}")
            return None

    def _publish_action(
        self,
        action: CartesianVelocityAction,
        move_robot: MoveRobotCallback,
    ) -> None:
        action = action.clamped(
            linear_limit=self._config.linear_velocity_limit,
            angular_limit=self._config.angular_velocity_limit,
        )
        motion_update = action.to_motion_update(stamp=self.get_clock().now().to_msg())
        move_robot(motion_update=motion_update)


def _observation_or_clock_seconds(observation, current_time) -> float:
    if observation is not None:
        timestamp_sec = getattr(observation, "timestamp_sec", None)
        if timestamp_sec is not None:
            return float(timestamp_sec)
    return _time_seconds(current_time)


def _time_seconds(time_obj) -> float:
    if isinstance(time_obj, (int, float)):
        return float(time_obj)
    nanoseconds = getattr(time_obj, "nanoseconds", None)
    if nanoseconds is not None:
        return float(nanoseconds) / 1_000_000_000.0
    seconds_nanoseconds = getattr(time_obj, "seconds_nanoseconds", None)
    if callable(seconds_nanoseconds):
        seconds, nanos = seconds_nanoseconds()
        return float(seconds) + float(nanos) / 1_000_000_000.0
    seconds = getattr(time_obj, "seconds", None)
    if seconds is not None:
        return float(seconds)
    raise TypeError(f"Unsupported ROS time type: {type(time_obj)!r}")
