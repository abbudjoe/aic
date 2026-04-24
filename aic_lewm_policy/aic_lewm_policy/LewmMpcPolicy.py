"""Lifecycle-compatible AIC policy entry point for LEWM-based planning."""

from __future__ import annotations

import time
from typing import Any

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

from .actions import CartesianVelocityAction
from .planners import make_planner
from .policy_trace_writer import (
    PolicyTraceWriter,
    action_payload,
    observation_payload,
    task_payload,
)
from .preprocessing import ObservationPreprocessor
from .schemas import PolicyRuntimeConfig, TaskSpec


class LewmMpcPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._config = PolicyRuntimeConfig.from_env()
        self._preprocessor = ObservationPreprocessor()
        self._planner = make_planner(config=self._config, logger=self.get_logger())
        self._trace = PolicyTraceWriter.from_env()
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
        trace_trial_id = self._next_trace_trial_id(task_spec)
        official_trial_id = self._official_trial_id_for_trace(task, trace_trial_id)
        self._emit_trace(
            trial_id=trace_trial_id,
            official_trial_id=official_trial_id,
            event_type="task_started",
            elapsed_sec=0.0,
            payload=task_payload(task_spec),
        )

        runtime_sec = min(self._config.max_runtime_sec, max(task_spec.time_limit_sec, 0.0))
        if runtime_sec <= 0.0:
            self.get_logger().warn("Task has non-positive time limit; publishing one hold command.")
            published_action = self._published_action_or_input(
                self._publish_action(CartesianVelocityAction.zero(self._config.frame_id), move_robot),
                CartesianVelocityAction.zero(self._config.frame_id),
            )
            self._emit_trace(
                trial_id=trace_trial_id,
                official_trial_id=official_trial_id,
                event_type="action_published",
                elapsed_sec=0.0,
                payload=action_payload(published_action),
            )
            self._emit_trace(
                trial_id=trace_trial_id,
                official_trial_id=official_trial_id,
                event_type="task_finished",
                elapsed_sec=0.0,
                payload={"result": True, "reason": "non_positive_time_limit"},
            )
            return True

        start_time_sec: float | None = None
        next_feedback_sec = 0.0
        last_elapsed_sec = 0.0

        while True:
            observation = self._get_preprocessed_observation(get_observation)
            current_time_sec = _observation_or_clock_seconds(observation, self.time_now())
            if start_time_sec is None:
                start_time_sec = current_time_sec
            elapsed_sec = max(0.0, current_time_sec - start_time_sec)
            last_elapsed_sec = elapsed_sec
            if elapsed_sec >= runtime_sec:
                break
            self._emit_trace(
                trial_id=trace_trial_id,
                official_trial_id=official_trial_id,
                event_type="observation",
                elapsed_sec=elapsed_sec,
                payload=observation_payload(observation),
            )

            action = self._planner.select_action(
                task=task_spec,
                observation=observation,
                elapsed_sec=elapsed_sec,
            )
            published_action = self._published_action_or_input(
                self._publish_action(action, move_robot),
                action,
            )
            self._emit_trace(
                trial_id=trace_trial_id,
                official_trial_id=official_trial_id,
                event_type="action_published",
                elapsed_sec=elapsed_sec,
                payload=action_payload(published_action),
            )

            if elapsed_sec >= next_feedback_sec:
                feedback = f"lewm policy running: elapsed={elapsed_sec:.1f}s task={task_spec.plug_type}->{task_spec.port_type}"
                send_feedback(feedback)
                self._emit_trace(
                    trial_id=trace_trial_id,
                    official_trial_id=official_trial_id,
                    event_type="feedback",
                    elapsed_sec=elapsed_sec,
                    payload={"message": feedback},
                )
                next_feedback_sec = elapsed_sec + self._config.feedback_period_sec

            post_action_elapsed_sec = max(
                elapsed_sec,
                _time_seconds(self.time_now()) - start_time_sec,
            )
            remaining_sec = runtime_sec - post_action_elapsed_sec
            if remaining_sec <= 0.0:
                last_elapsed_sec = post_action_elapsed_sec
                break
            time.sleep(min(self._config.control_period_sec, remaining_sec))

        final_action = CartesianVelocityAction.zero(self._config.frame_id)
        published_final_action = self._published_action_or_input(
            self._publish_action(final_action, move_robot),
            final_action,
        )
        self._emit_trace(
            trial_id=trace_trial_id,
            official_trial_id=official_trial_id,
            event_type="action_published",
            elapsed_sec=max(last_elapsed_sec, runtime_sec),
            payload=action_payload(published_final_action),
        )
        self._emit_trace(
            trial_id=trace_trial_id,
            official_trial_id=official_trial_id,
            event_type="task_finished",
            elapsed_sec=max(last_elapsed_sec, runtime_sec),
            payload={"result": True},
        )
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
    ) -> CartesianVelocityAction:
        action = action.clamped(
            linear_limit=self._config.linear_velocity_limit,
            angular_limit=self._config.angular_velocity_limit,
        )
        motion_update = action.to_motion_update(stamp=self.get_clock().now().to_msg())
        move_robot(motion_update=motion_update)
        return action

    def _emit_trace(
        self,
        *,
        trial_id: str,
        official_trial_id: str | None,
        event_type: str,
        elapsed_sec: float,
        payload: dict,
    ) -> None:
        trace = getattr(self, "_trace", None)
        if trace is None:
            return
        trace.emit(
            trial_id=trial_id,
            official_trial_id=official_trial_id,
            event_type=event_type,
            elapsed_sec=elapsed_sec,
            payload=payload,
        )

    def _next_trace_trial_id(self, task_spec: TaskSpec) -> str:
        trace_index = int(getattr(self, "_trace_trial_index", 0)) + 1
        self._trace_trial_index = trace_index
        task_id = task_spec.task_id.strip() or "task"
        return f"{task_id}__policy_call_{trace_index:04d}"

    def _official_trial_id_for_trace(
        self,
        task: Task,
        trace_trial_id: str,
    ) -> str | None:
        trace = getattr(self, "_trace", None)
        if trace is None:
            return None
        return trace.official_trial_id_for(
            trace_trial_id=trace_trial_id,
            policy_call_index=int(getattr(self, "_trace_trial_index", 0)),
            task_official_trial_id=_task_official_trial_id_candidate(task),
        )

    def _published_action_or_input(
        self,
        published_action: CartesianVelocityAction | None,
        input_action: CartesianVelocityAction,
    ) -> CartesianVelocityAction:
        if isinstance(published_action, CartesianVelocityAction):
            return published_action
        return input_action.clamped(
            linear_limit=self._config.linear_velocity_limit,
            angular_limit=self._config.angular_velocity_limit,
        )


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
        result: Any = seconds_nanoseconds()
        seconds, nanos = result
        return float(seconds) + float(nanos) / 1_000_000_000.0
    seconds = getattr(time_obj, "seconds", None)
    if seconds is not None:
        return float(seconds)
    raise TypeError(f"Unsupported ROS time type: {type(time_obj)!r}")


def _task_official_trial_id_candidate(task: Task) -> Any | None:
    value = getattr(task, "official_trial_id", None)
    if value is None:
        return None
    return value
