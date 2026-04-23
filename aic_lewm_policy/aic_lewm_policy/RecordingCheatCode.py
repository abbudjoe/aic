"""Ground-truth rollout recorder for building the first AIC LEWM dataset."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from aic_example_policies.ros.CheatCode import CheatCode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from rclpy.time import Time
from tf2_ros import TransformException

from .dataset_schema import AicEpisodeRecord, AicStepRecord, write_lewm_hdf5
from .preprocessing import ObservationPreprocessor
from .schemas import AicObservation, TaskSpec


class RecordingCheatCode(CheatCode):
    """Run CheatCode and write sampled observations/actions as LEWM HDF5.

    This policy is for simulation data generation only. It uses ground-truth TF
    through `CheatCode`, so it must not be used as a submitted evaluation policy.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._episodes: list[AicEpisodeRecord] = []
        self._current_steps: list[AicStepRecord] = []
        self._recording_get_observation: GetObservationCallback | None = None
        self._recording_task: TaskSpec | None = None
        self._recording_active = False
        self._last_record_time_sec: float | None = None
        self._preprocessor = ObservationPreprocessor()
        self._output_path = Path(
            os.getenv(
                "AIC_LEWM_DATASET_PATH",
                str(Path.home() / "aic_results" / "aic_qualification_train.h5"),
            )
        ).expanduser()
        self._record_hz = _float_env("AIC_LEWM_RECORD_HZ", 10.0)
        self._image_scale = _float_env("AIC_LEWM_RECORD_IMAGE_SCALE", 0.25)
        self._flush_every = _int_env("AIC_LEWM_RECORD_FLUSH_EVERY", 25)
        self._use_upstream_cheatcode = _bool_env(
            "AIC_LEWM_USE_UPSTREAM_CHEATCODE", False
        )
        self._approach_steps = _int_env("AIC_LEWM_CHEATCODE_APPROACH_STEPS", 50)
        self._approach_sleep_sec = _float_env(
            "AIC_LEWM_CHEATCODE_APPROACH_SLEEP", 0.03
        )
        self._insert_z_step = _float_env("AIC_LEWM_CHEATCODE_INSERT_Z_STEP", 0.002)
        self._insert_sleep_sec = _float_env("AIC_LEWM_CHEATCODE_INSERT_SLEEP", 0.03)
        self._insert_z_min = _float_env("AIC_LEWM_CHEATCODE_INSERT_Z_MIN", -0.015)
        self._stabilize_sleep_sec = _float_env(
            "AIC_LEWM_CHEATCODE_STABILIZE_SLEEP", 2.0
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self._recording_get_observation = get_observation
        self._recording_task = TaskSpec.from_msg(task)
        self._current_steps = []
        self._recording_active = False
        self._last_record_time_sec = None

        def recording_move_robot(motion_update=None, joint_motion_update=None) -> None:
            self._recording_active = True
            move_robot(
                motion_update=motion_update,
                joint_motion_update=joint_motion_update,
            )

        try:
            if self._use_upstream_cheatcode:
                success = super().insert_cable(
                    task=task,
                    get_observation=get_observation,
                    move_robot=recording_move_robot,
                    send_feedback=send_feedback,
                )
            else:
                success = self._insert_cable_for_recording(
                    task=task,
                    get_observation=get_observation,
                    move_robot=recording_move_robot,
                    send_feedback=send_feedback,
                )
        finally:
            self._record_current_observation(force=True)
            self._finish_episode()
            self._recording_get_observation = None
            self._recording_task = None
            self._recording_active = False

        return success

    def sleep_for(self, duration_sec: float) -> None:
        super().sleep_for(duration_sec)
        self._record_current_observation()

    def _record_current_observation(self, force: bool = False) -> None:
        if (
            not self._recording_active
            or self._recording_get_observation is None
            or self._recording_task is None
        ):
            return

        now_sec = _clock_seconds(self.time_now())
        min_period_sec = 1.0 / max(self._record_hz, 1e-6)
        if (
            not force
            and self._last_record_time_sec is not None
            and now_sec - self._last_record_time_sec < min_period_sec
        ):
            return

        obs_msg = self._recording_get_observation()
        if obs_msg is None:
            return

        try:
            observation = self._preprocessor.from_msg(obs_msg)
        except Exception as exc:
            self.get_logger().warn(f"Skipping LEWM sample: {exc}")
            return

        observation = _scale_observation_images(observation, self._image_scale)
        action = _executed_cartesian_velocity(obs_msg)
        self._current_steps.append(
            AicStepRecord(
                observation=observation,
                action=action,
                task=self._recording_task,
            )
        )
        self._last_record_time_sec = now_sec
        if (
            self._flush_every > 0
            and len(self._current_steps) % self._flush_every == 0
        ):
            self._write_dataset_snapshot(include_current_episode=True, partial=True)

    def _finish_episode(self) -> None:
        if not self._current_steps:
            self.get_logger().warn("No LEWM samples collected for this task")
            return

        self._episodes.append(AicEpisodeRecord(steps=list(self._current_steps)))
        self._write_dataset_snapshot(include_current_episode=False, partial=False)
        self.get_logger().info(
            "Wrote %d LEWM episode(s), %d current samples to %s"
            % (len(self._episodes), len(self._current_steps), self._output_path)
        )

    def _write_dataset_snapshot(
        self,
        *,
        include_current_episode: bool,
        partial: bool,
    ) -> None:
        episodes = list(self._episodes)
        if include_current_episode and self._current_steps:
            episodes.append(AicEpisodeRecord(steps=list(self._current_steps)))
        if not episodes:
            return

        write_lewm_hdf5(episodes, self._output_path)
        if partial:
            self.get_logger().info(
                "Wrote partial LEWM snapshot with %d episode(s), %d current samples to %s"
                % (len(episodes), len(self._current_steps), self._output_path)
            )

    def _insert_cable_for_recording(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        del get_observation, send_feedback
        self.get_logger().info(f"RecordingCheatCode.insert_cable() task: {task}")
        self._task = task

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"

        for frame in [port_frame, cable_tip_frame]:
            if not self._wait_for_tf("base_link", frame):
                return False

        try:
            port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                port_frame,
                Time(),
            )
        except TransformException as exc:
            self.get_logger().error(f"Could not look up port transform: {exc}")
            return False

        port_transform = port_tf_stamped.transform
        z_offset = 0.2
        approach_steps = max(self._approach_steps, 1)

        for t in range(approach_steps):
            interp_fraction = t / float(approach_steps)
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(
                        port_transform,
                        slerp_fraction=interp_fraction,
                        position_fraction=interp_fraction,
                        z_offset=z_offset,
                        reset_xy_integrator=True,
                    ),
                )
            except TransformException as exc:
                self.get_logger().warn(
                    f"TF lookup failed during interpolation: {exc}"
                )
            self.sleep_for(self._approach_sleep_sec)

        while z_offset >= self._insert_z_min:
            z_offset -= self._insert_z_step
            self.get_logger().info(f"z_offset: {z_offset:0.5}")
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(port_transform, z_offset=z_offset),
                )
            except TransformException as exc:
                self.get_logger().warn(f"TF lookup failed during insertion: {exc}")
            self.sleep_for(self._insert_sleep_sec)

        self.get_logger().info("Waiting for connector to stabilize...")
        self.sleep_for(self._stabilize_sleep_sec)
        self.get_logger().info("RecordingCheatCode.insert_cable() exiting...")
        return True


def _executed_cartesian_velocity(obs_msg) -> np.ndarray:
    velocity = obs_msg.controller_state.tcp_velocity
    return np.asarray(
        [
            velocity.linear.x,
            velocity.linear.y,
            velocity.linear.z,
            velocity.angular.x,
            velocity.angular.y,
            velocity.angular.z,
        ],
        dtype=np.float32,
    )


def _scale_observation_images(
    observation: AicObservation,
    scale: float,
) -> AicObservation:
    if scale == 1.0:
        return observation

    images = {
        name: _resize_image(image, scale) for name, image in observation.images.items()
    }
    return replace(observation, images=images)


def _resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("AIC_LEWM_RECORD_IMAGE_SCALE must be positive")
    if scale == 1.0:
        return image

    try:
        import cv2
    except ImportError:
        step = round(1.0 / scale)
        if step <= 0 or abs(scale - (1.0 / step)) > 1e-6:
            raise ImportError(
                "OpenCV is required for non-reciprocal image scales"
            ) from None
        return np.ascontiguousarray(image[::step, ::step])

    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _clock_seconds(time_obj) -> float:
    nanoseconds = getattr(time_obj, "nanoseconds", None)
    if nanoseconds is not None:
        return float(nanoseconds) * 1e-9
    return 0.0


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}
