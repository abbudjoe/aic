"""Typed policy-side contracts for AIC observations, tasks, and runtime config."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Literal, Mapping, cast

import numpy as np


IMAGE_NAMES: tuple[str, ...] = ("left", "center", "right")

STATE_FIELD_NAMES: tuple[str, ...] = (
    "tcp_pose.position.x",
    "tcp_pose.position.y",
    "tcp_pose.position.z",
    "tcp_pose.orientation.x",
    "tcp_pose.orientation.y",
    "tcp_pose.orientation.z",
    "tcp_pose.orientation.w",
    "tcp_velocity.linear.x",
    "tcp_velocity.linear.y",
    "tcp_velocity.linear.z",
    "tcp_velocity.angular.x",
    "tcp_velocity.angular.y",
    "tcp_velocity.angular.z",
    "tcp_error.x",
    "tcp_error.y",
    "tcp_error.z",
    "tcp_error.rx",
    "tcp_error.ry",
    "tcp_error.rz",
    "wrist_wrench.force.x",
    "wrist_wrench.force.y",
    "wrist_wrench.force.z",
    "wrist_wrench.torque.x",
    "wrist_wrench.torque.y",
    "wrist_wrench.torque.z",
    "joint_positions.0",
    "joint_positions.1",
    "joint_positions.2",
    "joint_positions.3",
    "joint_positions.4",
    "joint_positions.5",
    "joint_positions.6",
)

ACTION_FIELD_NAMES: tuple[str, ...] = (
    "linear.x",
    "linear.y",
    "linear.z",
    "angular.x",
    "angular.y",
    "angular.z",
)


@dataclass(frozen=True)
class TaskSpec:
    """A normalized view of the task request that can be logged and featurized."""

    task_id: str
    cable_type: str
    cable_name: str
    plug_type: str
    plug_name: str
    port_type: str
    port_name: str
    target_module_name: str
    time_limit_sec: float

    @classmethod
    def from_msg(cls, task) -> "TaskSpec":
        return cls(
            task_id=task.id,
            cable_type=task.cable_type,
            cable_name=task.cable_name,
            plug_type=task.plug_type,
            plug_name=task.plug_name,
            port_type=task.port_type,
            port_name=task.port_name,
            target_module_name=task.target_module_name,
            time_limit_sec=float(task.time_limit),
        )

    @property
    def is_sfp(self) -> bool:
        return self.plug_type.lower() == "sfp" or self.port_type.lower() == "sfp"

    @property
    def is_sc(self) -> bool:
        return self.plug_type.lower() == "sc" or self.port_type.lower() == "sc"


@dataclass(frozen=True)
class AicObservation:
    """Preprocessed observation passed across the policy/planner boundary."""

    timestamp_sec: float
    images: Mapping[str, np.ndarray]
    state: np.ndarray

    state_field_names: ClassVar[tuple[str, ...]] = STATE_FIELD_NAMES
    image_names: ClassVar[tuple[str, ...]] = IMAGE_NAMES

    def center_image(self) -> np.ndarray:
        return self.images["center"]


@dataclass(frozen=True)
class FinalServoProfile:
    """A typed post-replay velocity profile for guarded insertion attempts."""

    linear: tuple[float, float, float]
    angular: tuple[float, float, float]

    @classmethod
    def zero(cls) -> "FinalServoProfile":
        return cls(linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))

    def is_zero(self) -> bool:
        return all(abs(value) <= 1e-9 for value in (*self.linear, *self.angular))


@dataclass(frozen=True)
class FinalInsertionServoConfig:
    """Runtime contract for the legal final-centimeter servo phase."""

    enabled: bool
    duration_sec: float
    force_guard_n: float
    force_guard_mode: Literal["absolute", "delta"]
    sfp: FinalServoProfile
    sc: FinalServoProfile

    @classmethod
    def from_env(cls) -> "FinalInsertionServoConfig":
        return cls(
            enabled=_bool_env("AIC_LEWM_FINAL_SERVO_ENABLED", False),
            duration_sec=_float_env("AIC_LEWM_FINAL_SERVO_DURATION_SEC", 1.25),
            force_guard_n=_float_env("AIC_LEWM_FINAL_SERVO_FORCE_GUARD_N", 18.0),
            force_guard_mode=cast(
                Literal["absolute", "delta"],
                _choice_env(
                    "AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE",
                    default="absolute",
                    choices=("absolute", "delta"),
                ),
            ),
            sfp=FinalServoProfile(
                linear=_vector3_env("AIC_LEWM_SFP_FINAL_SERVO_LINEAR", (0.0, 0.0, -0.02)),
                angular=_vector3_env("AIC_LEWM_SFP_FINAL_SERVO_ANGULAR", (0.0, 0.0, 0.0)),
            ),
            sc=FinalServoProfile(
                linear=_vector3_env("AIC_LEWM_SC_FINAL_SERVO_LINEAR", (0.0, 0.0, 0.0)),
                angular=_vector3_env("AIC_LEWM_SC_FINAL_SERVO_ANGULAR", (0.0, 0.0, 0.0)),
            ),
        )

    @classmethod
    def disabled(cls) -> "FinalInsertionServoConfig":
        return cls(
            enabled=False,
            duration_sec=0.0,
            force_guard_n=0.0,
            force_guard_mode="absolute",
            sfp=FinalServoProfile.zero(),
            sc=FinalServoProfile.zero(),
        )


@dataclass(frozen=True)
class PolicyRuntimeConfig:
    """Runtime knobs that should be explicit instead of scattered env lookups."""

    planner_mode: str
    control_hz: float
    max_runtime_sec: float
    frame_id: str
    linear_velocity_limit: float
    angular_velocity_limit: float
    feedback_period_sec: float
    require_checkpoint: bool
    goal_dataset_path: str | None
    image_size: int
    history_size: int
    frameskip: int
    planning_horizon: int
    num_action_candidates: int
    replay_dataset_path: str | None
    replay_hz: float
    replay_time_scale: float
    replay_action_gain: float
    replay_sc_stop_sec: float | None
    final_servo: FinalInsertionServoConfig

    @classmethod
    def from_env(cls) -> "PolicyRuntimeConfig":
        planner_mode = os.getenv("AIC_LEWM_PLANNER_MODE", "lewm_mpc").strip().lower()
        replay_defaults = planner_mode == "replay"
        default_frame = "base_link" if replay_defaults else "gripper/tcp"
        default_linear_limit = 0.25 if replay_defaults else 0.025
        default_angular_limit = 2.0 if replay_defaults else 0.2
        goal_dataset_path = _optional_env("AIC_LEWM_GOAL_DATASET")
        return cls(
            planner_mode=planner_mode,
            control_hz=_float_env("AIC_LEWM_CONTROL_HZ", 10.0),
            max_runtime_sec=_float_env("AIC_LEWM_MAX_RUNTIME_SEC", 8.0),
            frame_id=_string_env("AIC_LEWM_COMMAND_FRAME", default_frame),
            linear_velocity_limit=_float_env("AIC_LEWM_LINEAR_VEL_LIMIT", default_linear_limit),
            angular_velocity_limit=_float_env("AIC_LEWM_ANGULAR_VEL_LIMIT", default_angular_limit),
            feedback_period_sec=_float_env("AIC_LEWM_FEEDBACK_PERIOD_SEC", 1.0),
            require_checkpoint=_bool_env("AIC_LEWM_REQUIRE_CHECKPOINT", False),
            goal_dataset_path=goal_dataset_path,
            image_size=_int_env("AIC_LEWM_IMAGE_SIZE", 224),
            history_size=_int_env("AIC_LEWM_HISTORY_SIZE", 3),
            frameskip=_int_env("AIC_LEWM_FRAMESKIP", 2),
            planning_horizon=_int_env("AIC_LEWM_PLANNING_HORIZON", 5),
            num_action_candidates=_int_env("AIC_LEWM_NUM_ACTION_CANDIDATES", 32),
            replay_dataset_path=_optional_env("AIC_LEWM_REPLAY_DATASET") or goal_dataset_path,
            replay_hz=_float_env("AIC_LEWM_REPLAY_HZ", 10.0),
            replay_time_scale=_float_env("AIC_LEWM_REPLAY_TIME_SCALE", 1.0),
            replay_action_gain=_float_env("AIC_LEWM_REPLAY_ACTION_GAIN", 1.0),
            replay_sc_stop_sec=_optional_float_env("AIC_LEWM_SC_REPLAY_STOP_SEC"),
            final_servo=FinalInsertionServoConfig.from_env(),
        )

    @property
    def control_period_sec(self) -> float:
        return 1.0 / max(self.control_hz, 1e-6)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _optional_float_env(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _vector3_env(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        return default
    try:
        return tuple(float(part) for part in parts)  # type: ignore[return-value]
    except ValueError:
        return default


def _string_env(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw


def _choice_env(name: str, *, default: str, choices: tuple[str, ...]) -> str:
    raw = _string_env(name, default).strip().lower()
    if raw in choices:
        return raw
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
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _optional_env(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    return raw
