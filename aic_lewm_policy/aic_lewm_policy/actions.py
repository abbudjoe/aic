"""Motion command helpers for the AIC controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from geometry_msgs.msg import Twist, Vector3, Wrench
from std_msgs.msg import Header


@dataclass(frozen=True)
class CartesianVelocityAction:
    linear: tuple[float, float, float]
    angular: tuple[float, float, float]
    frame_id: str = "gripper/tcp"

    @classmethod
    def zero(cls, frame_id: str = "gripper/tcp") -> "CartesianVelocityAction":
        return cls(linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0), frame_id=frame_id)

    def clamped(
        self,
        linear_limit: float,
        angular_limit: float,
    ) -> "CartesianVelocityAction":
        linear = np.clip(np.asarray(self.linear, dtype=np.float64), -linear_limit, linear_limit)
        angular = np.clip(
            np.asarray(self.angular, dtype=np.float64), -angular_limit, angular_limit
        )
        return CartesianVelocityAction(
            linear=tuple(float(x) for x in linear),
            angular=tuple(float(x) for x in angular),
            frame_id=self.frame_id,
        )

    def to_motion_update(
        self,
        stamp,
        stiffness: tuple[float, ...] = (80.0, 80.0, 80.0, 35.0, 35.0, 35.0),
        damping: tuple[float, ...] = (70.0, 70.0, 70.0, 18.0, 18.0, 18.0),
    ) -> MotionUpdate:
        msg = MotionUpdate()
        msg.header = Header(frame_id=self.frame_id, stamp=stamp)
        msg.velocity = Twist(
            linear=Vector3(
                x=float(self.linear[0]),
                y=float(self.linear[1]),
                z=float(self.linear[2]),
            ),
            angular=Vector3(
                x=float(self.angular[0]),
                y=float(self.angular[1]),
                z=float(self.angular[2]),
            ),
        )
        msg.target_stiffness = _diag(stiffness)
        msg.target_damping = _diag(damping)
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        return msg


def _diag(values: tuple[float, ...]) -> list[float]:
    if len(values) != 6:
        raise ValueError(f"Expected 6 diagonal values, got {len(values)}")
    return np.diag(np.asarray(values, dtype=np.float64)).flatten().tolist()
