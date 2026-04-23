"""Conversion from ROS AIC messages into explicit arrays for LEWM planning."""

from __future__ import annotations

import numpy as np

from .schemas import AicObservation, STATE_FIELD_NAMES


class ObservationPreprocessor:
    """Build the exact tensors/arrays that training and runtime must share."""

    def from_msg(self, msg) -> AicObservation:
        images = {
            "left": image_msg_to_rgb_array(msg.left_image),
            "center": image_msg_to_rgb_array(msg.center_image),
            "right": image_msg_to_rgb_array(msg.right_image),
        }
        return AicObservation(
            timestamp_sec=stamp_to_seconds(msg.center_image.header.stamp),
            images=images,
            state=state_vector_from_msg(msg),
        )


def stamp_to_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1e9


def image_msg_to_rgb_array(image_msg) -> np.ndarray:
    """Return an HWC uint8 RGB-like array without requiring cv_bridge."""

    height = int(image_msg.height)
    width = int(image_msg.width)
    if height <= 0 or width <= 0 or not image_msg.data:
        return np.zeros((max(height, 1), max(width, 1), 3), dtype=np.uint8)

    row_step = int(getattr(image_msg, "step", 0)) or width * 3
    raw = np.frombuffer(image_msg.data, dtype=np.uint8)
    expected = height * row_step
    if raw.size < expected:
        padded = np.zeros(expected, dtype=np.uint8)
        padded[: raw.size] = raw
        raw = padded
    else:
        raw = raw[:expected]

    rows = raw.reshape(height, row_step)
    channels = max(row_step // width, 1)
    pixels = rows[:, : width * channels].reshape(height, width, channels)

    if channels == 1:
        pixels = np.repeat(pixels, repeats=3, axis=2)
    elif channels >= 3:
        pixels = pixels[:, :, :3]
    else:
        padded = np.zeros((height, width, 3), dtype=np.uint8)
        padded[:, :, :channels] = pixels
        pixels = padded

    return np.ascontiguousarray(pixels, dtype=np.uint8)


def state_vector_from_msg(msg) -> np.ndarray:
    controller_state = msg.controller_state
    tcp_pose = controller_state.tcp_pose
    tcp_velocity = controller_state.tcp_velocity
    tcp_error = _fixed_len(controller_state.tcp_error, 6)
    joint_positions = _fixed_len(msg.joint_states.position, 7)
    wrench = msg.wrist_wrench.wrench

    values = np.array(
        [
            tcp_pose.position.x,
            tcp_pose.position.y,
            tcp_pose.position.z,
            tcp_pose.orientation.x,
            tcp_pose.orientation.y,
            tcp_pose.orientation.z,
            tcp_pose.orientation.w,
            tcp_velocity.linear.x,
            tcp_velocity.linear.y,
            tcp_velocity.linear.z,
            tcp_velocity.angular.x,
            tcp_velocity.angular.y,
            tcp_velocity.angular.z,
            *tcp_error,
            wrench.force.x,
            wrench.force.y,
            wrench.force.z,
            wrench.torque.x,
            wrench.torque.y,
            wrench.torque.z,
            *joint_positions,
        ],
        dtype=np.float32,
    )

    if values.shape != (len(STATE_FIELD_NAMES),):
        raise ValueError(
            f"Unexpected state vector shape {values.shape}; "
            f"expected {(len(STATE_FIELD_NAMES),)}"
        )
    return values


def _fixed_len(values, length: int) -> list[float]:
    output = [0.0] * length
    for i, value in enumerate(values[:length]):
        output[i] = float(value)
    return output
