"""LEWM goal-image MPC runtime for the AIC policy process."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from .actions import CartesianVelocityAction
from .schemas import AicObservation, PolicyRuntimeConfig, TaskSpec


IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class PlannerSetupError(RuntimeError):
    """Raised when the learned planner contract cannot be satisfied."""


@dataclass(frozen=True)
class GoalExample:
    pixels: np.ndarray
    task_id: str
    plug_type: str
    port_type: str
    target_module_name: str


class GoalLibrary:
    """Terminal demonstration images keyed by the public task description."""

    def __init__(self, examples: Sequence[GoalExample]):
        if not examples:
            raise PlannerSetupError("goal library requires at least one example")
        self._examples = tuple(examples)

    @classmethod
    def from_hdf5(cls, path: str | Path) -> "GoalLibrary":
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - dependency checked in image
            raise PlannerSetupError("AIC_LEWM_GOAL_DATASET requires h5py") from exc

        path = Path(path).expanduser()
        if not path.exists():
            raise PlannerSetupError(f"goal dataset does not exist: {path}")

        examples: list[GoalExample] = []
        with h5py.File(path, "r") as handle:
            required = (
                "ep_len",
                "ep_offset",
                "pixels",
                "task_id",
                "plug_type",
                "port_type",
                "target_module_name",
            )
            missing = [key for key in required if key not in handle]
            if missing:
                raise PlannerSetupError(
                    f"goal dataset missing required keys: {', '.join(missing)}"
                )

            ep_len = np.asarray(handle["ep_len"], dtype=np.int64)
            ep_offset = np.asarray(handle["ep_offset"], dtype=np.int64)
            for length, offset in zip(ep_len, ep_offset, strict=True):
                if int(length) <= 0:
                    continue
                index = int(offset + length - 1)
                examples.append(
                    GoalExample(
                        pixels=np.asarray(handle["pixels"][index], dtype=np.uint8),
                        task_id=_decode(handle["task_id"][index]),
                        plug_type=_decode(handle["plug_type"][index]),
                        port_type=_decode(handle["port_type"][index]),
                        target_module_name=_decode(handle["target_module_name"][index]),
                    )
                )

        return cls(examples)

    def select(self, task: TaskSpec) -> GoalExample:
        for example in self._examples:
            if (
                example.plug_type == task.plug_type
                and example.port_type == task.port_type
                and example.target_module_name == task.target_module_name
            ):
                return example
        for example in self._examples:
            if example.plug_type == task.plug_type and example.port_type == task.port_type:
                return example
        return self._examples[0]


class ActionNormalizer:
    """Normalize the two-frame action tokens used by the trained LEWM model."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, frameskip: int):
        if frameskip <= 0:
            raise PlannerSetupError("frameskip must be positive")
        token_dim = frameskip * 6
        if mean.shape != (token_dim,) or std.shape != (token_dim,):
            raise PlannerSetupError(
                f"action normalizer expected token dim {token_dim}, "
                f"got mean={mean.shape}, std={std.shape}"
            )
        self.mean = mean.astype(np.float32, copy=False)
        self.std = np.maximum(std.astype(np.float32, copy=False), 1e-6)
        self.frameskip = frameskip

    @classmethod
    def from_hdf5(cls, path: str | Path, frameskip: int) -> "ActionNormalizer":
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - dependency checked in image
            raise PlannerSetupError("action normalizer requires h5py") from exc

        path = Path(path).expanduser()
        if not path.exists():
            raise PlannerSetupError(f"normalizer dataset does not exist: {path}")

        with h5py.File(path, "r") as handle:
            if "action" not in handle or "ep_len" not in handle or "ep_offset" not in handle:
                raise PlannerSetupError(
                    "normalizer dataset requires action, ep_len, and ep_offset"
                )
            actions = np.asarray(handle["action"], dtype=np.float32)
            ep_len = np.asarray(handle["ep_len"], dtype=np.int64)
            ep_offset = np.asarray(handle["ep_offset"], dtype=np.int64)

        tokens: list[np.ndarray] = []
        for length, offset in zip(ep_len, ep_offset, strict=True):
            if int(length) <= 0:
                continue
            start = int(offset)
            stop = int(offset + length)
            episode_actions = actions[start:stop]
            for local_index in range(len(episode_actions)):
                window = []
                for frame_index in range(frameskip):
                    source_index = min(local_index + frame_index, len(episode_actions) - 1)
                    window.append(episode_actions[source_index])
                tokens.append(np.concatenate(window, axis=0))

        if not tokens:
            raise PlannerSetupError("normalizer dataset contains no action tokens")

        stacked = np.stack(tokens).astype(np.float32, copy=False)
        return cls(mean=stacked.mean(axis=0), std=stacked.std(axis=0), frameskip=frameskip)

    def normalize_token(self, raw_token: np.ndarray) -> np.ndarray:
        raw_token = np.asarray(raw_token, dtype=np.float32)
        return (raw_token - self.mean) / self.std

    def denormalize_token(self, norm_token: np.ndarray) -> np.ndarray:
        norm_token = np.asarray(norm_token, dtype=np.float32)
        return norm_token * self.std + self.mean

    def repeated_token(self, raw_action: np.ndarray) -> np.ndarray:
        raw_action = np.asarray(raw_action, dtype=np.float32)
        if raw_action.shape != (6,):
            raise ValueError(f"raw action must have shape (6,), got {raw_action.shape}")
        return np.tile(raw_action, self.frameskip)


class LearnedLewmMpcPlanner:
    """Random-shooting MPC using LEWM's goal-embedding cost."""

    def __init__(
        self,
        *,
        model,
        config: PolicyRuntimeConfig,
        goal_library: GoalLibrary,
        action_normalizer: ActionNormalizer,
        logger,
    ):
        self._model = model
        self._config = config
        self._goal_library = goal_library
        self._action_normalizer = action_normalizer
        self._logger = logger
        self._rng = np.random.default_rng(3072)
        self._image_history: list[object] = []
        self._action_history: list[np.ndarray] = []
        self._selected_goal = None
        self._last_raw_action = np.zeros(6, dtype=np.float32)

    def reset(self, task: TaskSpec) -> None:
        self._image_history = []
        self._action_history = []
        self._last_raw_action = np.zeros(6, dtype=np.float32)
        self._selected_goal = self._goal_library.select(task)
        self._logger.info(
            "LEWM goal selected: "
            f"task_id={self._selected_goal.task_id} "
            f"{self._selected_goal.plug_type}->{self._selected_goal.port_type} "
            f"module={self._selected_goal.target_module_name}"
        )

    def select_action(
        self,
        task: TaskSpec,
        observation: AicObservation | None,
        elapsed_sec: float,
    ) -> CartesianVelocityAction:
        _ = task, elapsed_sec
        if observation is None:
            return CartesianVelocityAction.zero(frame_id=self._config.frame_id)

        self._append_observation(observation)
        if len(self._image_history) < self._config.history_size:
            return CartesianVelocityAction.zero(frame_id=self._config.frame_id)

        try:
            started = time.monotonic()
            raw_action = self._plan_raw_action()
            self._logger.info(
                "LEWM MPC planned action in "
                f"{time.monotonic() - started:.3f}s"
            )
        except Exception as exc:  # pragma: no cover - exercised by integration runs
            self._logger.error(f"LEWM MPC inference failed; holding command: {exc}")
            raw_action = np.zeros(6, dtype=np.float32)

        self._remember_action(raw_action)
        return CartesianVelocityAction(
            linear=tuple(float(x) for x in raw_action[:3]),
            angular=tuple(float(x) for x in raw_action[3:]),
            frame_id=self._config.frame_id,
        )

    def _append_observation(self, observation: AicObservation) -> None:
        self._image_history.append(
            image_to_model_tensor(observation.center_image(), self._config.image_size)
        )
        self._image_history = self._image_history[-self._config.history_size :]
        if len(self._action_history) < len(self._image_history):
            self._remember_action(np.zeros(6, dtype=np.float32))

    def _remember_action(self, raw_action: np.ndarray) -> None:
        raw_action = np.asarray(raw_action, dtype=np.float32)
        token = self._action_normalizer.repeated_token(raw_action)
        self._action_history.append(self._action_normalizer.normalize_token(token))
        self._action_history = self._action_history[-self._config.history_size :]
        self._last_raw_action = raw_action

    def _plan_raw_action(self) -> np.ndarray:
        import torch

        if self._selected_goal is None:
            raise PlannerSetupError("planner used before reset selected a goal")

        device = next(self._model.parameters()).device
        history = torch.stack(self._image_history[-self._config.history_size :], dim=0)
        goal = image_to_model_tensor(
            self._selected_goal.pixels,
            self._config.image_size,
        )
        goal_history = goal.unsqueeze(0).repeat(self._config.history_size, 1, 1, 1)

        raw_future = self._sample_future_actions()
        norm_future = np.empty_like(raw_future, dtype=np.float32)
        for sample_index in range(raw_future.shape[0]):
            for step_index in range(raw_future.shape[1]):
                norm_future[sample_index, step_index] = self._action_normalizer.normalize_token(
                    raw_future[sample_index, step_index]
                )

        history_actions = np.stack(self._action_history[-self._config.history_size :])
        history_actions = np.array(
            np.broadcast_to(
                history_actions,
                (
                    raw_future.shape[0],
                    self._config.history_size,
                    history_actions.shape[-1],
                ),
            ),
            dtype=np.float32,
            copy=True,
        )
        action_candidates = np.concatenate([history_actions, norm_future], axis=1)

        pixels = history.unsqueeze(0).unsqueeze(0).repeat(
            1, raw_future.shape[0], 1, 1, 1, 1
        )
        goals = goal_history.unsqueeze(0).unsqueeze(0).repeat(
            1, raw_future.shape[0], 1, 1, 1, 1
        )
        info = {
            "pixels": pixels.to(device),
            "goal": goals.to(device),
            "action": torch.as_tensor(history_actions, dtype=torch.float32, device=device).unsqueeze(0),
        }
        candidates = torch.as_tensor(
            action_candidates[None, ...],
            dtype=torch.float32,
            device=device,
        )

        with torch.inference_mode():
            costs = self._model.get_cost(info, candidates)
        best_index = int(torch.argmin(costs[0]).detach().cpu().item())
        return raw_future[best_index, 0, :6].astype(np.float32, copy=False)

    def _sample_future_actions(self) -> np.ndarray:
        count = max(self._config.num_action_candidates, 4)
        horizon = max(self._config.planning_horizon, 1)
        token_dim = self._action_normalizer.frameskip * 6
        raw = np.zeros((count, horizon, token_dim), dtype=np.float32)

        base_actions = [
            np.zeros(6, dtype=np.float32),
            self._last_raw_action.astype(np.float32, copy=True),
            np.array([0.0, 0.0, -0.004, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, -0.010, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, -0.016, 0.0, 0.0, 0.0], dtype=np.float32),
        ]

        for sample_index in range(count):
            if sample_index < len(base_actions):
                action = base_actions[sample_index]
                sequence = np.tile(action, (horizon, 1))
            else:
                sequence = self._rng.normal(
                    loc=0.0,
                    scale=(0.008, 0.008, 0.009, 0.045, 0.045, 0.045),
                    size=(horizon, 6),
                ).astype(np.float32)
                sequence[:, 2] += self._rng.choice([0.0, -0.004, -0.008])

            sequence[:, :3] = np.clip(
                sequence[:, :3],
                -self._config.linear_velocity_limit,
                self._config.linear_velocity_limit,
            )
            sequence[:, 3:] = np.clip(
                sequence[:, 3:],
                -self._config.angular_velocity_limit,
                self._config.angular_velocity_limit,
            )
            for step_index, action in enumerate(sequence):
                raw[sample_index, step_index] = self._action_normalizer.repeated_token(action)
        return raw


def image_to_model_tensor(image: np.ndarray, image_size: int):
    """Convert HWC uint8 pixels into the normalized CHW tensor used for training."""

    import torch
    import torch.nn.functional as F

    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image must have shape HWCx3, got {image.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
    tensor = tensor.div(255.0).unsqueeze(0)
    if image_size > 0 and tuple(tensor.shape[-2:]) != (image_size, image_size):
        tensor = F.interpolate(
            tensor,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(1, 3, 1, 1)
    return ((tensor - mean) / std).squeeze(0)


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)
