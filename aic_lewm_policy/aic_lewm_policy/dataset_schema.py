"""LEWM-compatible HDF5 schema helpers for AIC rollout data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .schemas import AicObservation, ACTION_FIELD_NAMES, STATE_FIELD_NAMES, TaskSpec


@dataclass(frozen=True)
class AicStepRecord:
    """One training row: observation, executed action, and task context."""

    observation: AicObservation
    action: np.ndarray
    task: TaskSpec


@dataclass(frozen=True)
class AicEpisodeRecord:
    """A single rollout episode in the flattened HDF5 layout used by LEWM."""

    steps: Sequence[AicStepRecord]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("AicEpisodeRecord requires at least one step")


def write_lewm_hdf5(
    episodes: Sequence[AicEpisodeRecord],
    output_path: str | Path,
) -> None:
    """Write AIC episodes as a stable_worldmodel HDF5Dataset file.

    The stable_worldmodel loader expects flat per-step datasets plus `ep_len`
    and `ep_offset` metadata. Image columns are stored as THWC uint8 arrays.
    """

    if not episodes:
        raise ValueError("At least one episode is required")

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - offline tool dependency
        raise ImportError("write_lewm_hdf5 requires h5py") from exc

    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = _flatten_episodes(episodes)
    with h5py.File(output_path, "w") as handle:
        for key, value in columns.items():
            if value.dtype.kind in ("U", "O"):
                dtype = h5py.string_dtype(encoding="utf-8")
                handle.create_dataset(key, data=value.astype(dtype), dtype=dtype)
            else:
                kwargs = _compression_kwargs(value)
                handle.create_dataset(key, data=value, **kwargs)

        handle.attrs["action_field_names"] = np.asarray(ACTION_FIELD_NAMES, dtype="S")
        handle.attrs["state_field_names"] = np.asarray(STATE_FIELD_NAMES, dtype="S")
        handle.attrs["primary_pixels"] = "pixels"


def _flatten_episodes(episodes: Sequence[AicEpisodeRecord]) -> dict[str, np.ndarray]:
    ep_len: list[int] = []
    ep_offset: list[int] = []
    ep_idx: list[int] = []
    step_idx: list[int] = []
    pixels = []
    left_pixels = []
    right_pixels = []
    proprio = []
    state = []
    action = []
    task_id = []
    plug_type = []
    port_type = []
    target_module_name = []

    offset = 0
    for episode_index, episode in enumerate(episodes):
        ep_len.append(len(episode.steps))
        ep_offset.append(offset)
        for local_step, step in enumerate(episode.steps):
            obs = step.observation
            action_vec = np.asarray(step.action, dtype=np.float32)
            if action_vec.shape != (len(ACTION_FIELD_NAMES),):
                raise ValueError(
                    f"Action shape {action_vec.shape} does not match "
                    f"{len(ACTION_FIELD_NAMES)} fields"
                )

            ep_idx.append(episode_index)
            step_idx.append(local_step)
            pixels.append(obs.images["center"])
            left_pixels.append(obs.images["left"])
            right_pixels.append(obs.images["right"])
            proprio.append(obs.state.astype(np.float32, copy=False))
            state.append(obs.state.astype(np.float32, copy=False))
            action.append(action_vec)
            task_id.append(step.task.task_id)
            plug_type.append(step.task.plug_type)
            port_type.append(step.task.port_type)
            target_module_name.append(step.task.target_module_name)
        offset += len(episode.steps)

    return {
        "ep_len": np.asarray(ep_len, dtype=np.int32),
        "ep_offset": np.asarray(ep_offset, dtype=np.int64),
        "ep_idx": np.asarray(ep_idx, dtype=np.int32),
        "episode_idx": np.asarray(ep_idx, dtype=np.int32),
        "step_idx": np.asarray(step_idx, dtype=np.int32),
        "pixels": _stack_images(pixels, "pixels"),
        "left_pixels": _stack_images(left_pixels, "left_pixels"),
        "right_pixels": _stack_images(right_pixels, "right_pixels"),
        "proprio": np.stack(proprio).astype(np.float32, copy=False),
        "state": np.stack(state).astype(np.float32, copy=False),
        "action": np.stack(action).astype(np.float32, copy=False),
        "task_id": np.asarray(task_id, dtype=object),
        "plug_type": np.asarray(plug_type, dtype=object),
        "port_type": np.asarray(port_type, dtype=object),
        "target_module_name": np.asarray(target_module_name, dtype=object),
    }


def _stack_images(images: Sequence[np.ndarray], name: str) -> np.ndarray:
    stacked = np.stack(images)
    if stacked.ndim != 4 or stacked.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (T, H, W, 3), got {stacked.shape}")
    return stacked.astype(np.uint8, copy=False)


def _compression_kwargs(value: np.ndarray) -> dict:
    if value.ndim >= 3:
        return {"compression": "gzip", "compression_opts": 4}
    return {}
