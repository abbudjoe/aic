"""Dataset-backed replay planner for the first legal AIC control baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .actions import CartesianVelocityAction
from .schemas import (
    AicObservation,
    FinalServoProfile,
    PolicyRuntimeConfig,
    STATE_FIELD_NAMES,
    TaskSpec,
)


_WRIST_FORCE_START = STATE_FIELD_NAMES.index("wrist_wrench.force.x")
_WRIST_FORCE_STOP = _WRIST_FORCE_START + 3


@dataclass(frozen=True)
class ReplayEpisode:
    episode_index: int
    task_id: str
    plug_type: str
    port_type: str
    target_module_name: str
    actions: np.ndarray

    def match_score(self, task: TaskSpec) -> tuple[int, int, int, int]:
        return (
            int(_same(self.target_module_name, task.target_module_name)),
            int(_same(self.plug_type, task.plug_type)),
            int(_same(self.port_type, task.port_type)),
            int(_same(self.task_id, task.task_id)),
        )


class ReplayDataset:
    """A small, typed view over the HDF5 demonstrations used for replay."""

    def __init__(self, episodes: list[ReplayEpisode]):
        if not episodes:
            raise ValueError("ReplayDataset requires at least one episode")
        self.episodes = tuple(episodes)

    @classmethod
    def from_hdf5(cls, path: str | Path) -> "ReplayDataset":
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise ImportError("ReplayDataset.from_hdf5 requires h5py") from exc

        path = Path(path).expanduser()
        with h5py.File(path, "r") as handle:
            ep_len = handle["ep_len"][()]
            ep_offset = handle["ep_offset"][()]
            actions = handle["action"][()]
            episodes: list[ReplayEpisode] = []
            for episode_index, (offset, length) in enumerate(zip(ep_offset, ep_len)):
                start = int(offset)
                stop = start + int(length)
                episodes.append(
                    ReplayEpisode(
                        episode_index=episode_index,
                        task_id=_decode(handle["task_id"][start]),
                        plug_type=_decode(handle["plug_type"][start]),
                        port_type=_decode(handle["port_type"][start]),
                        target_module_name=_decode(handle["target_module_name"][start]),
                        actions=np.asarray(actions[start:stop], dtype=np.float32),
                    )
                )
        return cls(episodes)

    def select(self, task: TaskSpec) -> ReplayEpisode:
        ranked = sorted(
            self.episodes,
            key=lambda episode: (episode.match_score(task), -episode.episode_index),
            reverse=True,
        )
        best = ranked[0]
        if best.match_score(task)[1:3] == (0, 0):
            raise ValueError(
                "No replay episode matches task plug/port "
                f"{task.plug_type}->{task.port_type}"
            )
        return best


class DemonstrationReplayPlanner:
    """Replay recorded Cartesian TCP velocities using only public task metadata."""

    def __init__(self, *, dataset: ReplayDataset, config: PolicyRuntimeConfig, logger):
        if config.replay_hz <= 0.0:
            raise ValueError("AIC_LEWM_REPLAY_HZ must be positive")
        if config.replay_time_scale <= 0.0:
            raise ValueError("AIC_LEWM_REPLAY_TIME_SCALE must be positive")
        if config.replay_sc_stop_sec is not None and config.replay_sc_stop_sec <= 0.0:
            raise ValueError("AIC_LEWM_SC_REPLAY_STOP_SEC must be positive when set")
        if config.final_servo.enabled:
            if config.final_servo.duration_sec <= 0.0:
                raise ValueError("AIC_LEWM_FINAL_SERVO_DURATION_SEC must be positive when enabled")
            if config.final_servo.force_guard_n <= 0.0:
                raise ValueError(
                    "AIC_LEWM_FINAL_SERVO_FORCE_GUARD_N must be positive when enabled"
                )
            if config.final_servo.force_guard_mode not in {"absolute", "delta"}:
                raise ValueError(
                    "AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE must be absolute or delta"
                )
        self._dataset = dataset
        self._config = config
        self._logger = logger
        self._selected_episode: ReplayEpisode | None = None
        self._stop_logged = False
        self._servo_logged = False
        self._servo_guard_logged = False
        self._servo_force_baseline: np.ndarray | None = None

    @property
    def selected_episode(self) -> ReplayEpisode | None:
        return self._selected_episode

    def reset(self, task: TaskSpec) -> None:
        self._selected_episode = self._dataset.select(task)
        self._stop_logged = False
        self._servo_logged = False
        self._servo_guard_logged = False
        self._servo_force_baseline = None
        score = self._selected_episode.match_score(task)
        self._logger.info(
            "Replay episode selected: "
            f"episode={self._selected_episode.episode_index} "
            f"{self._selected_episode.plug_type}->{self._selected_episode.port_type} "
            f"module={self._selected_episode.target_module_name} "
            f"match={score}"
        )
        if score[0] == 0:
            self._logger.warn(
                "Replay is using a plug/port match with a different target module; "
                "expect poor transfer until visual servo correction is added."
            )

    def select_action(
        self,
        task: TaskSpec,
        observation: AicObservation | None,
        elapsed_sec: float,
    ) -> CartesianVelocityAction:
        del task
        if self._selected_episode is None:
            return CartesianVelocityAction.zero(frame_id=self._config.frame_id)

        stop_sec = replay_stop_sec_for_episode(self._selected_episode, self._config)
        if stop_sec is not None and elapsed_sec >= stop_sec:
            if not self._stop_logged:
                self._logger.info(
                    "Replay safety stop engaged: "
                    f"episode={self._selected_episode.episode_index} "
                    f"port_type={self._selected_episode.port_type} "
                    f"elapsed={elapsed_sec:.2f}s stop_sec={stop_sec:.2f}s"
                )
                self._stop_logged = True
            return CartesianVelocityAction.zero(frame_id=self._config.frame_id)

        action_index = int(elapsed_sec * self._config.replay_hz / self._config.replay_time_scale)
        if action_index >= len(self._selected_episode.actions):
            return self._select_final_servo_action(
                observation=observation,
                elapsed_sec=elapsed_sec,
            )

        raw_action = (
            self._selected_episode.actions[action_index, :6].astype(np.float32, copy=False)
            * self._config.replay_action_gain
        )
        return CartesianVelocityAction(
            linear=tuple(float(x) for x in raw_action[:3]),
            angular=tuple(float(x) for x in raw_action[3:]),
            frame_id=self._config.frame_id,
        )

    def _select_final_servo_action(
        self,
        *,
        observation: AicObservation | None,
        elapsed_sec: float,
    ) -> CartesianVelocityAction:
        if self._selected_episode is None or not self._config.final_servo.enabled:
            return CartesianVelocityAction.zero(frame_id=self._config.frame_id)

        replay_duration_sec = replay_duration_sec_for_episode(
            self._selected_episode,
            self._config,
        )
        servo_elapsed_sec = elapsed_sec - replay_duration_sec
        if not 0.0 <= servo_elapsed_sec < self._config.final_servo.duration_sec:
            return CartesianVelocityAction.zero(frame_id=self._config.frame_id)

        profile = final_servo_profile_for_episode(self._selected_episode, self._config)
        if profile.is_zero():
            return CartesianVelocityAction.zero(frame_id=self._config.frame_id)

        if observation is None:
            if not self._servo_guard_logged:
                self._logger.warn("Final insertion servo held: observation unavailable for force guard.")
                self._servo_guard_logged = True
            return CartesianVelocityAction.zero(frame_id=self._config.frame_id)

        force = _wrist_force_vector(observation)
        force_guard_value_n = self._force_guard_value_n(force)
        if force_guard_value_n >= self._config.final_servo.force_guard_n:
            if not self._servo_guard_logged:
                self._logger.warn(
                    "Final insertion servo force guard engaged: "
                    f"mode={self._config.final_servo.force_guard_mode} "
                    f"value={force_guard_value_n:.2f}N "
                    f"limit={self._config.final_servo.force_guard_n:.2f}N"
                )
                self._servo_guard_logged = True
            return CartesianVelocityAction.zero(frame_id=self._config.frame_id)

        if not self._servo_logged:
            self._logger.info(
                "Final insertion servo engaged: "
                f"episode={self._selected_episode.episode_index} "
                f"port_type={self._selected_episode.port_type} "
                f"elapsed={elapsed_sec:.2f}s "
                f"duration={self._config.final_servo.duration_sec:.2f}s "
                f"linear={profile.linear} angular={profile.angular}"
            )
            self._servo_logged = True
        return CartesianVelocityAction(
            linear=profile.linear,
            angular=profile.angular,
            frame_id=self._config.frame_id,
        )

    def _force_guard_value_n(self, force: np.ndarray) -> float:
        if self._config.final_servo.force_guard_mode == "delta":
            if self._servo_force_baseline is None:
                self._servo_force_baseline = np.asarray(force, dtype=np.float32).copy()
                self._logger.info(
                    "Final insertion servo force baseline captured: "
                    f"force=({self._servo_force_baseline[0]:.2f}, "
                    f"{self._servo_force_baseline[1]:.2f}, "
                    f"{self._servo_force_baseline[2]:.2f})N"
                )
            force = force - self._servo_force_baseline
        return float(np.linalg.norm(force))


def replay_stop_sec_for_episode(
    episode: ReplayEpisode,
    config: PolicyRuntimeConfig,
) -> float | None:
    if _same(episode.port_type, "sc"):
        return config.replay_sc_stop_sec
    return None


def replay_duration_sec_for_episode(
    episode: ReplayEpisode,
    config: PolicyRuntimeConfig,
) -> float:
    return float(len(episode.actions) * config.replay_time_scale / config.replay_hz)


def final_servo_profile_for_episode(
    episode: ReplayEpisode,
    config: PolicyRuntimeConfig,
) -> FinalServoProfile:
    if not config.final_servo.enabled:
        return FinalServoProfile.zero()
    if _same(episode.port_type, "sfp") or _same(episode.plug_type, "sfp"):
        return config.final_servo.sfp
    if _same(episode.port_type, "sc") or _same(episode.plug_type, "sc"):
        return config.final_servo.sc
    return FinalServoProfile.zero()


def final_servo_window_sec_for_episode(
    episode: ReplayEpisode,
    config: PolicyRuntimeConfig,
) -> tuple[float, float] | None:
    profile = final_servo_profile_for_episode(episode, config)
    if profile.is_zero() or not config.final_servo.enabled:
        return None
    start_sec = replay_duration_sec_for_episode(episode, config)
    return start_sec, start_sec + config.final_servo.duration_sec


def _wrist_force_vector(observation: AicObservation) -> np.ndarray:
    return observation.state[_WRIST_FORCE_START:_WRIST_FORCE_STOP].astype(np.float32, copy=False)


def _same(left: str, right: str) -> bool:
    return left.strip().lower() == right.strip().lower()


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
