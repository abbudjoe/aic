"""Planner boundary between AIC control and LEWM research code."""

from __future__ import annotations

from typing import Protocol

from .actions import CartesianVelocityAction
from .lewm_runtime import LewmRuntime, load_lewm_runtime
from .lewm_mpc import (
    ActionNormalizer,
    GoalLibrary,
    LearnedLewmMpcPlanner,
    PlannerSetupError,
)
from .replay_planner import DemonstrationReplayPlanner, ReplayDataset
from .schemas import AicObservation, PolicyRuntimeConfig, TaskSpec


class Planner(Protocol):
    def reset(self, task: TaskSpec) -> None: ...

    def select_action(
        self,
        task: TaskSpec,
        observation: AicObservation | None,
        elapsed_sec: float,
    ) -> CartesianVelocityAction: ...


class HoldStillPlanner:
    """Validity-first planner used until the learned LEWM cost head is trained."""

    def __init__(self, config: PolicyRuntimeConfig):
        self._config = config

    def reset(self, task: TaskSpec) -> None:
        _ = task

    def select_action(
        self,
        task: TaskSpec,
        observation: AicObservation | None,
        elapsed_sec: float,
    ) -> CartesianVelocityAction:
        _ = task, observation, elapsed_sec
        return CartesianVelocityAction.zero(frame_id=self._config.frame_id)


class LewmPlannerScaffold:
    """Loads LEWM and preserves the control contract while the AIC cost matures."""

    def __init__(
        self,
        runtime: LewmRuntime,
        fallback: HoldStillPlanner,
        logger,
    ):
        self._runtime = runtime
        self._fallback = fallback
        self._logger = logger
        self._warned = False

    def reset(self, task: TaskSpec) -> None:
        self._fallback.reset(task)
        self._warned = False

    def select_action(
        self,
        task: TaskSpec,
        observation: AicObservation | None,
        elapsed_sec: float,
    ) -> CartesianVelocityAction:
        if self._runtime.available and not self._warned:
            self._logger.warn(
                "LEWM checkpoint is loaded, but the AIC goal/cost head is not wired yet; "
                "using validity fallback commands."
            )
            self._warned = True
        return self._fallback.select_action(task, observation, elapsed_sec)


def make_planner(config: PolicyRuntimeConfig, logger) -> Planner:
    fallback = HoldStillPlanner(config=config)
    if config.planner_mode == "hold":
        logger.warn("AIC_LEWM_PLANNER_MODE=hold; publishing validity-only commands.")
        return fallback
    if config.planner_mode == "replay":
        if not config.replay_dataset_path:
            raise PlannerSetupError(
                "AIC_LEWM_REPLAY_DATASET or AIC_LEWM_GOAL_DATASET is required "
                "when AIC_LEWM_PLANNER_MODE=replay"
            )
        dataset = ReplayDataset.from_hdf5(config.replay_dataset_path)
        logger.info(
            "Demonstration replay planner enabled: "
            f"episodes={len(dataset.episodes)}, replay_hz={config.replay_hz}, "
            f"time_scale={config.replay_time_scale}, frame={config.frame_id}"
        )
        return DemonstrationReplayPlanner(dataset=dataset, config=config, logger=logger)
    if config.planner_mode != "lewm_mpc":
        raise PlannerSetupError(f"Unsupported AIC_LEWM_PLANNER_MODE={config.planner_mode!r}")

    runtime = load_lewm_runtime(logger)
    if runtime.available:
        if not config.goal_dataset_path:
            if config.require_checkpoint:
                raise PlannerSetupError(
                    "AIC_LEWM_GOAL_DATASET is required when "
                    "AIC_LEWM_REQUIRE_CHECKPOINT is enabled"
                )
            logger.warn(
                "LEWM checkpoint loaded but AIC_LEWM_GOAL_DATASET is not set; "
                "using validity fallback commands."
            )
            return LewmPlannerScaffold(runtime=runtime, fallback=fallback, logger=logger)
        goal_library = GoalLibrary.from_hdf5(config.goal_dataset_path)
        action_normalizer = ActionNormalizer.from_hdf5(
            config.goal_dataset_path,
            frameskip=config.frameskip,
        )
        logger.info(
            "LEWM learned MPC enabled: "
            f"history={config.history_size}, frameskip={config.frameskip}, "
            f"horizon={config.planning_horizon}, "
            f"candidates={config.num_action_candidates}"
        )
        return LearnedLewmMpcPlanner(
            model=runtime.model,
            config=config,
            goal_library=goal_library,
            action_normalizer=action_normalizer,
            logger=logger,
        )

    message = f"LEWM runtime unavailable: {runtime.unavailable_reason}"
    if config.require_checkpoint:
        raise PlannerSetupError(message)
    logger.info(message)
    return fallback
