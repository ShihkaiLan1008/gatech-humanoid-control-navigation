"""Energy-aware G1 task registration for Isaac Lab."""

from __future__ import annotations

import sys

import gymnasium as gym


TASK_ID = "Isaac-Energy-Human-Flat-G1"
ROUGH_TASK_ID = "Isaac-Energy-Smart-Rough-G1"


def _register() -> None:
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "isaaclab_tasks.contrib.g1_energy_agent.config:EnergyHumanG1EnvCfg",
                "rsl_rl_cfg_entry_point": "isaaclab_tasks.contrib.g1_energy_agent.config:EnergyHumanG1RunnerCfg",
            },
        )
    if ROUGH_TASK_ID not in gym.registry:
        gym.register(
            id=ROUGH_TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "isaaclab_tasks.contrib.g1_energy_agent.config:EnergySmartRoughG1EnvCfg",
                "rsl_rl_cfg_entry_point": "isaaclab_tasks.contrib.g1_energy_agent.config:EnergySmartRoughG1RunnerCfg",
            },
        )


def register_task() -> list[str]:
    """Register the custom task and return CLI arguments accepted by Hydra."""
    _register()
    return sys.argv[1:]


# Gym's ``module:task-id`` syntax imports this package before resolving the ID.
_register()
