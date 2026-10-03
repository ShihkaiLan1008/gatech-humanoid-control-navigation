"""RL runner config — the decoupled force line's two-PPO recipe under an eval
experiment name. The obs layout is identical by construction, so ANY of the
force-line checkpoints (joint_vel_mpc / vforce / force) load strict=True via
``--checkpoint``."""

import dataclasses

from rl.decoupled.runners.runner_cfgs import RslRlJointOnPolicyRunnerCfg
from rl.decoupled.tasks.joint_vel_force_mpc.rl_cfgs import (
    joint_vel_force_mpc_ppo_runner_cfg,
)


def force_maze_ppo_runner_cfg() -> RslRlJointOnPolicyRunnerCfg:
    return dataclasses.replace(
        joint_vel_force_mpc_ppo_runner_cfg(),
        experiment_name="g1_eval_force_maze",
    )
