"""RL runner config — joint_vel_mpc's two-PPO recipe under a new experiment
name. The obs layout is identical by construction (zero-reserve swap), so the
trained joint_vel_mpc checkpoint resumes strict=True."""

import dataclasses

from rl.decoupled.runners.runner_cfgs import RslRlJointOnPolicyRunnerCfg
from rl.decoupled.tasks.joint_vel_mpc.rl_cfgs import joint_policy_ppo_runner_cfg


def joint_vel_force_mpc_ppo_runner_cfg() -> RslRlJointOnPolicyRunnerCfg:
    return dataclasses.replace(
        joint_policy_ppo_runner_cfg(),
        experiment_name="g1_decoupled_joint_vel_force_mpc",
    )
