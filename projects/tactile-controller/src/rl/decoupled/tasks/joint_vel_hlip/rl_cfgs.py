"""RL runner config — joint_vel's two PPO stacks under this task's experiment name."""

from rl.decoupled.runners.runner_cfgs import RslRlJointOnPolicyRunnerCfg
from rl.decoupled.tasks.joint_vel.rl_cfgs import joint_policy_ppo_runner_cfg


def joint_vel_hlip_ppo_runner_cfg() -> RslRlJointOnPolicyRunnerCfg:
    cfg = joint_policy_ppo_runner_cfg()
    cfg.experiment_name = "g1_decoupled_joint_vel_hlip"
    return cfg
