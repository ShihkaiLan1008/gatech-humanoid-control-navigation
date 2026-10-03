from rl.decoupled.runners.runner_cfgs import RslRlJointOnPolicyRunnerCfg
from rl.decoupled.tasks.joint_foot_mpc.rl_cfgs import joint_foot_mpc_ppo_runner_cfg


def joint_foot_hlip_ppo_runner_cfg() -> RslRlJointOnPolicyRunnerCfg:
    """The foot_mpc PPO stacks verbatim under this task's experiment name."""
    cfg = joint_foot_mpc_ppo_runner_cfg()
    cfg.experiment_name = "g1_decoupled_joint_foot_hlip"
    return cfg
