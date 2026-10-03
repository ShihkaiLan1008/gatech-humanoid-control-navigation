"""RL runner configs — joint_vel_mpc's two PPO stacks under this task's
experiment name, and the vforce rails recipe (joint_vel_vforce_mpc's) for the
VForce variant."""

import dataclasses

from mjlab.rl import RslRlModelCfg

from rl.decoupled.runners.runner_cfgs import RslRlJointOnPolicyRunnerCfg
from rl.decoupled.tasks.joint_vel_mpc.rl_cfgs import joint_policy_ppo_runner_cfg


def joint_vel_hlip_mpc_ppo_runner_cfg() -> RslRlJointOnPolicyRunnerCfg:
    cfg = joint_policy_ppo_runner_cfg()
    cfg.experiment_name = "g1_decoupled_joint_vel_hlip_mpc"
    return cfg


def joint_vel_hlip_mpc_vforce_ppo_runner_cfg() -> RslRlJointOnPolicyRunnerCfg:
    base = joint_policy_ppo_runner_cfg()
    upper = dataclasses.replace(
        base.upper_body,
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.6,
                "std_type": "scalar",
                "std_range": (0.2, 0.55),
            },
        ),
    )
    lower = dataclasses.replace(
        base.lower_body,
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
                "std_range": (0.15, 0.35),
            },
        ),
        algorithm=dataclasses.replace(
            base.lower_body.algorithm,
            entropy_coef=0.0,
            schedule="fixed",
            learning_rate=2e-5,
        ),
    )
    return dataclasses.replace(
        base,
        experiment_name="g1_decoupled_joint_vel_hlip_vforce_mpc",
        upper_body=upper,
        lower_body=lower,
    )
