from mjlab.rl import RslRlPpoAlgorithmCfg, RslRlModelCfg
from rl.decoupled.runners.runner_cfgs import JointBodyCfg, RslRlJointOnPolicyRunnerCfg
import dataclasses

def joint_foot_mpc_ppo_runner_cfg() -> RslRlJointOnPolicyRunnerCfg:
    return RslRlJointOnPolicyRunnerCfg(
        body_names=("upper_body", "lower_body"),
        upper_body=JointBodyCfg(
            action_term="upper_body_joint_pos",
            actor_obs_set="upper_body_actor",
            critic_obs_set="upper_body_critic",
            reward_group="upper_body",
            actor=RslRlModelCfg(
                hidden_dims=(512, 256, 128),
                distribution_cfg={
                    "class_name": "GaussianDistribution",
                    "init_std": 0.6,
                    "std_type": "scalar",
                },
            ),
            critic=RslRlModelCfg(hidden_dims=(512, 256, 128)),
            algorithm=RslRlPpoAlgorithmCfg(entropy_coef=0.01),
        ),
        lower_body=JointBodyCfg(
            action_term="lower_body_joint_pos",
            actor_obs_set="lower_body_actor",
            critic_obs_set="lower_body_critic",
            reward_group="lower_body",
            actor=RslRlModelCfg(
                hidden_dims=(512, 256, 128),
                distribution_cfg={
                    "class_name": "GaussianDistribution",
                    "init_std": 1.0,
                    "std_type": "scalar",
                },
            ),
            critic=RslRlModelCfg(hidden_dims=(512, 256, 128)),
            algorithm=RslRlPpoAlgorithmCfg(entropy_coef=0.01),
        ),
        obs_groups={
            "upper_body_actor": ("upper_body_actor",),
            "upper_body_critic": ("upper_body_critic",),
            "lower_body_actor": ("lower_body_actor",),
            "lower_body_critic": ("lower_body_critic",),
        },
        experiment_name="g1_decoupled_joint_foot_mpc",
        clip_actions=100.0,  
        max_iterations=50000,
        save_interval=500,
    )

def joint_foot_vforce_mpc_runner_cfg() -> RslRlJointOnPolicyRunnerCfg:
    base = joint_foot_mpc_ppo_runner_cfg()
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
        experiment_name="g1_decoupled_joint_foot_vforce_mpc",
        upper_body=upper,
        lower_body=lower,
    )