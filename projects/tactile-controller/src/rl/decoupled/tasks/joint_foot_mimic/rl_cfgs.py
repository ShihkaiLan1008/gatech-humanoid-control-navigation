"""RL runner config for the joint_policy task (per-task, no framework-level
recipe — user decision 2026-08-03).

Two independent PPO stacks. `JointBodyCfg` wires each body to its action term,
its observation SETS and its reward group; `obs_groups` maps each set name onto
the environment observation groups declared in `env_configs/observation_cfgs.py`
(identity mapping here — one group per set).

The concatenated action follows ACTION-TERM declaration order in
`env_configs/action_cfgs.py`, not the order of `body_names`; the runner reads
the layout from the ActionManager and errors if the two disagree.
"""

from mjlab.rl import RslRlPpoAlgorithmCfg, RslRlModelCfg
import dataclasses
from rl.decoupled.runners.runner_cfgs import JointBodyCfg, RslRlJointOnPolicyRunnerCfg


def joint_policy_ppo_runner_cfg() -> RslRlJointOnPolicyRunnerCfg:
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
        experiment_name="g1_decoupled_joint_foot_mimic",
        clip_actions=100.0,  
        max_iterations=50000,
        save_interval=500,
    )


def joint_foot_mimic_vforce_ppo_runner_cfg() -> RslRlJointOnPolicyRunnerCfg:
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
        experiment_name="g1_decoupled_joint_foot_mimic_vforce",
        upper_body=upper,
        lower_body=lower,
    )