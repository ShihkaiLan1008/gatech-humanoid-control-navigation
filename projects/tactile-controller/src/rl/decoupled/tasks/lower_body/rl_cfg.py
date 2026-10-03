"""RL runner config for the lower_body task (per-task, no framework-level
recipe — user decision 2026-08-03)."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def lower_body_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            distribution_cfg={"class_name": "rsl_rl.modules.GaussianDistribution"},
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            entropy_coef=0.01,
        ),
        experiment_name="g1_decoupled_lower",
        clip_actions=100.0,
        max_iterations=50000,
        save_interval=500,
    )
