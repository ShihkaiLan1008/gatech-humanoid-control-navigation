"""RL configurations for the G1 MPC-guided tasks (port of themis rl_cfg)."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def g1_mpc_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create the RL runner configuration for the G1 MPC-guided tasks."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(1024, 512, 256),
            distribution_cfg={"class_name": "rsl_rl.modules.GaussianDistribution"},
        ),
        critic=RslRlModelCfg(
            hidden_dims=(1024, 512, 256),
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            entropy_coef=0.01,
        ),
        experiment_name="g1_mpc_velocity",
        max_iterations=50000,
        save_interval=500,
    )
