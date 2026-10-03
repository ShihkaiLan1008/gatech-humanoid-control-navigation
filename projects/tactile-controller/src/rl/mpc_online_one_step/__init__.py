"""Stage A1: 1-step QP + residual policy on the G1 (docs/framework_design.md §9).

Registers ``Mjlab-OneStep-Loco-G1`` — velocity locomotion where a batched
1-step SRBD force QP (pi^nMPC-style ADMM, ``g1_mpc.admm_qp``) runs inside the
action term and the policy outputs residuals only (Δa base accel 6 + Δq leg
joint positions 12). Gait clock + Raibert + weight shift are the Stage-0
recipe; arms/waist are held at the default pose (A1).
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import g1_onestep_flat_env_cfg


def g1_onestep_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            distribution_cfg={"class_name": "rsl_rl.modules.GaussianDistribution"},
        ),
        critic=RslRlModelCfg(hidden_dims=(512, 256, 128)),
        algorithm=RslRlPpoAlgorithmCfg(entropy_coef=0.01),
        experiment_name="g1_onestep",
        max_iterations=10_000,
        save_interval=500,
    )


register_mjlab_task(
    task_id="Mjlab-OneStep-Loco-G1",
    env_cfg=g1_onestep_flat_env_cfg(),
    play_env_cfg=g1_onestep_flat_env_cfg(play=True),
    rl_cfg=g1_onestep_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
