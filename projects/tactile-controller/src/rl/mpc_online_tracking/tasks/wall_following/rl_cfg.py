"""RL runner config for wall_following (shared G1 MPC-guided PPO recipe; own
experiment name). Obs layout is hand_force-identical (492/559), so hand_force
checkpoints PLAY here directly. The task is play-first — before ever training
in it, revisit the Se2-attractor-as-training-distribution rule AND the
missing-velocity-payment problem (2026-07-29 run verdict)."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ...rl_cfg import g1_mpc_ppo_runner_cfg


def wall_following_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = g1_mpc_ppo_runner_cfg()
    cfg.experiment_name = "g1_mpc_wall_following"
    return cfg
