"""RL runner config for hand_pose (shared G1 MPC-guided PPO recipe; own
experiment name — the obs layout is +22 dims vs the locomotion tasks, so
checkpoints are not interchangeable with theirs)."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ...rl_cfg import g1_mpc_ppo_runner_cfg


def hand_pose_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = g1_mpc_ppo_runner_cfg()
    cfg.experiment_name = "g1_mpc_hand_pose"
    return cfg
