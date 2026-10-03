"""RL runner config for push_box (shared G1 MPC-guided PPO recipe)."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ...rl_cfg import g1_mpc_ppo_runner_cfg


def push_box_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return g1_mpc_ppo_runner_cfg()
