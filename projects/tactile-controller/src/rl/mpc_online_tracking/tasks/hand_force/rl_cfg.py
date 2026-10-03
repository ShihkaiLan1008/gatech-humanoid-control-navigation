

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ...rl_cfg import g1_mpc_ppo_runner_cfg


def hand_force_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = g1_mpc_ppo_runner_cfg()
    cfg.experiment_name = "g1_mpc_hand_force"
    return cfg
