"""Rewards: foot_mpc's lower set minus the ``mpc_*`` family, plus the H-LIP
tracking (``USE_CLF`` picks the CLF pair or the plain exp+L2 kernels; the
other set is logged at weight 0) and the orbital-energy floor. Upper set
verbatim."""
import copy

from rl.decoupled import mdp
from rl.decoupled.tasks.joint_foot_mpc.env_configs.reward_cfgs import (
    LOWER_REWARDS as _BASE_LOWER,
)
from rl.decoupled.tasks.joint_foot_mpc.env_configs.reward_cfgs import (
    UPPER_REWARDS as _BASE_UPPER,
)
from rl.managers.reward_group_manager import RewardGroupTermCfg

USE_CLF = True
"""Which CoM-proxy tracking is PRICED: True = the CLF pair (plain kernels
logged at weight 0), False = the plain exp+L2 kernels (CLF pair logged at 0).
Both are always computed, so either run shows the other's diagnostics."""

_W = (
    dict(clf=10.0, clf_dec=-2.0, pos=0.0, pos_l2=0.0, vel=0.0, vel_l2=0.0)
    if USE_CLF else
    dict(clf=0.0, clf_dec=0.0, pos=2.0, pos_l2=-2.0, vel=2.0, vel_l2=-1.0)
)

LOWER_REWARDS: dict[str, RewardGroupTermCfg] = {
    k: copy.deepcopy(v) for k, v in _BASE_LOWER.items() if not k.startswith("mpc_")
}

LOWER_REWARDS.update({
    # ============================================================
    # H-LIP CoM-proxy tracking (CLF-RL form, weights from their LIP task)
    # ============================================================
    "clf_reward": RewardGroupTermCfg(
        func=mdp.clf_reward,
        weight=_W["clf"],
        params={"command_name": "hlip", "eta_max": 0.1},
        group="lower_body",
    ),
    "clf_decreasing": RewardGroupTermCfg(
        func=mdp.clf_decreasing_condition,
        weight=_W["clf_dec"],
        params={"command_name": "hlip", "eta_max": 0.1, "eta_dot_max": 0.3, "lam": 1.0},
        group="lower_body",
    ),
    "hlip_com_pos": RewardGroupTermCfg(
        func=mdp.hlip_com_pos,
        weight=_W["pos"],
        params={"command_name": "hlip", "std": 0.05},
        group="lower_body",
    ),
    "hlip_com_pos_l2": RewardGroupTermCfg(
        func=mdp.hlip_com_pos_l2,
        weight=_W["pos_l2"],
        params={"command_name": "hlip", "clamp": 0.09},
        group="lower_body",
    ),
    "hlip_com_vel": RewardGroupTermCfg(
        func=mdp.hlip_com_vel,
        weight=_W["vel"],
        params={"command_name": "hlip", "std": 0.08},
        group="lower_body",
    ),
    "hlip_com_vel_l2": RewardGroupTermCfg(
        func=mdp.hlip_com_vel_l2,
        weight=_W["vel_l2"],
        params={"command_name": "hlip", "clamp": 0.25},
        group="lower_body",
    ),
    # ============================================================
    # Orbital-energy floor: never below the reference's E in the SSP
    # ============================================================
    "orbital_energy_ssp": RewardGroupTermCfg(
        func=mdp.orbital_energy_ssp,
        weight=2.0,
        params={"command_name": "hlip", "std": 0.02},
        group="lower_body",
    ),
    "orbital_energy_ssp_l1": RewardGroupTermCfg(
        func=mdp.orbital_energy_ssp_l1,
        weight=-15.0,
        params={"command_name": "hlip", "clamp": 0.1},
        group="lower_body",
    ),
    "liftoff_energy_l1": RewardGroupTermCfg(
        func=mdp.liftoff_energy_l1,
        weight=-15.0,
        params={"command_name": "hlip", "clamp": 0.1},
        group="lower_body",
    ),
})

UPPER_REWARDS: dict[str, RewardGroupTermCfg] = copy.deepcopy(_BASE_UPPER)
