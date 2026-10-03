"""Rewards — joint_vel_mpc's two groups + the hand-force family (UPPER only).

The force terms are the measured hand_force recipe verbatim (weights from the
whole-body task of record): CONTINUITY over accuracy — ``hand_contact_hold``
charges dropout duration, ``hand_force_rate`` dropout frequency; magnitude
accuracy is demoted to shaping (wide-σ exp + clamped L1 — never square an
unbounded contact force). ``hand_force_press_depth`` is the force→position
reparameterization added 2026-08-11 (see mdp/rewards.py).

Hand/force gradients reach ONLY the arm group (repo rule) — every term here
lands in ``upper_body``; the lower group keeps the base task's mpc_* bracing
channel, which now plans WITH the hand slots (the QP re-distributes GRF for
the commanded press).
"""

import copy

from rl.decoupled import mdp
from rl.decoupled.tasks.joint_vel_mpc.env_configs.reward_cfgs import (
    LOWER_REWARDS as _BASE_LOWER,
    UPPER_REWARDS as _BASE_UPPER,
)
from rl.managers.reward_group_manager import RewardGroupTermCfg

LOWER_REWARDS: dict[str, RewardGroupTermCfg] = copy.deepcopy(_BASE_LOWER)
UPPER_REWARDS: dict[str, RewardGroupTermCfg] = copy.deepcopy(_BASE_UPPER)

EE_COMMAND_NAMES = ("left_ee_cmd", "right_ee_cmd")

UPPER_REWARDS["contact_alignment"] = RewardGroupTermCfg(
    func=mdp.contact_alignment,
    weight=0.5,
    params={"command_names": EE_COMMAND_NAMES},
    group="upper_body",
)
UPPER_REWARDS["hand_contact_hold"] = RewardGroupTermCfg(
    func=mdp.hand_contact_hold,
    weight=1.5,
    params={"floor": 3.0, "grace_s": 1.0, "command_names": EE_COMMAND_NAMES},
    group="upper_body",
)
UPPER_REWARDS["hand_force_tracking"] = RewardGroupTermCfg(
    func=mdp.hand_force_tracking,
    weight=1.0,
    params={"std": 15.0, "command_names": EE_COMMAND_NAMES},
    group="upper_body",
)
UPPER_REWARDS["hand_force_l1"] = RewardGroupTermCfg(
    func=mdp.hand_force_l1,
    weight=-0.02,
    params={"clamp": 40.0, "command_names": EE_COMMAND_NAMES},
    group="upper_body",
)
UPPER_REWARDS["hand_force_rate"] = RewardGroupTermCfg(
    func=mdp.HandForceRate,
    weight=-0.05,
    params={"clamp": 20.0, "command_names": EE_COMMAND_NAMES},
    group="upper_body",
)
UPPER_REWARDS["hand_force_press_depth"] = RewardGroupTermCfg(
    func=mdp.hand_force_press_depth,
    weight=0.5,
    params={"k_virt": 500.0, "std": 0.03, "command_names": EE_COMMAND_NAMES},
    group="upper_body",
)
