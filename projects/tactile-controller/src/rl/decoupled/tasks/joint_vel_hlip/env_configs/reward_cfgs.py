"""Rewards — lower body = robot_rl's LIP-CLF set (CLF pair, stance holonomic
pair, regularizers; no velocity-tracking term), upper body = joint_vel's."""

import copy

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp

from rl.decoupled import mdp
from rl.decoupled.mdp import LOWER_JOINT_NAMES
from rl.decoupled.tasks.joint_vel.env_configs.reward_cfgs import UPPER_REWARDS as _BASE_UPPER
from rl.decoupled.tasks.joint_vel_hlip import mdp as hlip_mdp
from rl.managers.reward_group_manager import RewardGroupTermCfg

_LOWER = SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES)

LOWER_REWARDS: dict[str, RewardGroupTermCfg] = {
    "clf_reward": RewardGroupTermCfg(
        func=mdp.clf_reward,
        weight=10.0,
        params={"command_name": "hlip_ref", "eta_max": 0.3},
        group="lower_body",
    ),
    "clf_decreasing_condition": RewardGroupTermCfg(
        func=mdp.clf_decreasing_condition,
        weight=-2.0,
        params={"command_name": "hlip_ref", "eta_max": 0.2, "eta_dot_max": 0.3, "lam": 1.0},
        group="lower_body",
    ),
    "holonomic_constraint": RewardGroupTermCfg(
        func=hlip_mdp.holonomic_constraint,
        weight=4.0,
        params={"command_name": "foot_plan", "sigma_pose": (5 * 0.01) ** 0.5, "z_offset": 0.0},
        group="lower_body",
    ),
    "holonomic_constraint_vel": RewardGroupTermCfg(
        func=hlip_mdp.holonomic_constraint_vel,
        weight=2.0,
        params={"command_name": "foot_plan", "sigma_vel": 0.1 ** 0.5},
        group="lower_body",
    ),
    "lower_body_pos_limits": RewardGroupTermCfg(
        func=mjlab_vel_mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": _LOWER},
        group="lower_body",
    ),
    "lower_body_torques_l2": RewardGroupTermCfg(
        func=mjlab_vel_mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": _LOWER},
        group="lower_body",
    ),
    "lower_body_action_rate_l2": RewardGroupTermCfg(
        func=mdp.term_action_rate_l2,
        weight=-0.01,
        params={"action_term_name": "lower_body_joint_pos"},
        group="lower_body",
    ),
}

UPPER_REWARDS: dict[str, RewardGroupTermCfg] = copy.deepcopy(_BASE_UPPER)
