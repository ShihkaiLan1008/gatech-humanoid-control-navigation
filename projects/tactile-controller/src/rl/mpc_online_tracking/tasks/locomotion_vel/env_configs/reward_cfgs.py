"""Reward declarations — MPC-primary structure (the v2 reward family).

The MPC plan carries locomotion (CoM / velocity / angular momentum / GRF /
foot placement tracking); the raw twist-tracking rewards of the stock velocity
task are deliberately ABSENT. Regularizers (upright, pose, limits, gait shape)
come from the mjlab velocity stack with G1-tuned parameters.
"""

import math

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp

from .. import mdp
from .command_cfgs import FOOT_SITE_NAMES, GAIT_PERIOD

# Per-foot phase offsets for the feet_gait clock reward, in SENSOR COLUMN
# order. The G1 model declares the LEFT leg first, so the feet_ground_contact
# columns are [left, right] — the OPPOSITE of Themis (right leg first). The
# MPC schedule pins LF stance to sin(phi+pi) < thr, i.e. the left foot is the
# offset-0.5 foot. Using Themis's [0.0, 0.5] here anti-phases the clock reward
# against the MPC's per-foot schedule (measured on the 2026-07-21 run:
# schedule agreement 17 % as trained, 83 % with the feet swapped — the policy
# followed the clock, so the per-foot MPC channels taught nothing).
FOOT_OFFSETS = [0.5, 0.0]
GAIT_THRESHOLD = 0.5

_MPC_LOOKAHEAD_FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)
_MPC_LOOKAHEAD_WEIGHTS = (0.5, 0.25, 0.15, 0.07, 0.03)

REWARDS: dict[str, RewardTermCfg] = {
    # ── posture / regularization (mjlab velocity stack, G1-tuned) ───────────
    "upright": RewardTermCfg(
        func=mjlab_vel_mdp.upright,
        weight=1.0,
        params={
            "std": math.sqrt(0.2),
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        },
    ),
    "pose": RewardTermCfg(
        func=mjlab_vel_mdp.variable_posture,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "command_name": "twist",
            "std_standing": {".*": 0.05},
            # Knees/hip_pitch loosest for natural stride; hip roll/yaw and
            # ankle roll tight for lateral stability; waist roll/pitch tight
            # for an upright torso; arms moderate for swing.
            "std_walking": {
                r".*hip_pitch.*": 0.3,
                r".*hip_roll.*": 0.15,
                r".*hip_yaw.*": 0.15,
                r".*knee.*": 0.35,
                r".*ankle_pitch.*": 0.25,
                r".*ankle_roll.*": 0.1,
                r".*waist_yaw.*": 0.2,
                r".*waist_roll.*": 0.08,
                r".*waist_pitch.*": 0.1,
                r".*shoulder_pitch.*": 0.15,
                r".*shoulder_roll.*": 0.15,
                r".*shoulder_yaw.*": 0.1,
                r".*elbow.*": 0.15,
                r".*wrist.*": 0.3,
            },
            "std_running": {
                r".*hip_pitch.*": 0.5,
                r".*hip_roll.*": 0.2,
                r".*hip_yaw.*": 0.2,
                r".*knee.*": 0.6,
                r".*ankle_pitch.*": 0.35,
                r".*ankle_roll.*": 0.15,
                r".*waist_yaw.*": 0.3,
                r".*waist_roll.*": 0.08,
                r".*waist_pitch.*": 0.2,
                r".*shoulder_pitch.*": 0.5,
                r".*shoulder_roll.*": 0.2,
                r".*shoulder_yaw.*": 0.15,
                r".*elbow.*": 0.35,
                r".*wrist.*": 0.3,
            },
            "walking_threshold": 0.05,
            "running_threshold": 1.5,
        },
    ),
    "dof_pos_limits": RewardTermCfg(func=mjlab_vel_mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mjlab_vel_mdp.action_rate_l2, weight=-0.25),
    "foot_clearance": RewardTermCfg(
        func=mjlab_vel_mdp.feet_clearance,
        weight=-4.0,
        params={
            "target_height": 0.12,  # G1 swing height (Themis 0.15 at its scale)
            "height_sensor_name": "foot_height_scan",
            "command_name": "twist",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
        },
    ),
    "foot_swing_height": RewardTermCfg(
        func=mjlab_vel_mdp.feet_swing_height,
        weight=-0.25,
        params={
            "sensor_name": "feet_ground_contact",
            "height_sensor_name": "foot_height_scan",
            "target_height": 0.1,
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    ),
    "soft_landing": RewardTermCfg(
        func=mjlab_vel_mdp.soft_landing,
        weight=-1e-5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    ),
    "self_collisions": RewardTermCfg(
        func=mjlab_vel_mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    "feet_too_near": RewardTermCfg(
        func=mdp.feet_too_near,
        weight=-0.50,
        params={
            "min_distance": 0.12,
            "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
        },
    ),
    "gait_symmetry": RewardTermCfg(
        func=mdp.gait_symmetry,
        weight=0.00,
        params={
            "left_sensor_name": "left_foot_ground",
            "right_sensor_name": "right_foot_ground",
        },
    ),
    # ── phase-locked gait shape ─────────────────────────────────────────────
    "foot_gait": RewardTermCfg(
        func=mdp.feet_gait,
        weight=1.0,
        params={
            "sensor_name": "feet_ground_contact",
            "period": GAIT_PERIOD,
            "offset": FOOT_OFFSETS,
            "threshold": GAIT_THRESHOLD,
            "command_name": "twist",
            "command_threshold": 0.1,
        },
    ),
    "stand_still": RewardTermCfg(
        func=mdp.stand_still,
        weight=-4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "command_name": "twist",
            "command_threshold": 0.1,
        },
    ),
    "foot_flat_orientation": RewardTermCfg(
        func=mdp.foot_flat_orientation,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
            "sigma": 0.15,
        },
    ),
    # ── MPC-plan tracking (the primary locomotion signal) ───────────────────
    "mpc_grf_tracking": RewardTermCfg(
        func=mdp.mpc_grf_tracking,
        weight=0.002,
        params={
            "command_name": "loco_mpc",
            "grf_sensor_name": "feet_ground_contact",
        },
    ),
    "mpc_ang_mom": RewardTermCfg(
        func=mdp.MpcAngMomTracking,
        weight=0.05,
        params={
            "command_name": "loco_mpc",
            "w_k": 1.0,
            "w_kdot": 0.01,
            "q_k": (1.0, 1.0, 1.0),
            "q_kdot": (0.1, 0.1, 0.05),
            "lookahead_fracs": _MPC_LOOKAHEAD_FRACS,
            "lookahead_weights": _MPC_LOOKAHEAD_WEIGHTS,
        },
    ),
    "mpc_com_tracking": RewardTermCfg(
        func=mdp.mpc_com_tracking,
        weight=2.0,
        params={
            "command_name": "loco_mpc",
            "w_pos": 1.0,
            "w_vel": 0.0,
            "lookahead_fracs": _MPC_LOOKAHEAD_FRACS,
            "lookahead_weights": _MPC_LOOKAHEAD_WEIGHTS,
        },
    ),
    "mpc_com_vel_tracking": RewardTermCfg(
        func=mdp.mpc_com_vel_tracking,
        weight=2.0,
        params={
            "command_name": "loco_mpc",
            "w_vel": 4.0,
            "lookahead_fracs": _MPC_LOOKAHEAD_FRACS,
            "lookahead_weights": _MPC_LOOKAHEAD_WEIGHTS,
        },
    ),
    "mpc_ang_vel_tracking": RewardTermCfg(
        func=mdp.mpc_ang_vel_tracking,
        weight=4.0,
        params={
            "command_name": "loco_mpc",
            "w_ang": 2.0,
            "lookahead_fracs": _MPC_LOOKAHEAD_FRACS,
            "lookahead_weights": _MPC_LOOKAHEAD_WEIGHTS,
        },
    ),
    "mpc_foot_placement": RewardTermCfg(
        func=mdp.mpc_foot_placement_tracking,
        weight=1.0,
        params={
            "command_name": "loco_mpc",
            "sigma": 0.2,
        },
    ),
}
