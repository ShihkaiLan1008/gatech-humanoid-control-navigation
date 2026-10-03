"""Reward declarations — locomotion_vel base + box-pushing terms.

Overrides on the base: MPC CoM tracking up-weighted; the pose reward is a
strong arm-holder (pushing posture) with tight shoulder/elbow/wrist stds; the
gait deadzone shrinks to the push threshold. Additions: hand-box contact,
box tracking, push/robot-box velocity matching, geometry costs, hand slip,
and MPC hand-force tracking.
"""

import copy

from mjlab.managers.reward_manager import RewardTermCfg

from ...locomotion_vel.env_configs.reward_cfgs import REWARDS as _VEL_REWARDS
from .. import mdp
from .command_cfgs import PUSH_CMD_THRESHOLD

REWARDS: dict[str, RewardTermCfg] = copy.deepcopy(_VEL_REWARDS)

# ── overrides on the locomotion base ────────────────────────────────────────
REWARDS["mpc_com_tracking"].weight = 3.0
REWARDS["foot_gait"].params["command_threshold"] = PUSH_CMD_THRESHOLD
REWARDS["stand_still"].params["command_threshold"] = PUSH_CMD_THRESHOLD
# Keep arms near the pushing posture (the arm_joint_deviation termination is
# ±0.25 rad about the loco-manip keyframe, so the pose reward must hold it).
# Override the per-joint keys the G1 base already defines (a broader
# ".*shoulder.*" key would multi-match and error).
REWARDS["pose"].weight = 5.0
for _std_key in ("std_walking", "std_running"):
    _stds = REWARDS["pose"].params[_std_key]
    for _k in (r".*shoulder_pitch.*", r".*shoulder_roll.*", r".*shoulder_yaw.*"):
        _stds[_k] = 0.05
    _stds[r".*elbow.*"] = 0.05
    _stds[r".*wrist.*"] = 0.10

# ── box-pushing additions ───────────────────────────────────────────────────
REWARDS["hand_box_contact"] = RewardTermCfg(
    func=mdp.hand_box_contact,
    weight=1.0,
    params={
        "lhand_sensor": "lhand_box_contact",
        "rhand_sensor": "rhand_box_contact",
        "both_hands_bonus": 0.5,
    },
)
REWARDS["box_com_tracking"] = RewardTermCfg(
    func=mdp.box_com_tracking,
    weight=4.0,
    params={
        "command_name": "loco_mpc",
        "box_name": "box",
        "lhand_sensor": "lhand_box_contact",
        "rhand_sensor": "rhand_box_contact",
        "w_pos": 2.0,
        "w_vel": 0.5,
        "lookahead_fracs": (0.0, 0.25, 0.5, 0.75, 1.0),
        "lookahead_weights": (0.35, 0.25, 0.20, 0.12, 0.08),
    },
)
REWARDS["push_velocity_match"] = RewardTermCfg(
    func=mdp.push_velocity_match,
    weight=5.0,
    params={
        "command_name": "twist",
        "box_name": "box",
        "robot_name": "robot",
        "lhand_sensor": "lhand_box_contact",
        "rhand_sensor": "rhand_box_contact",
        "sigma": 0.3,
    },
)
REWARDS["robot_box_velocity_match"] = RewardTermCfg(
    func=mdp.robot_box_velocity_match,
    weight=2.5,
    params={
        "box_name": "box",
        "robot_name": "robot",
        "lhand_sensor": "lhand_box_contact",
        "rhand_sensor": "rhand_box_contact",
        "sigma": 0.3,
    },
)
REWARDS["robot_box_xy_distance"] = RewardTermCfg(
    func=mdp.robot_box_xy_distance_cost,
    weight=-1.0,
    params={
        "box_name": "box",
        "robot_name": "robot",
        # G1 reach is shorter than Themis's (0.5 m there).
        "target_distance": 0.45,
    },
)
REWARDS["robot_box_yaw"] = RewardTermCfg(
    func=mdp.robot_box_yaw_cost,
    weight=-0.5,
    params={
        "box_name": "box",
        "robot_name": "robot",
    },
)
REWARDS["leg_box_collision"] = RewardTermCfg(
    func=mdp.leg_box_collision_cost,
    weight=-1.0,
    params={"sensor_name": "leg_box_contact"},
)
REWARDS["hand_slip"] = RewardTermCfg(
    func=mdp.HandSlipPenalty,
    weight=-0.5,
    params={
        "lhand_sensor": "lhand_box_contact",
        "rhand_sensor": "rhand_box_contact",
        "box_name": "box",
        "robot_name": "robot",
        # imu_in_torso replaces Themis's chest_camera site; the torso site
        # sits deeper in the body, so the radius grows accordingly.
        "reach_distance": 0.55,
        "camera_site_name": "imu_in_torso",
    },
)
REWARDS["mpc_hand_force_tracking"] = RewardTermCfg(
    func=mdp.mpc_hand_force_tracking,
    weight=1.0,
    params={
        "command_name": "loco_mpc",
        "lhand_sensor_name": "lhand_box_contact",
        "rhand_sensor_name": "rhand_box_contact",
        "sigma": 50.0,
    },
)
