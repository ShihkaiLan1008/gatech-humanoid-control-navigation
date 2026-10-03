import copy
from rl.managers.reward_group_manager import RewardGroupTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp

from rl.g1_assets.g1_constants import PELVIS_KNEES_BENT_Z
from rl.decoupled import mdp
from rl.decoupled.mdp import LOWER_JOINT_NAMES, UPPER_JOINT_NAMES
import math

_LOWER = SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES)
_UPPER = SceneEntityCfg("robot", joint_names=UPPER_JOINT_NAMES)
_ANKLES = SceneEntityCfg("robot", joint_names=(".*_ankle_pitch_joint",
                                               ".*_ankle_roll_joint"))
_TORSO = SceneEntityCfg("robot", body_names=("torso_link",))
# Node 0 of the plan is the measured state at the last solve: never priced.
_MPC_LOOKAHEAD_FRACS = (0.25, 0.5, 0.75, 1.0)
_MPC_LOOKAHEAD_WEIGHTS = (0.4, 0.3, 0.2, 0.1)

LOWER_REWARDS: dict[str, RewardGroupTermCfg] = {
    # ============================================================
    # Posture
    # ============================================================
    "upright": RewardGroupTermCfg(
        func=mjlab_vel_mdp.upright,
        weight=1.0,
        params={
            "std": math.sqrt(0.2),
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        },
        group="lower_body",
    ),
    "pelvis_flat_orientation": RewardGroupTermCfg(
        func=mjlab_vel_mdp.flat_orientation_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
        group="lower_body",
    ),
    "torso_flat_orientation": RewardGroupTermCfg(
        func=mdp.body_flat_orientation_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
        group="lower_body",
    ),
    "joint_deviation_waists": RewardGroupTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=("waist.*",))},
        group="lower_body",
    ),
    "body_ang_vel": RewardGroupTermCfg(
        func=mjlab_vel_mdp.body_angular_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},  # Set per-robot.
        group="lower_body",
    ),
    "pelvis_height": RewardGroupTermCfg(
        func=mdp.base_height_range,
        weight=-10.0,
        params={"min_height": PELVIS_KNEES_BENT_Z - 0.015, "max_height": 0.78,
                "squared": False},
        group="lower_body",
    ),
    # ============================================================
    # Regularization
    # ============================================================
    "lower_body_joint_acc_l2": RewardGroupTermCfg(
        func=mjlab_vel_mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": _LOWER},
        group="lower_body",
    ),
    "lower_body_pos_limits": RewardGroupTermCfg(
        func=mjlab_vel_mdp.joint_pos_limits, 
        weight=-1.0,
        params={"asset_cfg": _LOWER},
        group="lower_body",
    ),
    "lower_body_action_rate_l2": RewardGroupTermCfg(
        func=mdp.term_action_rate_l2,
        weight=-0.05,
        params={"action_term_name": "lower_body_joint_pos"},
        group="lower_body",
    ),
    "lower_body_self_collisions": RewardGroupTermCfg(
        func=mjlab_vel_mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "lower_self_collision", "force_threshold": 10.0},
        group="lower_body",
    ),

    # ============================================================
    # MPC
    # ============================================================
    "mpc_grf_tracking": RewardGroupTermCfg(
        func=mdp.mpc_grf_tracking,
        weight=0.002,
        params={
            "command_name": "loco_mpc",
            "grf_sensor_name": "feet_ground_contact",
        },
        group="lower_body",
    ),
    "mpc_ang_mom": RewardGroupTermCfg(
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
        group="lower_body",
    ),
    "mpc_com_vel_tracking": RewardGroupTermCfg(
        func=mdp.mpc_com_vel_tracking,
        weight=2.0,
        params={
            "command_name": "loco_mpc",
            "w_vel": 16.0,    # sigma 0.25 m/s
            "lookahead_fracs": _MPC_LOOKAHEAD_FRACS,
            "lookahead_weights": _MPC_LOOKAHEAD_WEIGHTS,
        },
        group="lower_body",
    ),
    "mpc_ang_vel_tracking": RewardGroupTermCfg(
        func=mdp.mpc_ang_vel_tracking,
        weight=1.0,
        params={
            "command_name": "loco_mpc",
            "w_ang": 2.0,
            "lookahead_fracs": _MPC_LOOKAHEAD_FRACS,
            "lookahead_weights": _MPC_LOOKAHEAD_WEIGHTS,
        },
        group="lower_body",
    ),
    "mpc_com_tracking": RewardGroupTermCfg(
        func=mdp.mpc_com_tracking,
        weight=2.0,
        params={
            "command_name": "loco_mpc",
            "w_pos": 100.0,
            "w_vel": 0.0,
            "lookahead_fracs": _MPC_LOOKAHEAD_FRACS,
            "lookahead_weights": _MPC_LOOKAHEAD_WEIGHTS,
        },
        group="lower_body",
    ),
    # ============================================================
    # Foot
    # ============================================================
    "track_swing_foot_pos": RewardGroupTermCfg(
        func=mdp.track_feet_pos,
        weight=5.0,
        params={"command_name": "foothold", "std": 0.05, "role": "swing",
                "hold_neutral": True, "airborne_only": True,
                "sensor_name": "feet_ground_contact",
                "airborne_release": 0.85},
        group="lower_body",
    ),
    "track_stance_foot_pos": RewardGroupTermCfg(
        func=mdp.track_feet_pos,
        weight=1.0,          # keeps the anti-skate signal, no longer the bulk
        params={"command_name": "foothold", "std": 0.05, "role": "stance",
                "hold_neutral": True},
        group="lower_body",
    ),
    "track_swing_foot_yaw": RewardGroupTermCfg(
        func=mdp.track_feet_yaw,
        weight=3.0,
        params={"command_name": "foothold", "std": 0.1, "role": "swing",
                "hold_neutral": True, "airborne_only": True,
                "sensor_name": "feet_ground_contact",
                "airborne_release": 0.85},
        group="lower_body",
    ),
    "track_stance_foot_yaw": RewardGroupTermCfg(
        func=mdp.track_feet_yaw,
        weight=0.5,
        params={"command_name": "foothold", "std": 0.1, "role": "stance",
                "hold_neutral": True},
        group="lower_body",
    ),
    "track_feet_pos_l2": RewardGroupTermCfg(
        func=mdp.track_feet_pos_l2,
        weight=-5.0,
        params={"command_name": "foothold", "clamp": 0.3, "hold_neutral": True},
        group="lower_body",
    ),
    "feet_contact_schedule": RewardGroupTermCfg(
        func=mdp.feet_contact_schedule,
        weight=2.0,
        params={"command_name": "foothold", "sensor_name": "feet_ground_contact"},
        group="lower_body",
    ),
    # Pelvis heading vs the stance foot: the frame DOF the foothold chain
    # leaves free (hip yaw); unpriced it is sold as a twisted lateral gait.
    "foothold_heading": RewardGroupTermCfg(
        func=mdp.foothold_heading,
        weight=0.5,
        params={"command_name": "foothold", "std": 0.5},
        group="lower_body",
    ),
    "foothold_heading_l2": RewardGroupTermCfg(
        func=mdp.foothold_heading_l2,
        weight=-2.0,
        params={"command_name": "foothold", "clamp": 1.5},
        group="lower_body",
    ),
    "soft_landing": RewardGroupTermCfg(
        func=mjlab_vel_mdp.soft_landing,
        weight=-1e-5,
        params={
            "sensor_name": "feet_ground_contact",
        },
        group="lower_body",
    ),
    "foot_tilt": RewardGroupTermCfg(
        func=mdp.foot_tilt_l2,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "sensor_name": "feet_ground_contact",
        },
        group="lower_body",
    ),
}

UPPER_REWARDS: dict[str, RewardGroupTermCfg] = {
    "track_ee_pos_coarse": RewardGroupTermCfg(
        func=mdp.ee_pos_tracking,
        weight=2.0,
        params={"std": 0.4, "command_names": ("left_ee_cmd", "right_ee_cmd")},
        group="upper_body",
    ),
    "track_ee_pos_fine": RewardGroupTermCfg(
        func=mdp.ee_pos_tracking,
        weight=2.0,
        params={"std": 0.1, "command_names": ("left_ee_cmd", "right_ee_cmd")},
        group="upper_body",
    ),
    "track_ee_pos_precise": RewardGroupTermCfg(
        func=mdp.ee_pos_tracking,
        weight=2.0,
        params={"std": 0.08, "command_names": ("left_ee_cmd", "right_ee_cmd")},
        group="upper_body",
    ),
    "track_ee_ori_coarse": RewardGroupTermCfg(
        func=mdp.ee_ori_tracking,
        weight=1.0,
        params={"std": 0.8, "command_names": ("left_ee_cmd", "right_ee_cmd")},
        group="upper_body",
    ),
    "track_ee_ori_fine": RewardGroupTermCfg(
        func=mdp.ee_ori_tracking,
        weight=1.0,
        params={"std": 0.5, "command_names": ("left_ee_cmd", "right_ee_cmd")},
        group="upper_body",
    ),
    "track_ee_ori_precise": RewardGroupTermCfg(
        func=mdp.ee_ori_tracking,
        weight=1.0,
        params={"std": 0.3, "command_names": ("left_ee_cmd", "right_ee_cmd")},
        group="upper_body",
    ),
    "arm_swivel_exp": RewardGroupTermCfg(
        func=mdp.arm_swivel_exp,
        weight=1.0,
        params={"std": 0.5, "psi_target": 0.05,
                "command_names": ("left_ee_cmd", "right_ee_cmd")},
        group="upper_body",
    ),
    "arm_swivel_l1": RewardGroupTermCfg(
        func=mdp.arm_swivel_l1,
        weight=-0.3,
        params={"psi_target": 0.05,
                "command_names": ("left_ee_cmd", "right_ee_cmd")},
        group="upper_body",
    ),
    "arm_deviation_l1": RewardGroupTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": _UPPER},
        group="upper_body",
    ),
    "upper_body_action_rate_l2": RewardGroupTermCfg(
        func=mdp.term_action_rate_l2,
        weight=-0.05,
        params={"action_term_name": "upper_body_joint_pos"},
        group="upper_body",
    ),
    "upper_body_joint_torque_l2": RewardGroupTermCfg(
        func=mjlab_vel_mdp.joint_torques_l2,
        weight=-2e-5,
        params={"asset_cfg": _UPPER},
        group="upper_body",
    ),
    "upper_body_joint_vel_l2": RewardGroupTermCfg(
        func=mjlab_vel_mdp.joint_vel_l2,
        weight=-1e-3,
        params={"asset_cfg": _UPPER},
        group="upper_body",
    ),
    "upper_body_joint_acc_l2": RewardGroupTermCfg(
        func=mdp.joint_acc_l2,
        weight=-1e-6,
        params={"asset_cfg": _UPPER, "clamp": 400.0},
        group="upper_body",
    ),
    "ee_acc_l2": RewardGroupTermCfg(
        func=mdp.ee_acc_l2,
        weight=-1.0e-2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=("left_tactile", "right_tactile")),
            "clamp": 20.0,
        },
        group="upper_body",
    ),
    "upper_body_dof_pos_limits": RewardGroupTermCfg(
        func=mjlab_vel_mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": _UPPER},
        group="upper_body",
    ),
    "upper_body_self_collisions": RewardGroupTermCfg(
        func=mjlab_vel_mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "upper_self_collision", "force_threshold": 10.0},
        group="upper_body",
    ),
}


# ============================================================
# Posture prior (the -Prior task only; the base dicts stay A)
# ============================================================
PRIOR_REWARDS: dict[str, RewardGroupTermCfg] = {
    "foothold_posture_prior": RewardGroupTermCfg(
        func=mdp.foothold_posture_prior,
        weight=1.0,
        params={"command_name": "foothold", "std": 0.3},
        group="lower_body",
    ),
}


# ============================================================
# Force Rewards
# ============================================================
FORCE_LOWER_REWARDS: dict[str, RewardGroupTermCfg] = copy.deepcopy(LOWER_REWARDS)
FORCE_UPPER_REWARDS: dict[str, RewardGroupTermCfg] = copy.deepcopy(UPPER_REWARDS)

EE_COMMAND_NAMES = ("left_ee_cmd", "right_ee_cmd")

FORCE_UPPER_REWARDS["hand_contact_hold"] = RewardGroupTermCfg(
    func=mdp.hand_contact_hold,
    weight=1.5,
    params={"floor": 3.0, "grace_s": 1.0, "command_names": EE_COMMAND_NAMES},
    group="upper_body",
)
FORCE_UPPER_REWARDS["hand_force_tracking"] = RewardGroupTermCfg(
    func=mdp.hand_force_tracking,
    weight=1.0,
    params={"std": 15.0, "command_names": EE_COMMAND_NAMES},
    group="upper_body",
)
FORCE_UPPER_REWARDS["hand_force_l1"] = RewardGroupTermCfg(
    func=mdp.hand_force_l1,
    weight=-0.02,
    params={"clamp": 40.0, "command_names": EE_COMMAND_NAMES},
    group="upper_body",
)
FORCE_UPPER_REWARDS["hand_force_rate"] = RewardGroupTermCfg(
    func=mdp.HandForceRate,
    weight=-0.05,
    params={"clamp": 20.0, "command_names": EE_COMMAND_NAMES},
    group="upper_body",
)
FORCE_UPPER_REWARDS["hand_force_tracking_fine"] = RewardGroupTermCfg(
    func=mdp.hand_force_tracking,
    weight=1.0,
    params={"std": 5.0, "command_names": EE_COMMAND_NAMES},
    group="upper_body",
)
FORCE_UPPER_REWARDS["arm_anchor_mimic_coarse"] = RewardGroupTermCfg(
    func=mdp.arm_anchor_mimic_exp,
    weight=0.5,
    params={"std": 1.5, "command_names": EE_COMMAND_NAMES, "press_scale": 0.3},
    group="upper_body",
)
FORCE_UPPER_REWARDS["arm_anchor_mimic_exp"] = RewardGroupTermCfg(
    func=mdp.arm_anchor_mimic_exp,
    weight=1.0,
    params={"std": 0.35, "command_names": EE_COMMAND_NAMES, "press_scale": 0.3},
    group="upper_body",
)
FORCE_UPPER_REWARDS["arm_anchor_mimic_l2"] = RewardGroupTermCfg(
    func=mdp.arm_anchor_mimic_l2,
    weight=-0.15,
    params={"command_names": EE_COMMAND_NAMES, "press_scale": 0.3},
    group="upper_body",
)
FORCE_LOWER_REWARDS["mpc_grf_press_brace"] = RewardGroupTermCfg(
    func=mdp.mpc_grf_press_tracking,
    weight=0.5,
    params={
        "command_name": "loco_mpc",
        "grf_sensor_name": "feet_ground_contact",
        "std": 80.0,  # measured rms p50 vs plan = 79 N (gradient-alive)
    },
    group="lower_body",
)