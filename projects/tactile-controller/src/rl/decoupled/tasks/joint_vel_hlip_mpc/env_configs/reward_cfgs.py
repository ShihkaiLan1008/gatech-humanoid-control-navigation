from rl.managers.reward_group_manager import RewardGroupTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp
import copy
import math
from rl.decoupled import mdp
from rl.decoupled.mdp import LOWER_JOINT_NAMES, UPPER_JOINT_NAMES

from .command_cfgs import FOOT_OFFSETS, GAIT_THRESHOLD

_LOWER = SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES)
_UPPER = SceneEntityCfg("robot", joint_names=UPPER_JOINT_NAMES)
_MPC_LOOKAHEAD_FRACS = (0.0, 0.25, 0.5, 0.75, 1.0)
_MPC_LOOKAHEAD_WEIGHTS = (0.5, 0.25, 0.15, 0.07, 0.03)


LOWER_REWARDS: dict[str, RewardGroupTermCfg] = {
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
    "pose": RewardGroupTermCfg(
        func=mjlab_vel_mdp.variable_posture,
        weight=1.0,
        params={
            "asset_cfg": _LOWER,  
            "command_name": "twist",
            "std_standing": {".*": 0.05},  # Set per-robot.
            "std_walking": {
                # Lower body.
                r".*hip_pitch.*": 0.5,
                r".*hip_roll.*": 0.15,
                r".*hip_yaw.*": 0.15,
                r".*knee.*": 0.5,
                r".*ankle_pitch.*": 0.25,
                r".*ankle_roll.*": 0.1,
                # Waist.
                r".*waist_yaw.*": 0.2,
                r".*waist_roll.*": 0.1,
                r".*waist_pitch.*": 0.1,
            },  # Set per-robot.
            "std_running": {
                # Lower body.
                r".*hip_pitch.*": 0.5,
                r".*hip_roll.*": 0.25,
                r".*hip_yaw.*": 0.25,
                r".*knee.*": 0.6,
                r".*ankle_pitch.*": 0.35,
                r".*ankle_roll.*": 0.15,
                # Waist.
                r".*waist_yaw.*": 0.3,
                r".*waist_roll.*": 0.1,
                r".*waist_pitch.*": 0.2,
            },  # Set per-robot.
            "walking_threshold": 0.05,
            "running_threshold": 1.5,
        },
        group="lower_body",
    ),
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
    "foot_clearance": RewardGroupTermCfg(
        func=mjlab_vel_mdp.feet_clearance,
        weight=-4.0,
        params={
            "target_height": 0.12,  # G1 swing height (Themis 0.15 at its scale)
            "height_sensor_name": "foot_height_scan",
            "command_name": "twist",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", site_names=('left_foot', 'right_foot')),  # Set per-robot.
        },
        group="lower_body",
    ),
    "foot_swing_height": RewardGroupTermCfg(
        func=mjlab_vel_mdp.feet_swing_height,
        weight=-0.25,
        params={
            "sensor_name": "feet_ground_contact",
            "height_sensor_name": "foot_height_scan",
            "target_height": 0.1,
            "command_name": "twist",
            "command_threshold": 0.05,
        },
        group="lower_body",
    ),
    "soft_landing": RewardGroupTermCfg(
        func=mjlab_vel_mdp.soft_landing,
        weight=-1e-5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.05,
        },
        group="lower_body",
    ),
    "foot_gait": RewardGroupTermCfg(
        func=mdp.mpc_feet_gait,
        weight=1.0,
        params={
            "sensor_name": "feet_ground_contact",
            # Phase comes from the "gait" command; stance window is DERIVED
            # from the MPC's duty factor so reward and schedule agree.
            "offset": FOOT_OFFSETS,
            "threshold": 0.5,
            "command_name": "twist",
            "command_threshold": 0.1,
        },
        group="lower_body",
    ),
    "stand_still": RewardGroupTermCfg(
        func=mdp.stand_still,
        weight=-1.0,
        params={
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": _LOWER, 
        },
        group="lower_body",
    ),
    "foot_flat_orientation": RewardGroupTermCfg(
        func=mdp.foot_flat_orientation,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=('left_foot', 'right_foot')),  # Set per-robot.
            "sigma": 0.15,
        },
        group="lower_body",
    ),
    "lower_body_self_collisions": RewardGroupTermCfg(
        func=mjlab_vel_mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "lower_self_collision", "force_threshold": 10.0},
        group="lower_body",
    ),
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
    "mpc_com_tracking": RewardGroupTermCfg(
        func=mdp.mpc_com_tracking,
        weight=2.0,
        params={
            "command_name": "loco_mpc",
            "w_pos": 1.0,
            "w_vel": 0.0,
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
            "w_vel": 4.0,
            "lookahead_fracs": _MPC_LOOKAHEAD_FRACS,
            "lookahead_weights": _MPC_LOOKAHEAD_WEIGHTS,
        },
        group="lower_body",
    ),
    "mpc_ang_vel_tracking": RewardGroupTermCfg(
        func=mdp.mpc_ang_vel_tracking,
        weight=4.0,
        params={
            "command_name": "loco_mpc",
            "w_ang": 2.0,
            "lookahead_fracs": _MPC_LOOKAHEAD_FRACS,
            "lookahead_weights": _MPC_LOOKAHEAD_WEIGHTS,
        },
        group="lower_body",
    ),
    "mpc_foot_placement": RewardGroupTermCfg(
        func=mdp.mpc_foot_placement_tracking,
        weight=1.0,
        params={
            "command_name": "loco_mpc",
            "sigma": 0.2,
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

# ── VForce variant: joint_vel_vforce_mpc's force family on this task's tables ─
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
FORCE_UPPER_REWARDS["hand_force_tracking_fine"] = RewardGroupTermCfg(
    func=mdp.hand_force_tracking,
    weight=1.0,
    params={"std": 5.0, "command_names": EE_COMMAND_NAMES},
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

FORCE_LOWER_REWARDS["stand_still"].weight = -2.5
FORCE_LOWER_REWARDS["mpc_grf_press_brace"] = RewardGroupTermCfg(
    func=mdp.mpc_grf_press_tracking,
    weight=0.5,
    params={
        "command_name": "loco_mpc",
        "grf_sensor_name": "feet_ground_contact",
        "std": 80.0,
    },
    group="lower_body",
)
