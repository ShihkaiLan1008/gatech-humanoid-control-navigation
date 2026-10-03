from rl.managers.reward_group_manager import RewardGroupTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp
import math
from rl.decoupled import mdp
from rl.decoupled.mdp import LOWER_JOINT_NAMES, UPPER_JOINT_NAMES

_LOWER = SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES)
_UPPER = SceneEntityCfg("robot", joint_names=UPPER_JOINT_NAMES)
_ANKLES = SceneEntityCfg("robot", joint_names=(".*_ankle_pitch_joint",
                                               ".*_ankle_roll_joint"))

LOWER_REWARDS: dict[str, RewardGroupTermCfg] = {
    "track_linear_velocity": RewardGroupTermCfg(
        func=mdp.track_linear_velocity,
        weight=1.0,
        params={"command_name": "twist", "std": math.sqrt(0.25)},
        group="lower_body",
    ),
    "track_angular_velocity": RewardGroupTermCfg(
        func=mdp.track_angular_velocity,
        weight=1.0,
        params={"command_name": "twist", "std": math.sqrt(0.5)},
        group="lower_body",
    ),
    "body_orientation_l2": RewardGroupTermCfg(
        func=mdp.body_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},  # Set per-robot.
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
            r".*ankle_pitch.*": 0.15,
            r".*ankle_roll.*": 0.1,
            # Waist.
            r".*waist_yaw.*": 0.15,
            r".*waist_roll.*": 0.1,
            r".*waist_pitch.*": 0.1,
        },  # Set per-robot.
        "std_running": {
            # Lower body.
            r".*hip_pitch.*": 0.5,
            r".*hip_roll.*": 0.25,
            r".*hip_yaw.*": 0.25,
            r".*knee.*": 0.5,
            r".*ankle_pitch.*": 0.25,
            r".*ankle_roll.*": 0.1,
            # Waist.
            r".*waist_yaw.*": 0.25,
            r".*waist_roll.*": 0.1,
            r".*waist_pitch.*": 0.1,
        },  # Set per-robot.
        "walking_threshold": 0.1,
        "running_threshold": 1.5,
        },
        group="lower_body",
    ),
    "body_ang_vel": RewardGroupTermCfg(
        func=mjlab_vel_mdp.body_angular_velocity_penalty,
        weight=-0.05,  # Override per-robot
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},  # Set per-robot.
        group="lower_body",
    ),
    "angular_momentum": RewardGroupTermCfg(
        func=mjlab_vel_mdp.angular_momentum_penalty,
        weight=-0.025,  # Override per-robot
        params={"sensor_name": "robot/root_angmom"},
        group="lower_body",
    ),
    "is_terminated": RewardGroupTermCfg(func=mjlab_vel_mdp.is_terminated, weight=-200.0, group="shared",),
    "lower_body_joint_acc_l2": RewardGroupTermCfg(
        func=mjlab_vel_mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": _LOWER},
        group="lower_body",
    ),
    "lower_body_pos_limits": RewardGroupTermCfg(
        func=mjlab_vel_mdp.joint_pos_limits, 
        weight=-10.0,
        params={"asset_cfg": _LOWER},
        group="lower_body",
    ),
    "lower_body_action_rate_l2": RewardGroupTermCfg(
        func=mdp.term_action_rate_l2,
        weight=-0.05,
        params={"action_term_name": "lower_body_joint_pos"},
        group="lower_body",
    ),
    "foot_gait": RewardGroupTermCfg(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.6,
            "offset": [0.0, 0.5],
            "threshold": 0.56,
            "command_threshold": 0.1,
            "command_name": "twist",
            "sensor_name": "feet_ground_contact",
        },
        group="lower_body",
    ),
    "foot_clearance": RewardGroupTermCfg(
        func=mdp.feet_clearance,
        weight=-1.0,
        params={
            "target_height": 0.10,
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),  # Set per-robot.
        },
        group="lower_body",
    ),
    "foot_slip": RewardGroupTermCfg(
        func=mjlab_vel_mdp.feet_slip,
        weight=-0.25,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),  # Set per-robot.
        },
        group="lower_body",
    ),
    "soft_landing": RewardGroupTermCfg(
        func=mjlab_vel_mdp.soft_landing,
        weight=-1e-3,
        params={
            "sensor_name": "feet_ground_contact",
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
    "lower_body_self_collisions": RewardGroupTermCfg(
        func=mjlab_vel_mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "lower_self_collision", "force_threshold": 10.0},
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
    "arm_deviation_l1": RewardGroupTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": _UPPER},
        group="upper_body",
    ),
    "upper_body_action_rate_l2": RewardGroupTermCfg(
        func=mdp.term_action_rate_l2,
        weight=-0.01,
        params={"action_term_name": "upper_body_joint_pos"},
        group="upper_body",
    ),
    "upper_body_torque_limits": RewardGroupTermCfg(
        func=mdp.torque_soft_limits,
        weight=-0.1,
        params={"ratio": 0.9, "actuator_names": UPPER_JOINT_NAMES},
        group="upper_body",
    ),
    "upper_body_joint_vel_l2": RewardGroupTermCfg(
        func=mjlab_vel_mdp.joint_vel_l2,
        weight=-1e-3,
        params={"asset_cfg": _UPPER},
        group="upper_body",
    ),
    "upper_body_joint_acc_l2": RewardGroupTermCfg(
        func=mjlab_vel_mdp.joint_acc_l2,
        weight=-1e-6,
        params={"asset_cfg": _UPPER},
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