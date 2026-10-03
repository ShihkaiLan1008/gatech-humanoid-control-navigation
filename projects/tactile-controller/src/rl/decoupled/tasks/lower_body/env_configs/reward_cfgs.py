from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp

from rl.decoupled import mdp
from rl.decoupled.mdp import LOWER_JOINT_NAMES

_LOWER = SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES)
_ANKLES = SceneEntityCfg("robot", joint_names=(".*_ankle_pitch_joint",
                                               ".*_ankle_roll_joint"))

REWARDS: dict[str, RewardTermCfg] = {
    "track_swing_foot_pos": RewardTermCfg(
        func=mdp.track_feet_pos,
        weight=5.0,
        params={"command_name": "step_cmd", "std": 0.05, "role": "swing"},
    ),
    "track_stance_foot_pos": RewardTermCfg(
        func=mdp.track_feet_pos,
        weight=1.0,          # keeps the anti-skate signal, no longer the bulk
        params={"command_name": "step_cmd", "std": 0.05, "role": "stance"},
    ),
    "track_swing_foot_yaw": RewardTermCfg(
        func=mdp.track_feet_yaw,
        weight=3.0,
        params={"command_name": "step_cmd", "std": 0.1, "role": "swing"},
    ),
    "track_stance_foot_yaw": RewardTermCfg(
        func=mdp.track_feet_yaw,
        weight=0.5,
        params={"command_name": "step_cmd", "std": 0.1, "role": "stance"},
    ),
    "track_com_pos": RewardTermCfg(
        func=mdp.track_com_pos,
        weight=1.0,
        params={"command_name": "step_cmd", "std": 0.08},
    ),
    "track_com_pos_l2": RewardTermCfg(
        func=mdp.track_com_pos_l2,
        weight=-1.0,
        params={"command_name": "step_cmd", "clamp": 0.3},
    ),
    "track_base_vel": RewardTermCfg(
        func=mdp.track_base_lin_vel,
        weight=1.0,
        params={"command_name": "step_cmd", "std": 0.25},
    ),
    "track_base_ang_vel": RewardTermCfg(
        func=mdp.track_base_ang_vel,
        weight=0.5,
        params={"command_name": "step_cmd", "std": 0.5},
    ),
    "track_joint_pos": RewardTermCfg(
        func=mdp.track_joint_pos,
        weight=0.5,
        params={"command_name": "step_cmd", "std": 0.3,
                "joint_names": LOWER_JOINT_NAMES},
    ),
    "track_feet_pos_l2": RewardTermCfg(
        func=mdp.track_feet_pos_l2,
        weight=-5.0,
        params={"command_name": "step_cmd"},
    ),
    "track_joint_pos_l2": RewardTermCfg(
        func=mdp.track_joint_pos_l2,
        weight=-0.1,
        params={"command_name": "step_cmd", "joint_names": LOWER_JOINT_NAMES},
    ),
    "track_joint_vel": RewardTermCfg(
        func=mdp.track_joint_vel,
        weight=0.3,
        params={"command_name": "step_cmd", "std": 1.0,
                "joint_names": LOWER_JOINT_NAMES},
    ),
    "track_joint_vel_l2": RewardTermCfg(
        func=mdp.track_joint_vel_l2,
        weight=-0.01,
        params={"command_name": "step_cmd", "joint_names": LOWER_JOINT_NAMES},
    ),
    "track_joint_torque": RewardTermCfg(
        func=mdp.track_joint_torque,
        weight=0.3,
        params={"command_name": "step_cmd", "std": 30.0,
                "joint_names": LOWER_JOINT_NAMES},
    ),
    "track_joint_torque_l2": RewardTermCfg(
        func=mdp.track_joint_torque_l2,
        weight=-1e-4,
        params={"command_name": "step_cmd", "clamp": 30.0,
                "joint_names": LOWER_JOINT_NAMES},
    ),
    "ankle_posture": RewardTermCfg(
        func=mdp.ankle_posture_l2,
        weight=-2.0,
        params={
            "asset_cfg": _ANKLES,
            "targets": {".*_ankle_pitch_joint": -0.2, ".*_ankle_roll_joint": 0.0},
        },
    ),
    "landing_air_time": RewardTermCfg(
        func=mdp.landing_air_time,
        weight=20.0,
        params={"sensor_name": "feet_ground_contact", "threshold": 0.4},
    ),
    "stance_no_air": RewardTermCfg(
        func=mdp.stance_no_air,
        weight=-2.0,
        params={"command_name": "step_cmd", "sensor_name": "feet_ground_contact"},
    ),
    "swing_no_lift": RewardTermCfg(
        func=mdp.swing_no_lift,
        weight=-2.0,
        params={"command_name": "step_cmd", "sensor_name": "feet_ground_contact"},
    ),
    "base_height_range": RewardTermCfg(
        func=mdp.base_height_l2,
        weight=-3.0,
        params={"target_height": 0.78},
    ),
    "foot_stumble": RewardTermCfg(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_name": "feet_ground_contact", "threshold": 4.0},
    ),
    "feet_slide": RewardTermCfg(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_name": "feet_ground_contact",
            "asset_cfg": SceneEntityCfg("robot", body_names=(
                "left_ankle_roll_link", "right_ankle_roll_link")),
            "force_threshold": 1.0,
        },
    ),
    "alive": RewardTermCfg(
        func=mjlab_vel_mdp.is_alive,
        weight=0.15,
    ),
    # ── regularization (curriculum ramps the first two — see curriculum_cfgs)
    "action_rate_l2": RewardTermCfg(
        func=mjlab_vel_mdp.action_rate_l2,
        weight=-0.05,
    ),
    "joint_acc_l2": RewardTermCfg(
        func=mjlab_vel_mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": _LOWER},
    ),
    "joint_vel_l2": RewardTermCfg(
        func=mjlab_vel_mdp.joint_vel_l2,
        weight=-1.0e-3,
        params={"asset_cfg": _LOWER},
    ),
    "joint_torque_l2": RewardTermCfg(
        func=mjlab_vel_mdp.joint_torques_l2,
        weight=-3.0e-6,
        params={"asset_cfg": _LOWER},
    ),
    "dof_pos_limits": RewardTermCfg(
        func=mjlab_vel_mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": _LOWER},
    ),
    "self_collisions": RewardTermCfg(
        func=mjlab_vel_mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    "pelvis_flat_orientation": RewardTermCfg(
        func=mjlab_vel_mdp.flat_orientation_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "torso_flat_orientation": RewardTermCfg(
        func=mdp.body_flat_orientation_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
    ),
    "joint_deviation_waists": RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=("waist.*",))},
    ),
    "waist_vel_l2": RewardTermCfg(
        func=mjlab_vel_mdp.joint_vel_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=("waist.*",))},
    ),
    "soft_landing": RewardTermCfg(
        func=mjlab_vel_mdp.soft_landing,
        weight=-1e-5,
        params={
            "sensor_name": "feet_ground_contact",
        },
    )
}
