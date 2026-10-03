"""Reward declarations — Opt2Skill Table I structure (arXiv:2409.20514),
minus world-frame BASE anchoring: the base is free to drift, only the
hand <-> wall relationship is world-tied.

Frame logic (the load-bearing design point):
- joint-space tracking, velocity feedforward, yaw-rebased base orientation,
  anchor-relative feet terms: all DRIFT-AGNOSTIC.
- EE pos/ori tracking uses the clip's WORLD pose on purpose — the wall is
  world-fixed within an episode, so world EE tracking == reaching the
  commanded target-in-body-frame (the actor's motion_ee_pose_ref). This is
  the only term that ties the hand to the actual surface.
- the dropped terms (global anchor pos/ori, 14-body relative mimic pos/ori)
  are what re-anchored the old policy to the world frame / full mimic shape.

Force-penalty rule (project lesson): never square an unbounded contact force
in a penalty — exp kernels are bounded (fine), the deviation penalty is Huber.
"""

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .. import mdp
from .command_cfgs import EE_BODY_NAMES

REWARDS: dict[str, RewardTermCfg] = {
    # --- Opt2Skill joint-space tracking --------------------------------------
    "motion_joint_pos": RewardTermCfg(
        func=mdp.motion_joint_pos_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.45},
    ),
    # Torque reference teaches WHEN/HOW to load contacts (Pos+F+T ablation).
    "motion_joint_torque": RewardTermCfg(
        func=mdp.motion_joint_torque_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 30.0},
    ),
    # --- EE task-space tracking (world == commanded-target frame) ------------
    "motion_ee_pos": RewardTermCfg(
        func=mdp.motion_global_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.15, "body_names": EE_BODY_NAMES},
    ),
    # Sensor face stays pointed into the wall (EE x-axis = wall normal in the
    # reference; roll about it is what the TO left free).
    "motion_ee_ori": RewardTermCfg(
        func=mdp.motion_global_body_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4, "body_names": EE_BODY_NAMES},
    ),
    # --- base feedforward (drift-agnostic) -----------------------------------
    "motion_body_lin_vel": RewardTermCfg(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    ),
    # Upright/level base in the yaw-rebased relative frame (replaces the
    # world-frame anchor-ori term; collaborator requirement: low oscillation,
    # upright torso).
    "motion_base_ori": RewardTermCfg(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.4,
            "body_names": ("pelvis", "torso_link"),
        },
    ),
    # --- contact / force tracking --------------------------------------------
    "hand_force": RewardTermCfg(
        func=mdp.reward_hand_contact_force_exp,
        weight=1.0,
        params={"command_name": "motion", "sensor_name": "hand_wall", "std": 30.0},
    ),
    "hand_force_l2": RewardTermCfg(
        func=mdp.penalty_hand_contact_force_l2,
        weight=-2e-4,
        params={"command_name": "motion", "sensor_name": "hand_wall"},
    ),
    "hand_contact": RewardTermCfg(
        func=mdp.reward_hand_contact,
        weight=1.0,
        params={"command_name": "motion", "sensor_name": "hand_wall"},
    ),
    "feet_grf": RewardTermCfg(
        func=mdp.reward_feet_grf_exp,
        weight=0.25,
        params={
            "command_name": "motion",
            "sensor_name": "feet_ground",
            "std": 150.0,  # N; GRF scale ~200-350 N
        },
    ),
    "feet_contact": RewardTermCfg(
        func=mdp.reward_feet_contact,
        weight=1.0,
        params={"command_name": "motion", "sensor_name": "feet_ground"},
    ),
    # --- gait quality (anchor-relative = drift-agnostic; anti-skating) -------
    "motion_feet_pos": RewardTermCfg(
        func=mdp.motion_feet_pos_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.15},
    ),
    "feet_slide": RewardTermCfg(
        func=mdp.penalty_feet_slide,
        weight=-0.5,
        params={"command_name": "motion", "sensor_name": "feet_ground"},
    ),
    # --- regularization ------------------------------------------------------
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
    "joint_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-3e-6),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-1e-7),
    "joint_limit": RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-10.0,
        params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    "alive": RewardTermCfg(func=mdp.is_alive, weight=1.0),
}
