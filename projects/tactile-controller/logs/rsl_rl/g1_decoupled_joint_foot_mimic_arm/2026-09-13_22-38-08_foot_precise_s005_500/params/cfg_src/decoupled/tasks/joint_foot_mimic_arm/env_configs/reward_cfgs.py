"""Reward declarations — CLF-RL walking recipe over the step-clip references,
one price per quantity (lower body = joint_foot_mimic's clf_v3 table), and the
upper body tracking the per-hand `EEQposeCommand` tip poses (the shared
kernels) plus their reference joint positions (`ee_ref_joint_pos_tracking`).

Lower body = pelvis (pos, ori, lin vel) and feet (pos, ori) tracking against
the clip's links, joint pos tracking, torso attitude, the stance foot's
stillness, the CLF decrease condition, the phase-contact pair and four
bounded regularizers. Every exp kernel is exp(-Σe²/σ²); no L2 pairs by
design — the CLF decrease term is the unsaturating companion.
"""

from rl.managers.reward_group_manager import RewardGroupTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp
import copy
import math
from rl.decoupled import mdp
from rl.decoupled.mdp import LOWER_JOINT_NAMES, UPPER_JOINT_NAMES
from .command_cfgs import EE_COMMAND_NAMES

_LOWER = SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES)
_UPPER = SceneEntityCfg("robot", joint_names=UPPER_JOINT_NAMES)
_FEET = ("left_ankle_roll_link", "right_ankle_roll_link")

TRACK_BODY_GROUPS: dict[str, tuple[tuple[str, ...], str]] = {
    "pelvis": (("pelvis",), "sum"),
    "torso": (("torso_link",), "sum"),
    "feet": (("left_ankle_roll_link", "right_ankle_roll_link"), "sum"),
    "limbs": (("left_hip_yaw_link", "left_knee_link",
               "right_hip_yaw_link", "right_knee_link"), "mean"),
}
# kind -> (kernel, σ): m, rad, m/s, rad/s — the same for every group.
BODY_KERNELS = {
    "pos": (mdp.track_body_pos, 0.2),
    "ori": (mdp.track_body_ori, 0.4),
    "lin_vel": (mdp.track_body_lin_vel, 1.0),
    "ang_vel": (mdp.track_body_ang_vel, 1.4),
}
BODY_TRACK_WEIGHT = 1.0
# CLF decrease condition: Vdot <= -lam V over the declared error range.
CLF_DECREASE = dict(eta_max=0.2, eta_dot_max=0.3, lam=0.5)

LOWER_REWARDS: dict[str, RewardGroupTermCfg] = {}
for _grp, (_links, _reduce) in TRACK_BODY_GROUPS.items():
    for _kind, (_func, _std) in BODY_KERNELS.items():
        LOWER_REWARDS[f"track_{_grp}_{_kind}"] = RewardGroupTermCfg(
            func=_func,
            weight=BODY_TRACK_WEIGHT,
            params={"command_name": "step_cmd", "body_names": _links,
                    "std": _std, "reduce": _reduce},
            group="lower_body",
        )

# ============================================================
# Base Rewards
# ============================================================
LOWER_REWARDS.update({
    # A second, narrower foot-position kernel preserves the original 0.2 m
    # coarse signal while increasing the incentive below roughly 5 cm.  This
    # is the only reward change in the foot-accuracy experiment.
    "track_feet_pos_precise": RewardGroupTermCfg(
        func=mdp.track_body_pos,
        weight=1.0,
        params={"command_name": "step_cmd", "body_names": _FEET,
                "std": 0.05, "reduce": "sum"},
        group="lower_body",
    ),
    # task-space trunk attitude (the clip trunk is level); the pelvis's own
    # attitude is priced by track_pelvis_ori
    "torso_flat_orientation": RewardGroupTermCfg(
        func=mdp.body_flat_orientation_l2,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
        group="lower_body",
    ),
     "upright": RewardGroupTermCfg(
        func=mjlab_vel_mdp.upright,
        weight=1.0,
        params={
            "std": math.sqrt(0.2),
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        },
        group="lower_body",
    ),
    "track_joint_pos": RewardGroupTermCfg(
        func=mdp.track_joint_pos,
        weight=1.0,
        params={"command_name": "step_cmd", "std": 0.3 * 15 ** 0.5,
                "joint_names": LOWER_JOINT_NAMES},
        group="lower_body",
    ),
    "stance_foot_still": RewardGroupTermCfg(
        func=mdp.stance_foot_still,
        weight=2.0,
        params={"command_name": "step_cmd", "std": 0.32},
        group="lower_body",
    ),
    "clf_decreasing_condition": RewardGroupTermCfg(
        func=mdp.clf_decreasing_condition,
        weight=-3.0,
        params={"command_name": "step_cmd", **CLF_DECREASE},
        group="lower_body",
    ),
    # phase contact: the commanded swing foot must leave the ground inside its
    # swing window, the stance foot must never fly
    "swing_no_lift": RewardGroupTermCfg(
        func=mdp.swing_no_lift,
        weight=-2.0,
        params={"command_name": "step_cmd", "sensor_name": "feet_ground_contact"},
        group="lower_body",
    ),
    "stance_no_air": RewardGroupTermCfg(
        func=mdp.stance_no_air,
        weight=-2.0,
        params={"command_name": "step_cmd", "sensor_name": "feet_ground_contact"},
        group="lower_body",
    ),
    # regularizers
    "lower_body_action_rate_l2": RewardGroupTermCfg(
        func=mdp.term_action_rate_l2,
        weight=-0.05,
        params={"action_term_name": "lower_body_joint_pos"},
        group="lower_body",
    ),
    "joint_torque_l2": RewardGroupTermCfg(
        func=mjlab_vel_mdp.joint_torques_l2,
        weight=-3.0e-6,
        params={"asset_cfg": _LOWER},
        group="lower_body",
    ),
    "lower_body_pos_limits": RewardGroupTermCfg(
        func=mjlab_vel_mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": _LOWER},
        group="lower_body",
    ),
    "lower_body_self_collisions": RewardGroupTermCfg(
        func=mjlab_vel_mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "lower_self_collision", "force_threshold": 10.0},
        group="lower_body",
    ),
})



UPPER_REWARDS: dict[str, RewardGroupTermCfg] = {
    "track_ee_pos_coarse": RewardGroupTermCfg(
        func=mdp.ee_pos_tracking,
        weight=2.0,
        params={"std": 0.4, "command_names": EE_COMMAND_NAMES},
        group="upper_body",
    ),
    "track_ee_pos_fine": RewardGroupTermCfg(
        func=mdp.ee_pos_tracking,
        weight=2.0,
        params={"std": 0.1, "command_names": EE_COMMAND_NAMES},
        group="upper_body",
    ),
    "track_ee_pos_precise": RewardGroupTermCfg(
        func=mdp.ee_pos_tracking,
        weight=2.0,
        params={"std": 0.08, "command_names": EE_COMMAND_NAMES},
        group="upper_body",
    ),
    "track_ee_ori_coarse": RewardGroupTermCfg(
        func=mdp.ee_ori_tracking,
        weight=1.0,
        params={"std": 0.8, "command_names": EE_COMMAND_NAMES},
        group="upper_body",
    ),
    "track_ee_ori_fine": RewardGroupTermCfg(
        func=mdp.ee_ori_tracking,
        weight=1.0,
        params={"std": 0.5, "command_names": EE_COMMAND_NAMES},
        group="upper_body",
    ),
    "track_ee_ori_precise": RewardGroupTermCfg(
        func=mdp.ee_ori_tracking,
        weight=1.0,
        params={"std": 0.3, "command_names": EE_COMMAND_NAMES},
        group="upper_body",
    ),
    "track_left_ee_qpos_coarse": RewardGroupTermCfg(
        func=mdp.ee_ref_joint_pos_tracking,
        weight=1.0,
        params={"std": 1.0, "command_names": "left_ee_cmd"},
        group="upper_body",
    ),
    "track_left_ee_qpos_fine": RewardGroupTermCfg(
        func=mdp.ee_ref_joint_pos_tracking,
        weight=1.0,
        params={"std": 0.5, "command_names": "left_ee_cmd"},
        group="upper_body",
    ),
    "track_left_ee_qpos_precise": RewardGroupTermCfg(
        func=mdp.ee_ref_joint_pos_tracking,
        weight=1.0,
        params={"std": 0.3, "command_names": "left_ee_cmd"},
        group="upper_body",
    ),
    "track_right_ee_qpos_coarse": RewardGroupTermCfg(
        func=mdp.ee_ref_joint_pos_tracking,
        weight=1.0,
        params={"std": 1.0, "command_names": "right_ee_cmd"},
        group="upper_body",
    ),
    "track_right_ee_qpos_fine": RewardGroupTermCfg(
        func=mdp.ee_ref_joint_pos_tracking,
        weight=1.0,
        params={"std": 0.5, "command_names": "right_ee_cmd"},
        group="upper_body",
    ),
    "track_right_ee_qpos_precise": RewardGroupTermCfg(
        func=mdp.ee_ref_joint_pos_tracking,
        weight=1.0,
        params={"std": 0.3, "command_names": "right_ee_cmd"},
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
# Force layer (joint_foot_mimic's -VForce family; the qpos ladder stays as
# the posture term, unscaled during presses)
# ============================================================
FORCE_LOWER_REWARDS: dict[str, RewardGroupTermCfg] = copy.deepcopy(LOWER_REWARDS)
FORCE_UPPER_REWARDS: dict[str, RewardGroupTermCfg] = copy.deepcopy(UPPER_REWARDS)
FORCE_LOWER_REWARDS["upright"].group = "shared"
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
