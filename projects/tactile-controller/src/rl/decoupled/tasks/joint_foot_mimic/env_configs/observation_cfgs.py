"""Observation declarations — REFINE-DP Table I (locomotion block).

TERM ORDER IS THE CHECKPOINT OBS LAYOUT — never reorder existing terms, append
only. Actor = deployable proprio, single frame + previous action, NO base
linear velocity (privileged); critic adds base lin vel, reference/current feet
positions and the foot contact wrench — measured contact force stays OUT of
the actor per the standing user rule.

Both actors share the SAME full-body proprio block (all 29 joints + the full
29-dim action concat) — user decision: the policies coordinate through shared
whole-body observability; only the command block differs per group.
"""

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise
import copy
from rl.decoupled import mdp
from rl.decoupled.mdp import LOWER_JOINT_NAMES
from mjlab.managers.scene_entity_config import SceneEntityCfg
_LOWER = SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES)

# ============================================
# Shared Actor term
# ============================================

ACTOR_TERMS: dict[str, ObservationTermCfg] = {
    "base_ang_vel": ObservationTermCfg(
        func=mjlab_vel_mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        noise=Unoise(n_min=-0.2, n_max=0.2),
        history_length=5,
    ),
    "projected_gravity": ObservationTermCfg(
        func=mjlab_vel_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
        history_length=5,
    ),
    "joint_pos": ObservationTermCfg(
        func=mjlab_vel_mdp.joint_pos_rel,
        params={"biased": True},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        history_length=5,
    ),
    "joint_vel": ObservationTermCfg(
        func=mjlab_vel_mdp.joint_vel_rel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        history_length=5,
    ),
    "actions": ObservationTermCfg(
        func=mjlab_vel_mdp.last_action,
        history_length=5
    ),
}

# ============================================
# Upper body Actor
# ============================================
UPPER_ACTOR_TERMS: dict[str, ObservationTermCfg] = copy.deepcopy(ACTOR_TERMS)
UPPER_ACTOR_TERMS["left_ee_commands"] = ObservationTermCfg(
    func=mjlab_vel_mdp.generated_commands,
    params={"command_name": "left_ee_cmd"},
)
UPPER_ACTOR_TERMS["right_ee_commands"] = ObservationTermCfg(
    func=mjlab_vel_mdp.generated_commands,
    params={"command_name": "right_ee_cmd"},
)
UPPER_ACTOR_TERMS["measured_ee_pose"] = ObservationTermCfg(
    func=mdp.ee_pose_b,
    params={"command_names": ("left_ee_cmd", "right_ee_cmd"),
            "pos_noise": 0.02, "rot_noise": 0.05},
)
UPPER_ACTOR_TERMS["ee_pose_err"] = ObservationTermCfg(
    func=mdp.ee_pose_err,
    params={"command_names": ("left_ee_cmd", "right_ee_cmd"),
            "pos_noise": 0.02, "rot_noise": 0.05},
)
UPPER_ACTOR_TERMS["hand_contact_force_cmd"] = ObservationTermCfg(
    func=mdp.zero_obs, params={"dim": 8},
)

# ============================================
# Lower body Actor
# ============================================
LOWER_ACTOR_TERMS: dict[str, ObservationTermCfg] = copy.deepcopy(ACTOR_TERMS)
LOWER_ACTOR_TERMS["step_commands"] = ObservationTermCfg(
    func=mjlab_vel_mdp.generated_commands,
    params={"command_name": "step_cmd"},
)
LOWER_ACTOR_TERMS["hand_contact_force_cmd"] = ObservationTermCfg(
    func=mdp.zero_obs, params={"dim": 8},   # reserved (see upper actor note)
)

# ============================================
# Shared Critic
# ============================================
CRITIC_TERMS: dict[str, ObservationTermCfg] = {
    "base_lin_vel": ObservationTermCfg(
        func=mjlab_vel_mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_vel"},
        history_length=5,
    ),
    "base_ang_vel": ObservationTermCfg(
        func=mjlab_vel_mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        history_length=5,
    ),
    "projected_gravity": ObservationTermCfg(func=mjlab_vel_mdp.projected_gravity, history_length=5),
    "joint_pos": ObservationTermCfg(func=mjlab_vel_mdp.joint_pos_rel, history_length=5),
    "joint_vel": ObservationTermCfg(func=mjlab_vel_mdp.joint_vel_rel, history_length=5),
    "actions": ObservationTermCfg(func=mjlab_vel_mdp.last_action, history_length=5),
}

# ============================================
# Upper body Critic
# ============================================
UPPER_CRITIC_TERMS: dict[str, ObservationTermCfg] = copy.deepcopy(CRITIC_TERMS)
UPPER_CRITIC_TERMS["left_ee_commands"] = ObservationTermCfg(
    func=mjlab_vel_mdp.generated_commands,
    params={"command_name": "left_ee_cmd"},
)
UPPER_CRITIC_TERMS["right_ee_commands"] = ObservationTermCfg(
    func=mjlab_vel_mdp.generated_commands,
    params={"command_name": "right_ee_cmd"},
)
UPPER_CRITIC_TERMS["measured_ee_pose"] = ObservationTermCfg(
    func=mdp.ee_pose_b,
    params={"command_names": ("left_ee_cmd", "right_ee_cmd")},
)
UPPER_CRITIC_TERMS["ee_pose_err"] = ObservationTermCfg(
    func=mdp.ee_pose_err,
    params={"command_names": ("left_ee_cmd", "right_ee_cmd")},
)
UPPER_CRITIC_TERMS["hand_contact_force_cmd"] = ObservationTermCfg(
    func=mdp.zero_obs, params={"dim": 8},
)
UPPER_CRITIC_TERMS["hand_ext_force"] = ObservationTermCfg(
    func=mdp.zero_obs, params={"dim": 6},
)

# ============================================
# Lower body Critic
# ============================================
LOWER_CRITIC_TERMS: dict[str, ObservationTermCfg] = copy.deepcopy(CRITIC_TERMS)
LOWER_CRITIC_TERMS["step_commands"] = ObservationTermCfg(
    func=mjlab_vel_mdp.generated_commands,
    params={"command_name": "step_cmd"},
)
LOWER_CRITIC_TERMS["ref_joint_pos"] = ObservationTermCfg(
    func=mdp.ref_joint_pos,
    params={"command_name": "step_cmd", "asset_cfg": _LOWER},
)
LOWER_CRITIC_TERMS["ref_joint_vel"] = ObservationTermCfg(
    func=mdp.ref_joint_vel,
    params={"command_name": "step_cmd", "asset_cfg": _LOWER},
)
LOWER_CRITIC_TERMS["ref_feet_pos"] = ObservationTermCfg(
    func=mdp.ref_feet_pos_b,
    params={"command_name": "step_cmd"},
)
LOWER_CRITIC_TERMS["feet_pos"] = ObservationTermCfg(
    func=mdp.feet_pos_b,
    params={"command_name": "step_cmd"},
)
LOWER_CRITIC_TERMS["foot_contact"] = ObservationTermCfg(
    func=mjlab_vel_mdp.foot_contact,
    params={"sensor_name": "feet_ground_contact"},
)
LOWER_CRITIC_TERMS["foot_contact_forces"] = ObservationTermCfg(
    func=mjlab_vel_mdp.foot_contact_forces,
    params={"sensor_name": "feet_ground_contact"},
)
LOWER_CRITIC_TERMS["hand_contact_force_cmd"] = ObservationTermCfg(
    func=mdp.zero_obs, params={"dim": 8},   # reserved (see upper critic note)
)
LOWER_CRITIC_TERMS["hand_ext_force"] = ObservationTermCfg(
    func=mdp.zero_obs, params={"dim": 6},
)
# reference + measured link states of the reward's tracked links, base frame
CRITIC_BODIES = (
    "pelvis", "torso_link",
    "left_hip_yaw_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_yaw_link", "right_knee_link", "right_ankle_roll_link",
)
LOWER_CRITIC_TERMS["ref_body_state"] = ObservationTermCfg(
    func=mdp.ref_body_state_b,
    params={"command_name": "step_cmd", "body_names": CRITIC_BODIES},
)
LOWER_CRITIC_TERMS["body_state"] = ObservationTermCfg(
    func=mdp.body_state_b,
    params={"command_name": "step_cmd", "body_names": CRITIC_BODIES},
)
LOWER_CRITIC_TERMS["clf_state"] = ObservationTermCfg(
    func=mdp.clf_state,
    params={"command_name": "step_cmd"},
)


OBSERVATIONS: dict[str, ObservationGroupCfg] = {
    "upper_body_actor": ObservationGroupCfg(
        terms=UPPER_ACTOR_TERMS,
        concatenate_terms=True,
        enable_corruption=True,
    ),
    "upper_body_critic": ObservationGroupCfg(
        terms=UPPER_CRITIC_TERMS,
        concatenate_terms=True,
        enable_corruption=False,
    ),
    "lower_body_actor": ObservationGroupCfg(
        terms=LOWER_ACTOR_TERMS,
        concatenate_terms=True,
        enable_corruption=True,
    ),
    "lower_body_critic": ObservationGroupCfg(
        terms=LOWER_CRITIC_TERMS,
        concatenate_terms=True,
        enable_corruption=False,
    ),
}

OBS_CLIP: tuple[float, float] = (-100.0, 100.0)

for _group in OBSERVATIONS.values():
    for _term in _group.terms.values():
        if _term.clip is None:
            _term.clip = OBS_CLIP

# foot_croco dynamics references, critic-only (the actor never sees them).
# The -Croco task appends these to the lower critic AFTER the base terms, so
# the base critic layout is a prefix of the croco one.
CROCO_CRITIC_TERMS: dict[str, ObservationTermCfg] = {
    "ref_joint_torque": ObservationTermCfg(
        func=mdp.ref_joint_torque,
        params={"command_name": "step_cmd"},
        clip=OBS_CLIP,
    ),
    "ref_feet_grf": ObservationTermCfg(
        func=mdp.ref_feet_grf,
        params={"command_name": "step_cmd"},
        clip=OBS_CLIP,
    ),
}

# ============================================================
# Force Obs
# ============================================================
FORCE_OBSERVATIONS = copy.deepcopy(OBSERVATIONS)

_EE = ("left_ee_cmd", "right_ee_cmd")

for _group in FORCE_OBSERVATIONS.values():
    _terms = _group.terms
    if "hand_contact_force_cmd" in _terms:
        _terms["hand_contact_force_cmd"].func = mdp.hand_contact_force_command
        _terms["hand_contact_force_cmd"].params = {"command_names": _EE}
    if "hand_ext_force" in _terms:
        _terms["hand_ext_force"].func = mdp.hand_ext_force_meas
        _terms["hand_ext_force"].params = {"command_names": _EE}