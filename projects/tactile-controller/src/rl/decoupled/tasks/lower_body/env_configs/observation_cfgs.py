"""Observation declarations — REFINE-DP Table I (locomotion block).

TERM ORDER IS THE CHECKPOINT OBS LAYOUT — never reorder existing terms, append
only. Actor = deployable proprio, single frame + previous action, NO base
linear velocity (privileged); critic adds base lin vel, reference/current feet
positions and the foot contact wrench — measured contact force stays OUT of
the actor per the standing user rule.
"""

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from rl.decoupled import mdp
from rl.decoupled.mdp import LOWER_JOINT_NAMES

_LOWER = SceneEntityCfg("robot", joint_names=LOWER_JOINT_NAMES)

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
        params={"asset_cfg": _LOWER, "biased": True},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        history_length=5,
    ),
    "joint_vel": ObservationTermCfg(
        func=mjlab_vel_mdp.joint_vel_rel,
        params={"asset_cfg": _LOWER},
        noise=Unoise(n_min=-1.5, n_max=1.5),
        history_length=5,
    ),
    "actions": ObservationTermCfg(func=mjlab_vel_mdp.last_action, history_length=5),
    "command": ObservationTermCfg(
        func=mjlab_vel_mdp.generated_commands,
        params={"command_name": "step_cmd"},
    ),
}

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
    # Critic sees ALL 29 joints unbiased (incl. the PD-held arms — privileged).
    "joint_pos": ObservationTermCfg(func=mjlab_vel_mdp.joint_pos_rel, history_length=5),
    "joint_vel": ObservationTermCfg(func=mjlab_vel_mdp.joint_vel_rel, history_length=5),
    "actions": ObservationTermCfg(func=mjlab_vel_mdp.last_action, history_length=5),
    "command": ObservationTermCfg(
        func=mjlab_vel_mdp.generated_commands,
        params={"command_name": "step_cmd"},
    ),
    "ref_joint_pos": ObservationTermCfg(
        func=mdp.ref_joint_pos,
        params={"command_name": "step_cmd", "asset_cfg": _LOWER},
    ),
    "ref_joint_vel": ObservationTermCfg(
        func=mdp.ref_joint_vel,
        params={"command_name": "step_cmd", "asset_cfg": _LOWER},
    ),
    "ref_joint_torques": ObservationTermCfg(
        func=mdp.ref_joint_torques,
        params={"command_name": "step_cmd", "asset_cfg": _LOWER},
    ),
    "ref_feet_pos": ObservationTermCfg(
        func=mdp.ref_feet_pos_b,
        params={"command_name": "step_cmd"},
    ),
    "feet_pos": ObservationTermCfg(
        func=mdp.feet_pos_b,
        params={"command_name": "step_cmd"},
    ),
    "foot_contact": ObservationTermCfg(
        func=mjlab_vel_mdp.foot_contact,
        params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
        func=mjlab_vel_mdp.foot_contact_forces,
        params={"sensor_name": "feet_ground_contact"},
    ),
}

OBSERVATIONS: dict[str, ObservationGroupCfg] = {
    "actor": ObservationGroupCfg(
        terms=ACTOR_TERMS,
        concatenate_terms=True,
        enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
        terms=CRITIC_TERMS,
        concatenate_terms=True,
        enable_corruption=False,
    ),
}

# Uniform safety clip on EVERY observation, paired with `clip_actions` in
# rl_cfgs.py. Run 2026-08-06_04-58-50 died of an unbounded feedback loop: the
# raw action is itself an observation (the `actions` term), and
# nothing in the stack bounded either end (no obs clip, no action clip, no obs
# normalization), so a saturating action re-entered the network and grew — a
# self-sustaining fixed point measured at |a| 37 typical / 8.4e8 worst, which
# the quadratic action_rate penalty turned into 1e13 returns, a 1e29 value
# loss and NaN weights. +-100 sits far outside every legitimate obs/action
# range, so it is inert in normal operation and only caps the runaway.
OBS_CLIP: tuple[float, float] = (-100.0, 100.0)

for _group in OBSERVATIONS.values():
    for _term in _group.terms.values():
        if _term.clip is None:
            _term.clip = OBS_CLIP
