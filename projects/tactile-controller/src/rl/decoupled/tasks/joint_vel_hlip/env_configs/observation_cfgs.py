"""Observations — robot_rl LIP-CLF policy/critic sets for the lower body
(single frame; the actor never sees the reference), joint_vel's upper sets
unchanged. TERM ORDER IS THE CHECKPOINT OBS LAYOUT — append only."""

import copy

from mjlab.managers import SceneEntityCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from rl.decoupled import mdp
from rl.decoupled.tasks.joint_vel.env_configs.observation_cfgs import (
    UPPER_ACTOR_TERMS as _UPPER_ACTOR,
    UPPER_CRITIC_TERMS as _UPPER_CRITIC,
)
from rl.decoupled.tasks.joint_vel_hlip import mdp as hlip_mdp

_FEET = SceneEntityCfg("robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link"))

LOWER_ACTOR_TERMS: dict[str, ObservationTermCfg] = {
    "base_ang_vel": ObservationTermCfg(
        func=mjlab_vel_mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
        func=mjlab_vel_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "vel_cmd": ObservationTermCfg(
        func=mjlab_vel_mdp.generated_commands,
        params={"command_name": "twist"},
        scale=2.0,
    ),
    "joint_pos": ObservationTermCfg(
        func=mjlab_vel_mdp.joint_pos_rel,
        params={"biased": True},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
        func=mjlab_vel_mdp.joint_vel_rel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        scale=0.05,
    ),
    "actions": ObservationTermCfg(func=mjlab_vel_mdp.last_action),
    "phase": ObservationTermCfg(
        func=mjlab_vel_mdp.generated_commands,
        params={"command_name": "gait"},
    ),
}

LOWER_CRITIC_TERMS: dict[str, ObservationTermCfg] = {
    "base_lin_vel": ObservationTermCfg(
        func=mjlab_vel_mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "base_ang_vel": ObservationTermCfg(
        func=mjlab_vel_mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "projected_gravity": ObservationTermCfg(func=mjlab_vel_mdp.projected_gravity),
    "vel_cmd": ObservationTermCfg(
        func=mjlab_vel_mdp.generated_commands,
        params={"command_name": "twist"},
        scale=2.0,
    ),
    "joint_pos": ObservationTermCfg(func=mjlab_vel_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mjlab_vel_mdp.joint_vel_rel, scale=0.05),
    "actions": ObservationTermCfg(func=mjlab_vel_mdp.last_action),
    "phase": ObservationTermCfg(
        func=mjlab_vel_mdp.generated_commands,
        params={"command_name": "gait"},
    ),
    "foot_contact": ObservationTermCfg(
        func=mjlab_vel_mdp.foot_contact,
        params={"sensor_name": "feet_ground_contact"},
    ),
    "feet_lin_vel": ObservationTermCfg(func=hlip_mdp.feet_lin_vel_w, params={"asset_cfg": _FEET}),
    "feet_ang_vel": ObservationTermCfg(func=hlip_mdp.feet_ang_vel_w, params={"asset_cfg": _FEET}),
    "hlip_ref_traj": ObservationTermCfg(func=hlip_mdp.hlip_ref_traj, params={"command_name": "hlip_ref"}),
    "hlip_act_traj": ObservationTermCfg(func=hlip_mdp.hlip_act_traj, params={"command_name": "hlip_ref"}),
    "hlip_ref_vel": ObservationTermCfg(
        func=hlip_mdp.hlip_ref_vel, params={"command_name": "hlip_ref"}, clip=(-20.0, 20.0)),
    "hlip_act_vel": ObservationTermCfg(
        func=hlip_mdp.hlip_act_vel, params={"command_name": "hlip_ref"}, clip=(-20.0, 20.0)),
    "clf_state": ObservationTermCfg(func=mdp.clf_state, params={"command_name": "hlip_ref"}),
}

OBSERVATIONS: dict[str, ObservationGroupCfg] = {
    "upper_body_actor": ObservationGroupCfg(
        terms=copy.deepcopy(_UPPER_ACTOR), concatenate_terms=True, enable_corruption=True),
    "upper_body_critic": ObservationGroupCfg(
        terms=copy.deepcopy(_UPPER_CRITIC), concatenate_terms=True, enable_corruption=False),
    "lower_body_actor": ObservationGroupCfg(
        terms=LOWER_ACTOR_TERMS, concatenate_terms=True, enable_corruption=True),
    "lower_body_critic": ObservationGroupCfg(
        terms=LOWER_CRITIC_TERMS, concatenate_terms=True, enable_corruption=False),
}

OBS_CLIP: tuple[float, float] = (-100.0, 100.0)

for _group in OBSERVATIONS.values():
    for _term in _group.terms.values():
        if _term.clip is None:
            _term.clip = OBS_CLIP
