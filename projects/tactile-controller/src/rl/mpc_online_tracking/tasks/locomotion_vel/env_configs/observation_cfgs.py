"""Observation declarations.

TERM ORDER IS THE CHECKPOINT OBS LAYOUT — do not reorder existing terms; append
only. Actor is deployable proprio (no privileged base_lin_vel); critic adds
privileged foot/contact channels and the MPC plan references.
"""

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from .. import mdp
from .command_cfgs import GAIT_PERIOD

ACTOR_TERMS: dict[str, ObservationTermCfg] = {
    "base_ang_vel": ObservationTermCfg(
        func=mjlab_vel_mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
        func=mjlab_vel_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
        func=mjlab_vel_mdp.joint_pos_rel,
        params={"biased": True},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
        func=mjlab_vel_mdp.joint_vel_rel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mjlab_vel_mdp.last_action),
    "command": ObservationTermCfg(
        func=mjlab_vel_mdp.generated_commands,
        params={"command_name": "twist"},
    ),
    # Gait clock; freezes to stance phase when standing.
    "phase": ObservationTermCfg(
        func=mdp.phase,
        params={
            "period": GAIT_PERIOD,
            "command_name": "twist",
            "command_threshold": 0.1,
            "stance_decay": 0.95,
        },
    ),
}

CRITIC_TERMS: dict[str, ObservationTermCfg] = {
    "base_lin_vel": ObservationTermCfg(
        func=mjlab_vel_mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_vel"},
        noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
        func=mjlab_vel_mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
        func=mjlab_vel_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    # True (unbiased) joint positions as privileged information.
    "joint_pos": ObservationTermCfg(func=mjlab_vel_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(
        func=mjlab_vel_mdp.joint_vel_rel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mjlab_vel_mdp.last_action),
    "command": ObservationTermCfg(
        func=mjlab_vel_mdp.generated_commands,
        params={"command_name": "twist"},
    ),
    "foot_height": ObservationTermCfg(
        func=mjlab_vel_mdp.foot_height,
        params={"sensor_name": "foot_height_scan"},
    ),
    "foot_air_time": ObservationTermCfg(
        func=mjlab_vel_mdp.foot_air_time,
        params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
        func=mjlab_vel_mdp.foot_contact,
        params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
        func=mjlab_vel_mdp.foot_contact_forces,
        params={"sensor_name": "feet_ground_contact"},
    ),
    "phase": ObservationTermCfg(
        func=mdp.phase_readout,
        params={"period": GAIT_PERIOD, "command_name": "twist"},
    ),
    # MPC plan references (critic-only).
    "mpc_com_ref": ObservationTermCfg(
        func=mdp.mpc_com_ref,
        params={"command_name": "loco_mpc"},
    ),
    "mpc_k_ref": ObservationTermCfg(
        func=mdp.mpc_ang_mom_ref,
        params={"command_name": "loco_mpc"},
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
