"""Stage A1 task cfg: 1-step QP + residual policy on the mjlab G1 flat env.

Base = mjlab's in-tree ``unitree_g1_flat_env_cfg``; three changes:
1. Robot: the 12 leg joints move from Builtin position actuators to IdealPd
   groups (same effort limits / armature per motor family) so the action term
   can write RAMBO's three channels (q_des, qd_des, tau_ff) and switch
   stance/swing gains at runtime. Waist yaw gets its own Builtin group (it
   shared a group with the hips upstream). Arms/waist stay Builtin position —
   A1 holds them at the default pose.
2. Action: ``OneStepQpActionCfg`` replaces the joint-position action.
3. Obs: a gait-phase term (sin/cos phase, clock active, scheduled contacts)
   is added for both groups — the policy must see the clock it is correcting.
"""

from __future__ import annotations

import torch

from mjlab.actuator import IdealPdActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    ACTUATOR_5020,
    ACTUATOR_7520_14,
    ACTUATOR_7520_22,
    FULL_COLLISION,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_5020,
    G1_ACTUATOR_WAIST,
    KNEES_BENT_KEYFRAME,
    STIFFNESS_7520_14,
    DAMPING_7520_14,
    get_spec,
)
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from .action import OneStepQpActionCfg

STANCE_KP, STANCE_KD = 40.0, 4.0


def _leg_idealpd_actuators() -> tuple[IdealPdActuatorCfg, ...]:
    return (
        IdealPdActuatorCfg(
            target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint"),
            stiffness=STANCE_KP,
            damping=STANCE_KD,
            effort_limit=ACTUATOR_7520_14.effort_limit,
            armature=ACTUATOR_7520_14.reflected_inertia,
        ),
        IdealPdActuatorCfg(
            target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
            stiffness=STANCE_KP,
            damping=STANCE_KD,
            effort_limit=ACTUATOR_7520_22.effort_limit,
            armature=ACTUATOR_7520_22.reflected_inertia,
        ),
        IdealPdActuatorCfg(
            target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
            stiffness=STANCE_KP,
            damping=STANCE_KD,
            effort_limit=ACTUATOR_5020.effort_limit * 2,
            armature=ACTUATOR_5020.reflected_inertia * 2,
        ),
    )


def get_g1_onestep_robot_cfg() -> EntityCfg:
    articulation = EntityArticulationInfoCfg(
        actuators=(
            G1_ACTUATOR_5020,  # shoulders/elbows/wrist_roll (Builtin, held)
            BuiltinPositionActuatorCfg(
                target_names_expr=("waist_yaw_joint",),
                stiffness=STIFFNESS_7520_14,
                damping=DAMPING_7520_14,
                effort_limit=ACTUATOR_7520_14.effort_limit,
                armature=ACTUATOR_7520_14.reflected_inertia,
            ),
            G1_ACTUATOR_4010,  # wrists
            G1_ACTUATOR_WAIST,  # waist pitch/roll
            *_leg_idealpd_actuators(),
        ),
        soft_joint_pos_limit_factor=0.9,
    )
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=articulation,
    )


def gait_phase_obs(env) -> torch.Tensor:
    """[B, 5]: sin/cos of the clock phase, clock-active flag, scheduled contacts."""
    term = env.action_manager.get_term("onestep")
    p = term.gait.phase / term.gait.p.cycle
    contact = term.gait.contact_state().float()
    return torch.cat(
        [
            torch.sin(2 * torch.pi * p).unsqueeze(-1),
            torch.cos(2 * torch.pi * p).unsqueeze(-1),
            term.gait.active.float().unsqueeze(-1),
            contact,
        ],
        dim=-1,
    )


def g1_onestep_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = unitree_g1_flat_env_cfg(play=play)

    cfg.scene.entities = {"robot": get_g1_onestep_robot_cfg()}
    cfg.actions = {"onestep": OneStepQpActionCfg(entity_name="robot")}

    for group in ("actor", "critic"):
        cfg.observations[group].terms["gait_phase"] = ObservationTermCfg(
            func=gait_phase_obs
        )

    # Command envelope clamped to the validated controller range (Stage 0:
    # forward ceiling ~0.45 at the fixed cadence, lateral 0.2, and every
    # reference in this architecture family trains inside +-0.5 — RAMBO
    # +-0.5/+-0.5, g1_wall fixed 0.3. The first A1 run sampled up to 1.0 m/s
    # and spent training falling at commands the gait stack cannot reach.
    # Re-widen with the freq/apex channels via curriculum, not up front.
    envelope = {
        "lin_vel_x": (-0.3, 0.45),
        "lin_vel_y": (-0.2, 0.2),
        "ang_vel_z": (-0.5, 0.5),
    }
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = envelope["lin_vel_x"]
    twist.ranges.lin_vel_y = envelope["lin_vel_y"]
    twist.ranges.ang_vel_z = envelope["ang_vel_z"]
    if cfg.curriculum and "command_vel" in cfg.curriculum:
        cfg.curriculum["command_vel"].params["velocity_stages"] = [
            {"step": 0, **envelope}
        ]

    return cfg
