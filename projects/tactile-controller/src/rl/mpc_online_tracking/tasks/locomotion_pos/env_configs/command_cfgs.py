"""Command declarations — SE(2) pose goal in the "twist" slot + centroidal MPC.

The ONLY manager that differs from locomotion_vel: sampled twists are replaced
by :class:`Se2PoseVelocityCommand` (world-latched pelvis SE(2) target →
goal-attractor twist). Its ``ranges`` are the attractor's velocity CLIP box —
keep them equal to the vel-follower checkpoint's training box so the emitted
twist stays in-distribution (play widens them like locomotion_vel's play).
"""

import math

from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from ...locomotion_vel.env_configs.command_cfgs import (  # noqa: F401
    FOOT_SITE_NAMES,
    G1_COM_HEIGHT,
    G1_HIP_WIDTH,
    G1_MASS,
    GAIT_PERIOD,
)
from ..mdp import LocoMPCCommandCfg, Se2PoseVelocityCommandCfg

COMMANDS: dict[str, CommandTermCfg] = {
    # World-latched SE(2) pelvis target → goal-attractor body-frame twist.
    # The policy stays a goal-blind vel-follower (obs = the emitted v_des), so
    # the emitted distribution must cover the checkpoint's training
    # distribution — see the twist/speed knobs below.
    "twist": Se2PoseVelocityCommandCfg(
        entity_name="robot",
        # Mixed cadence: the 3 s end trains robustness to fast goal changes
        # (the policy saw 3-8 s step twists originally); arrival triggers an
        # early resample anyway (arrive_dwell_range).
        resampling_time_range=(3.0, 12.0),
        debug_vis=False,  # play assembly turns the goal frame/markers on
        # Clip limits AND twist-env sampling box = the distribution the
        # vel-follower checkpoint actually TRAINED under (the command_vel
        # curriculum stage of locomotion_vel), NOT the play-widened box —
        # anything wider is OOD for the policy. heading must stay None: the
        # base class rejects heading ranges when heading_command=False (this
        # term replaces the heading machinery).
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.75, 0.75),
            ang_vel_z=(-1.0, 1.0),
            heading=None,
        ),
        radius_range=(1.0, 2.5),
        bearing_range=(-math.pi, math.pi),
        rel_yaw_range=(-math.pi, math.pi),
        decel=0.5,
        slew=2.0,
        yaw_stiffness=1.5,
        pos_tol=0.15,
        yaw_tol=0.20,
        arrive_dwell_range=(1.0, 2.0),
        # Coverage guarantee: this fraction of envs gets plain sampled twists
        # (the original training distribution) — the attractor alone rarely
        # sustains fast+max-turn arcs or long pure-lateral/backward legs.
        rel_twist_envs=0.3,
        # DR over approach speed (deployment planners may run gentler
        # profiles); play/deployment resets this to (1.0, 1.0).
        speed_scale_range=(0.5, 1.0),
    ),
    # Same parallel centroidal MPC as locomotion_vel (jax[cuda12] installed
    # 2026-07-25 — the earlier torch-pimpc fallback for CPU-only jax is gone).
    "loco_mpc": LocoMPCCommandCfg(
        debug_vis=False,  # play assembly turns this on
        asset_cfg=SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
        mpc_dt=0.07,
        mpc_horizon=10,
        mass=G1_MASS,
        hip_width=G1_HIP_WIDTH,
        com_height=G1_COM_HEIGHT,
        gait_period=GAIT_PERIOD,
        duty_factor=0.5,
        vel_cmd_name="twist",
        grf_sensor_name="feet_ground_contact",
        run_every_n_steps=5,
        resampling_time_range=(1e6, 1e6),
        solver_type="jax_pimpc",
    ),
}
