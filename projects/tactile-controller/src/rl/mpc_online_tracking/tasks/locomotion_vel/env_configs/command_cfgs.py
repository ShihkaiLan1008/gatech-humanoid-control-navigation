"""Command declarations — sampled twist + the parallel centroidal MPC.

Shared robot-scale constants are declared here (imported by the other manager
files and by the derived tasks): gait period, mass/geometry, foot sites.
"""

import math

from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from ..mdp import LocoMPCCommandCfg, TrackingMetricsVelocityCommandCfg

# ── G1 robot-scale constants (Themis re-target) ─────────────────────────────
G1_MASS = 33.34  # mj_getTotalmass on g1.xml
G1_HIP_WIDTH = 0.12  # ankle_roll y offset is ±0.119 m
G1_COM_HEIGHT = 0.76  # knees-bent init height
# Gait clock period shared by observation, reward, and the MPC contact
# schedule (seconds). 0.8 s for the G1 (Themis used 0.9 s at 1.17 m stance).
GAIT_PERIOD = 0.8
FOOT_SITE_NAMES = ("left_foot", "right_foot")

COMMANDS: dict[str, CommandTermCfg] = {
    # Sampled body-frame twist (the policy's command obs) + exp-kernel
    # tracking-score metrics.
    "twist": TrackingMetricsVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.3,
        rel_forward_envs=0.2,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
        viz=UniformVelocityCommandCfg.VizCfg(z_offset=1.15, scale=0.5),
        metric_std_lin=math.sqrt(0.25),
        metric_std_ang=math.sqrt(0.5),
    ),
    # Batched centroidal MPC running in parallel: consumes the twist, plans
    # CoM / angular momentum / GRF / foot placements that shape the rewards.
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
        run_every_n_steps=5,  # 10 Hz (policy = 50 Hz → every 5 steps)
        resampling_time_range=(1e6, 1e6),
        solver_type="jax_pimpc",
    ),
}
