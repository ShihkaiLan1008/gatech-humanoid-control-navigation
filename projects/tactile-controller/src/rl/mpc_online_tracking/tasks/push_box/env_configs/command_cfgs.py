"""Command declarations — mode-gated twist + loco-manipulation MPC.

Differences vs locomotion_vel: the twist is mode-gated (walk-only envs get a
sampled twist, push envs a forward push command), and the MPC is the
loco-manipulation variant that also solves for hand push forces.
"""

import math

from mjlab.managers.command_manager import CommandTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from ...locomotion_vel.env_configs.command_cfgs import (  # noqa: F401
    FOOT_SITE_NAMES,
    G1_COM_HEIGHT,
    G1_HIP_WIDTH,
    G1_MASS,
    GAIT_PERIOD,
)
from ..mdp import LocoManipMPCCommandCfg, ModeGatedVelocityCommandCfg

# Tight stand-still deadzone for push envs (a push command is small but must
# not read as "standing").
PUSH_CMD_THRESHOLD = 0.02

COMMANDS: dict[str, CommandTermCfg] = {
    "twist": ModeGatedVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.3,
        rel_forward_envs=0.2,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        # Full 3-D commands for the walk-only half; push envs gate internally.
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.75),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.75, 0.75),
            heading=(-math.pi, math.pi),
        ),
        viz=UniformVelocityCommandCfg.VizCfg(z_offset=1.15, scale=0.5),
    ),
    "loco_mpc": LocoManipMPCCommandCfg(
        debug_vis=False,  # play assembly turns this on
        mpc_dt=0.07,
        mpc_horizon=10,
        mass=G1_MASS,
        hip_width=G1_HIP_WIDTH,
        com_height=G1_COM_HEIGHT,
        gait_period=GAIT_PERIOD,
        run_every_n_steps=5,
        resampling_time_range=(1e6, 1e6),
        # Hand geometry: palm sites on the wrist_yaw links.
        hand_site_names=("left_palm", "right_palm"),
        lhand_box_sensor_name="lhand_box_contact",
        rhand_box_sensor_name="rhand_box_contact",
        body_box_sensor_name="body_box_contact",
        # Box physics
        box_mass=8.0,
        mu_ground=0.5,
        # Hand MPC costs
        mu_hand=0.6,
        f_hand_max=150.0,
        R_f_hand=1e-4,
        R_hand_balance=1e-3,
        solver_type="jax_pimpc",
    ),
}
