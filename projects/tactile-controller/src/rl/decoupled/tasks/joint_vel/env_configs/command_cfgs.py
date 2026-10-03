"""Command declarations — the step-clip command (one clip per episode).

Resampling happens only on episode reset (the in-episode timer is effectively
infinite); after the clip's last frame the references hold the final standing
pose until timeout — the stand-and-wait half of the start-stop regime.
"""

import math

from mjlab.managers.command_manager import CommandTermCfg

from rl.decoupled.mdp import (
    EeTargetCommandCfg,
    UniformVelocityCommandCfg,
)

EE_REGION = dict(
    x_margin=0.1,
    y_reach=0.3,
    y_margin=0.05,
    z_max=0.3,
    z_min=0.1,
)

# Nominal tip pose (blend-curriculum anchor), robot frame — measured at the
# HOME-keyframe standing pose. The blend pulls toward this CONSTANT, never
# toward the measured hand (2026-08-06 bug: measured-blend made targets follow
# a tangled arm onto the wrong side of the body and kept paying it there).
NOMINAL_TIP = dict(
    left=dict(nominal_pos_b=(0.3323, 0.1516, 0.0952),
              nominal_quat_b=(1.0, 0.0, 0.0, 0.0)),
    right=dict(nominal_pos_b=(0.3323, -0.1516, 0.0952),
               nominal_quat_b=(1.0, 0.0, 0.0, 0.0)),
)

# Transition governor: the reference GLIDES to each pool draw instead of
# jumping (smoothness over urgency — the arm-slam-on-resample fix). Mean-speed
# caps are the real knobs (min-jerk peak = 1.875x); the resample window (3, 6)
# sits above the 2.5 s duration ceiling so EVERY goal ends with a hold phase
# where the accuracy kernels do their work.
EE_TRANSITION = dict(
    transition_lin_speed=(0.15, 0.40),   # m/s mean → peak 0.28..0.75
    transition_ang_speed=(0.5, 1.2),     # rad/s mean
    transition_time_range=(0.4, 2.5),    # s
    jump_prob=0.1,                       # keep step commands in-distribution
)

COMMANDS: dict[str, CommandTermCfg] = {
    "left_ee_cmd": EeTargetCommandCfg(
        resampling_time_range=(3.0, 6.0),
        debug_vis=False,  # play assembly turns the goal frames on
        entity_name="robot",
        body_name="left_tactile",
        use_yaw_align_frame=False,   # paper-faithful ROOT body frame (REFINE-DP/HERO)
        ee_frames=("left_tactile", "right_tactile"),
        **EE_REGION,
        **EE_TRANSITION,
        **NOMINAL_TIP["left"],
    ),
    "right_ee_cmd": EeTargetCommandCfg(
        resampling_time_range=(3.0, 6.0),
        debug_vis=False,
        entity_name="robot",
        body_name="right_tactile",
        use_yaw_align_frame=False,   # paper-faithful ROOT body frame (REFINE-DP/HERO)
        ee_frames=("left_tactile", "right_tactile"),
        **EE_REGION,
        **EE_TRANSITION,
        **NOMINAL_TIP["right"],
    ),
    "twist": UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.05,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
    )
}

