import copy
import math

from mjlab.managers.command_manager import CommandTermCfg

from rl.decoupled.mdp import (
    EeTargetCommandCfg,
    UniformVelocityCommandCfg,
    TrackingMetricsVelocityCommandCfg,
    GaitPhaseCommandCfg,
    LocoMPCCommandCfg,
    HLIPCommandTermCfg,
    LocoHlipMPCCommandCfg,
    EeSpringContactCommandCfg,
    LocoHlipManipMPCCommandCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from rl.g1_assets.g1_constants import (
    G1_KNEES_BENT_INERTIA,
    G1_MASS,
    HIP_WIDTH,
    PELVIS_KNEES_BENT_Z,
    FEET_Y_DIST
)

Y_NOM = FEET_Y_DIST
"""Nominal lateral foot offset y_nom = the robot's standing stance width, m."""

FOOT_Y_ABS_RANGE = (0.10, 0.50)
"""Clamp on the lateral distance of the landing target from the stance foot, m."""

GAIT_PERIOD = 0.8
"""Full cycle T_cycle (two touchdowns), seconds."""

DUTY_FACTOR = 0.5
"""Per-foot stance fraction of the cycle. 0.5 = single-domain SSP->SSP, no
double support."""

GAIT_THRESHOLD = math.cos(math.pi * DUTY_FACTOR)
"""Sine threshold DERIVED from DUTY_FACTOR — must never be tuned on its own.

Both the MPC schedule (`cd_mpc.contact_schedule.make_walking_schedule`) and the
gait reward (`mdp.mpc_feet_gait`) decide stance with `sin(phi + 2*pi*offset) <
threshold`; the schedule computes it from the duty factor, so the reward has to
as well. Themis's hard-coded 0.5 is a 2/3 duty (33 % double support) and
anti-phases the reward against the MPC's own per-foot schedule for a third of
every cycle."""

FOOT_OFFSETS = [0.5, 0.0]
"""Per-foot phase offsets in SENSOR COLUMN order. The G1 declares the LEFT leg
first, so `feet_ground_contact` columns are [left, right] — the OPPOSITE of
Themis. The MPC pins LF stance to sin(phi+pi) < thr, i.e. the left foot is the
offset-0.5 foot. Themis's [0.0, 0.5] anti-phases reward against schedule
(measured 2026-07-21: 17 % agreement as trained, 83 % with the feet swapped)."""

EE_REGION = dict(
    x_margin=0.1,
    y_reach=0.3,
    y_margin=0.05,
    z_max=0.3,
    z_min=0.1,
)

NOMINAL_TIP = dict(
    left=dict(nominal_pos_b=(0.3323, 0.1516, 0.0952),
              nominal_quat_b=(1.0, 0.0, 0.0, 0.0)),
    right=dict(nominal_pos_b=(0.3323, -0.1516, 0.0952),
               nominal_quat_b=(1.0, 0.0, 0.0, 0.0)),
)

EE_TRANSITION = dict(
    transition_lin_speed=(0.15, 0.40),   
    transition_ang_speed=(0.5, 1.2),    
    transition_time_range=(0.4, 2.5),    
    jump_prob=0.1,         
)

COMMANDS: dict[str, CommandTermCfg] = {
    "left_ee_cmd": EeTargetCommandCfg(
        resampling_time_range=(3.0, 6.0),
        debug_vis=False,  
        entity_name="robot",
        body_name="left_tactile",
        use_yaw_align_frame=False,  
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
        use_yaw_align_frame=False,
        ee_frames=("left_tactile", "right_tactile"),
        **EE_REGION,
        **EE_TRANSITION,
        **NOMINAL_TIP["right"],
    ),
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
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
        viz=UniformVelocityCommandCfg.VizCfg(z_offset=1.15, scale=0.5),
        metric_std_lin=math.sqrt(0.25),
        metric_std_ang=math.sqrt(0.5),
    ),
    "gait": GaitPhaseCommandCfg(
        period=GAIT_PERIOD,
        command_name="twist",
        command_threshold=0.1,
        stance_decay=0.95
    ),
    "foot_plan": HLIPCommandTermCfg(
        debug_vis=False,  # play assembly turns this on
        entity_name="robot",
        twist_name="twist",
        gait_name="gait",
        com_source="pelvis",
        com_height_ref=PELVIS_KNEES_BENT_Z,
        stance_width=Y_NOM,
        feedback_gain=0.0,
        perturb_xy=(0.0, 0.0),
        perturb_yaw=0.0,
        landing_x_range=(-0.35, 0.35),
        landing_dy_range=(FOOT_Y_ABS_RANGE[0] - Y_NOM, FOOT_Y_ABS_RANGE[1] - Y_NOM),
        landing_yaw_range=(-0.30, 0.30),
    ),
    "loco_mpc": LocoHlipMPCCommandCfg(
        debug_vis=False,  # play assembly turns this on
        asset_cfg=SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
        planner_name="foot_plan",
        mpc_dt=0.07,
        mpc_horizon=10,
        mass=G1_MASS,
        hip_width=HIP_WIDTH,
        com_height=PELVIS_KNEES_BENT_Z,
        body_inertia=G1_KNEES_BENT_INERTIA,
        gait_period=GAIT_PERIOD,
        duty_factor=DUTY_FACTOR,
        vel_cmd_name="twist",
        grf_sensor_name="feet_ground_contact",
        run_every_n_steps=5,
        resampling_time_range=(1e6, 1e6),
        mpc_impl="cd",
    ),
}


# ── VForce variant: the virtual-spring force layer on the same chain ────────
_CONTACT = dict(
    contact_prob=0.4,
    phantom_prob=0.10,
    contact_resampling_time_range=(3.0, 6.0),
    force_range=(5.0, 25.0),
    force_ramp_time=0.6,
    k_ext=1000.0,
    d_ext=30.0,
)

FORCE_COMMANDS: dict[str, CommandTermCfg] = copy.deepcopy(COMMANDS)
FORCE_COMMANDS["left_ee_cmd"] = EeSpringContactCommandCfg(
    resampling_time_range=(3.0, 6.0),
    debug_vis=False,  # play assembly turns the goal frames on
    entity_name="robot",
    body_name="left_tactile",
    use_yaw_align_frame=False,
    ee_frames=("left_tactile", "right_tactile"),
    **EE_REGION,
    **EE_TRANSITION,
    **NOMINAL_TIP["left"],
    **_CONTACT,
)
FORCE_COMMANDS["right_ee_cmd"] = EeSpringContactCommandCfg(
    resampling_time_range=(3.0, 6.0),
    debug_vis=False,
    entity_name="robot",
    body_name="right_tactile",
    use_yaw_align_frame=False,
    ee_frames=("left_tactile", "right_tactile"),
    **EE_REGION,
    **EE_TRANSITION,
    **NOMINAL_TIP["right"],
    **_CONTACT,
)
FORCE_COMMANDS["loco_mpc"] = LocoHlipManipMPCCommandCfg(
    debug_vis=False,  # play assembly turns this on
    asset_cfg=SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    planner_name="foot_plan",
    mpc_dt=0.07,
    mpc_horizon=10,
    mass=G1_MASS,
    hip_width=HIP_WIDTH,
    com_height=PELVIS_KNEES_BENT_Z,
    body_inertia=G1_KNEES_BENT_INERTIA,
    gait_period=GAIT_PERIOD,
    duty_factor=DUTY_FACTOR,
    vel_cmd_name="twist",
    grf_sensor_name="feet_ground_contact",
    run_every_n_steps=5,
    resampling_time_range=(1e6, 1e6),
    mpc_impl="cd",
    ee_command_names=("left_ee_cmd", "right_ee_cmd"),
    gate_mode="command",  # no contact sensors in the spring task
)
