import copy

from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pathlib import Path

from rl.decoupled.mdp import (
    FootHoldCommandCfg,
    FootholdMPCCommandCfg,
    FootholdManipMPCCommandCfg,
    EeTargetCommandCfg,
    EeSpringContactCommandCfg,
    PosturePriorCfg,
)
from rl.g1_assets.g1_constants import (
    G1_KNEES_BENT_INERTIA,
    G1_MASS,
    HIP_WIDTH,
    PELVIS_KNEES_BENT_Z,
)
GAIT_PERIOD = 0.8
"""Full cycle T_cycle (two touchdowns), seconds."""

DUTY_FACTOR = 0.5
"""Per-foot stance fraction of the cycle. 0.5 = single-domain SSP->SSP, no
double support."""

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
    "foothold": FootHoldCommandCfg(
        entity_name="robot",
        debug_vis=False,  # play assembly turns the target arrows on
        sampler=FootHoldCommandCfg.Sampler(
            x_bins=((-0.2, -0.08), (-0.08, 0.08), (0.08, 0.2), (0.2, 0.25)),
            x_probs=(0.20, 0.20, 0.45, 0.15),
            jitter_x=(-0.04, 0.04),
            x_clamp=(-0.2, 0.25),
        ),
        timing=FootHoldCommandCfg.Timing(step_period=0.5),
    ),
    "loco_mpc": FootholdMPCCommandCfg(
        debug_vis=False,  # play assembly turns this on
        asset_cfg=SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
        mpc_dt=0.07,
        mpc_horizon=10,
        mass=G1_MASS,
        hip_width=HIP_WIDTH,
        com_height=PELVIS_KNEES_BENT_Z,      # KNEES_BENT: real grounded pelvis z
        body_inertia=G1_KNEES_BENT_INERTIA,  # KNEES_BENT: I about the true CoM
        gait_period=GAIT_PERIOD,
        duty_factor=DUTY_FACTOR,
        grf_sensor_name="feet_ground_contact",
        run_every_n_steps=5,  # 10 Hz (policy = 50 Hz -> every 5 steps)
        resampling_time_range=(1e6, 1e6),
        mpc_impl="cd",
        foothold_name="foothold",
    ),
}


# ============================================================
# Posture prior (the -Prior task): orbit clips on THIS clock
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[6]
POSTURE_PRIOR = PosturePriorCfg(
    dataset_path=str(REPO_ROOT / "data" / "foot_lip_orbit_v1"),
    yaw_weight=0.2,
    windup_time=0.30,   # = FootholdMPCCommandCfg.windup_time
)


# ============================================================
# Force Command
# ============================================================
_CONTACT = dict(
    contact_prob=0.4,
    phantom_prob=0.10,
    contact_resampling_time_range=(3.0, 6.0),
    force_range=(5.0, 25.0),
    force_ramp_time=0.6,
    k_ext=1000.0,
    d_ext=30.0,  # k/d sweep 2026-08-11: d=30 dominates d=10 at k=1000
)

FORCE_COMMANDS: dict[str, CommandTermCfg] = {
    "left_ee_cmd": EeSpringContactCommandCfg(
        resampling_time_range=(3.0, 6.0),
        debug_vis=False,  # play assembly turns the goal frames on
        entity_name="robot",
        body_name="left_tactile",
        use_yaw_align_frame=False,   # paper-faithful ROOT body frame (REFINE-DP/HERO)
        ee_frames=("left_tactile", "right_tactile"),
        # nominal_blend=1.0,  # the nominal tip is the spring's anchor (no blend)
        **EE_REGION,
        **EE_TRANSITION,
        **NOMINAL_TIP["left"],
        **_CONTACT,
    ),
    "right_ee_cmd": EeSpringContactCommandCfg(
        resampling_time_range=(3.0, 6.0),
        debug_vis=False,
        entity_name="robot",
        body_name="right_tactile",
        use_yaw_align_frame=False,   # paper-faithful ROOT body frame (REFINE-DP/HERO)
        ee_frames=("left_tactile", "right_tactile"),
        # nominal_blend=1.0,  # the nominal tip is the spring's anchor (no blend)
        **EE_REGION,
        **EE_TRANSITION,
        **NOMINAL_TIP["right"],
        **_CONTACT,
    ),
    "foothold": FootHoldCommandCfg(
        entity_name="robot",
        debug_vis=False,  # play assembly turns the target arrows on
        sampler=FootHoldCommandCfg.Sampler(
            x_bins=((-0.2, -0.08), (-0.08, 0.08), (0.08, 0.2), (0.2, 0.25)),
            x_probs=(0.20, 0.20, 0.45, 0.15),
            jitter_x=(-0.04, 0.04),
            x_clamp=(-0.2, 0.25),
        ),
        timing=FootHoldCommandCfg.Timing(step_period=0.5),
    ),
    "loco_mpc": FootholdManipMPCCommandCfg(
        debug_vis=False,  # play assembly turns this on
        asset_cfg=SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
        mpc_dt=0.07,
        mpc_horizon=10,
        mass=G1_MASS,
        hip_width=HIP_WIDTH,
        com_height=PELVIS_KNEES_BENT_Z,      # KNEES_BENT: real grounded pelvis z
        body_inertia=G1_KNEES_BENT_INERTIA,  # KNEES_BENT: I about the true CoM
        gait_period=GAIT_PERIOD,
        duty_factor=DUTY_FACTOR,
        grf_sensor_name="feet_ground_contact",
        run_every_n_steps=5,  # 10 Hz (policy = 50 Hz -> every 5 steps)
        resampling_time_range=(1e6, 1e6),
        mpc_impl="cd",
        foothold_name="foothold",
        ee_command_names=("left_ee_cmd", "right_ee_cmd"),
        gate_mode="command",  # no contact sensors in the spring task
    ),
}