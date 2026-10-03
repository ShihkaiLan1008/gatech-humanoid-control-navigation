"""Command declarations — joint_vel_mpc's commands + the VIRTUAL-SPRING force
layer (no press cubes, no contact geometry).

``left/right_ee_cmd`` become :class:`EeSpringContactCommandCfg` (same
pool/region/transition parameters as the base task; force realized by the
unilateral spring on the goal) and ``loco_mpc`` is the loco-manip QP gated on
the COMMAND state (no contact sensors exist here). ``twist``/``gait``
inherited untouched.
"""

import copy

from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from rl.decoupled.mdp import EeSpringContactCommandCfg, LocoManipMPCCommandCfg
from rl.decoupled.tasks.joint_vel_mpc.env_configs.command_cfgs import (
    COMMANDS as _BASE_COMMANDS,
    DUTY_FACTOR,
    EE_REGION,
    EE_TRANSITION,
    GAIT_PERIOD,
    NOMINAL_TIP,
)
from rl.g1_assets.g1_constants import (
    G1_KNEES_BENT_INERTIA,
    G1_MASS,
    HIP_WIDTH,
    PELVIS_KNEES_BENT_Z,
)

COMMANDS: dict[str, CommandTermCfg] = copy.deepcopy(_BASE_COMMANDS)

_CONTACT = dict(
    contact_prob=0.4,
    phantom_prob=0.10,
    contact_resampling_time_range=(3.0, 6.0),
    force_range=(5.0, 25.0),
    force_ramp_time=0.6,
    k_ext=1000.0,
    d_ext=30.0,  # k/d sweep 2026-08-11: d=30 dominates d=10 at k=1000
)

COMMANDS["left_ee_cmd"] = EeSpringContactCommandCfg(
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
)
COMMANDS["right_ee_cmd"] = EeSpringContactCommandCfg(
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
)

COMMANDS["loco_mpc"] = LocoManipMPCCommandCfg(
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
    vel_cmd_name="twist",
    grf_sensor_name="feet_ground_contact",
    run_every_n_steps=5,  # 10 Hz (policy = 50 Hz → every 5 steps)
    resampling_time_range=(1e6, 1e6),
    ee_command_names=("left_ee_cmd", "right_ee_cmd"),
    gate_mode="command",  # no contact sensors in the spring task
)
