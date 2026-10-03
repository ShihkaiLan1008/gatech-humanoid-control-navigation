import copy

import assets
from mjlab.managers.command_manager import CommandTermCfg

from ...hand_pose.env_configs.command_cfgs import COMMANDS as _HP_COMMANDS
from ...hand_pose.mdp import UniformEePoseCommandCfg
from ...locomotion_vel.env_configs.command_cfgs import (
    G1_COM_HEIGHT,
    G1_HIP_WIDTH,
    G1_MASS,
    GAIT_PERIOD,
)
from ..mdp import HandForceMPCCommandCfg, ReachPoolHandForceEePoseCommandCfg


# Tip-frame IK chain, from the repo's one asset tree (src/assets). Same file the
# offline TO solves against; pytorch-kinematics parses the TEXT only.
_TACTILE_URDF = str(assets.ASSET_DIR / assets.TACTILE_URDF)

COMMANDS: dict[str, CommandTermCfg] = {}


COMMANDS["twist"] = copy.deepcopy(_HP_COMMANDS["twist"])


COMMANDS["loco_mpc"] = HandForceMPCCommandCfg(
    debug_vis=False,  
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
    hand_site_names=("left_tactile_site", "right_tactile_site"),
    ee_command_names=("left_ee_pose", "right_ee_pose"),
    lhand_sensor_name="lhand_cube_contact",
    rhand_sensor_name="rhand_cube_contact",
    mu_hand=0.6,
    f_hand_max=50.0,
    f_hand_headroom=1.25,
    R_f_hand=1e-1,
    R_hand_balance=0.0,
    contact_debounce_steps=5,
)


# NOT consumed by this task since the reach-pool switch (2026-08-01) —
# hand_force contact draws keep the shared pool pose. wall_following still
# imports these as its wall-spot y/z bands (its x comes from the wall).
_LEFT_CONTACT_RANGES = UniformEePoseCommandCfg.Ranges(
    pos_x=(0.33, 0.45),
    pos_y=(0.05, 0.30),
    pos_z=(0.10, 0.20),
    roll=(-0.1, 0.1),
    pitch=(-0.1, 0.1),
    yaw=(-0.1, 0.1),
)
_RIGHT_CONTACT_RANGES = UniformEePoseCommandCfg.Ranges(
    pos_x=(0.33, 0.45),
    pos_y=(-0.30, -0.05),
    pos_z=(0.10, 0.20),
    roll=(-0.1, 0.1),
    pitch=(-0.1, 0.1),
    yaw=(-0.1, 0.1),
)

# v7 (2026-08-01): pose sampling switched from the free box + contact sub-box
# to the REACH POOL on the TACTILE TIP frames — every draw (free AND contact:
# contact is a mask on the shared draw) is reachable by construction and
# seeds the solver with its stored exact anchor. The old boxes (free
# (0.33,0.53) half-outside by design, contact (0.33,0.45) trimmed off the
# biased penalty-IK audit) live in git history.
COMMANDS["left_ee_pose"] = ReachPoolHandForceEePoseCommandCfg(
    resampling_time_range=(1.0, 3.0),
    debug_vis=False,
    entity_name="robot",
    body_name="left_tactile",
    urdf_path=_TACTILE_URDF,
    use_yaw_align_frame=True,
    pool_ee_frames=("left_tactile", "right_tactile"),
    contact_prob=0.3,
    phantom_prob=0.10,
    contact_resampling_time_range=(3.0, 6.0),
    force_range=(5.0, 20.0),
    force_ramp_time=0.6,
    cube_entity_name="left_press_cube",
    contact_sensor_name="lhand_cube_contact",
)
COMMANDS["right_ee_pose"] = ReachPoolHandForceEePoseCommandCfg(
    resampling_time_range=(1.0, 3.0),
    debug_vis=False,
    entity_name="robot",
    body_name="right_tactile",
    urdf_path=_TACTILE_URDF,
    use_yaw_align_frame=True,
    pool_ee_frames=("left_tactile", "right_tactile"),
    contact_prob=0.3,
    phantom_prob=0.10,
    contact_resampling_time_range=(3.0, 6.0),
    force_range=(5.0, 20.0),
    force_ramp_time=0.6,
    cube_entity_name="right_press_cube",
    contact_sensor_name="rhand_cube_contact",
)

EE_COMMAND_NAMES: tuple[str, str] = ("left_ee_pose", "right_ee_pose")
