"""Commands — joint_vel's EE terms plus the robot_rl LIP-CLF chain:
twist (their base_velocity box) -> gait clock -> orbit-step foothold planner
-> H-LIP reference + CLF. Declaration order is the update order."""

import copy
import math

from mjlab.managers.command_manager import CommandTermCfg

from rl.decoupled.mdp import (
    HLIPCommandTermCfg,
    GaitPhaseCommandCfg,
    TrackingMetricsVelocityCommandCfg,
    UniformVelocityCommandCfg,
)
from rl.decoupled.tasks.joint_vel.env_configs.command_cfgs import COMMANDS as _BASE
from rl.g1_assets.g1_constants import PELVIS_KNEES_BENT_Z
from rl.decoupled.tasks.joint_vel_hlip.mdp import HlipReferenceCommandCfg

GAIT_PERIOD = 0.8
"""Full gait cycle T_cycle (two touchdowns), seconds; T_step = 0.4."""

Y_NOM = 0.25
"""Nominal lateral foot offset y_nom, m."""

FOOT_Y_ABS_RANGE = (0.10, 0.50)
"""Clamp on the lateral distance of the landing target from the stance foot, m."""

HLIP_CLF = {
    # (q_pos, q_vel) per channel
    "Q_weights": {
        "com:x": (25.0, 200.0), "com:y": (300.0, 50.0), "com:z": (400.0, 10.0),
        "pelvis:roll": (420.0, 20.0), "pelvis:pitch": (200.0, 10.0), "pelvis:yaw": (300.0, 10.0),
        "swing:x": (1500.0, 125.0), "swing:y": (1700.0, 125.0), "swing:z": (3500.0, 100.0),
        "swing:roll": (30.0, 1.0), "swing:pitch": (10.0, 1.0), "swing:yaw": (400.0, 10.0),
        "joint:waist_yaw": (500.0, 10.0), "joint:waist_roll": (500.0, 10.0), "joint:waist_pitch": (500.0, 10.0),
    },
    "R_weights": {
        "com:x": 0.1, "com:y": 0.1, "com:z": 0.1,
        "pelvis:roll": 0.05, "pelvis:pitch": 0.05, "pelvis:yaw": 0.05,
        "swing:x": 0.05, "swing:y": 0.05, "swing:z": 0.05,
        "swing:roll": 0.02, "swing:pitch": 0.02, "swing:yaw": 0.02,
        "joint:waist_yaw": 0.1, "joint:waist_roll": 0.1, "joint:waist_pitch": 0.1,
    },
}
"""CARE weights of the CLF over the 15 reference channels."""

COMMANDS: dict[str, CommandTermCfg] = {
    "left_ee_cmd": copy.deepcopy(_BASE["left_ee_cmd"]),
    "right_ee_cmd": copy.deepcopy(_BASE["right_ee_cmd"]),
    "twist": TrackingMetricsVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        rel_forward_envs=0.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.75, 0.75),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
    ),
    "gait": GaitPhaseCommandCfg(
        period=GAIT_PERIOD,
        command_name="twist",
        command_threshold=0.1,
        stance_decay=0.95,
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
    "hlip_ref": HlipReferenceCommandCfg(
        debug_vis=False,  # play assembly turns this on
        entity_name="robot",
        planner_name="foot_plan",
        gait_name="gait",
        twist_name="twist",
        stance_width=Y_NOM,
        swing_start="nominal",
        z_sw_max=0.065,
        pitch_ref=0.0,
        pitch_amp=0.02,
        roll_asym_amp=-0.05,
        roll_bias_lat_max=0.15,
        roll_bias_yaw_max=0.2,
        invalidate_on_event=True,
        clf=HLIP_CLF,
    ),
}
