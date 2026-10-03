"""Termination declarations — locomotion_vel base + box-task terminations."""

import copy

from mjlab.managers.termination_manager import TerminationTermCfg

from ...locomotion_vel.env_configs.termination_cfgs import TERMINATIONS as _VEL_TERMINATIONS
from .. import mdp

TERMINATIONS: dict[str, TerminationTermCfg] = copy.deepcopy(_VEL_TERMINATIONS)

TERMINATIONS["robot_far_from_box"] = TerminationTermCfg(
    func=mdp.robot_far_from_box,
    params={
        "max_distance": 2.0,
        "box_name": "box",
        "robot_name": "robot",
    },
)
TERMINATIONS["box_toppled"] = TerminationTermCfg(
    func=mdp.box_toppled,
    params={
        "max_tilt_rad": 1.0,
        "box_name": "box",
    },
)
TERMINATIONS["box_pitched"] = TerminationTermCfg(
    func=mdp.box_pitched,
    params={
        "max_pitch_rad": 0.1745,  # 10 degrees
        "box_name": "box",
    },
)
TERMINATIONS["arm_joint_deviation"] = TerminationTermCfg(
    func=mdp.arm_joint_deviation,
    params={
        "max_deviation_rad": 0.25,
        "robot_name": "robot",
    },
)
TERMINATIONS["box_velocity_stall"] = TerminationTermCfg(
    func=mdp.box_velocity_stall,
    params={
        "min_speed_fraction": 0.5,
        "grace_period_s": 5.0,
        "activate_after_step": 2500,
        "box_name": "box",
        "command_name": "twist",
    },
)
