"""Termination declarations."""

import math

from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp

TERMINATIONS: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mjlab_vel_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
        func=mjlab_vel_mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
    ),
    "base_height": TerminationTermCfg(
        func=mjlab_vel_mdp.root_height_below_minimum,
        params={"minimum_height": 0.5},
    ),
    "illegal_contact": TerminationTermCfg(
        func=mjlab_vel_mdp.illegal_contact,
        params={"sensor_name": "nonfoot_ground_touch"},
    ),
}
