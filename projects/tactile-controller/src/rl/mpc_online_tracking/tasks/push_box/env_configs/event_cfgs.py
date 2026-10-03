"""Event declarations — locomotion_vel base + push-mode flag, box pose/mass
and box friction resets."""

import copy

from mjlab.managers.event_manager import EventTermCfg

from ...locomotion_vel.env_configs.event_cfgs import (  # noqa: F401
    EVENTS as _VEL_EVENTS,
    FOOT_GEOMS,
    NON_CONTACT_GEOMS,
    RANDOMIZE_TERRAIN_EVENT,
)
from .. import mdp

EVENTS: dict[str, EventTermCfg] = copy.deepcopy(_VEL_EVENTS)

# 50/50 push/walk split in train; play makes every env a push env (see
# env_cfgs — with few envs round(N*0.5) can hit zero and walk-only envs park
# their box 100 m away).
EVENTS["init_push_mode"] = EventTermCfg(
    mode="startup",
    func=mdp.init_push_mode,
    params={"push_fraction": 0.5},
)
EVENTS["reset_box"] = EventTermCfg(
    mode="reset",
    func=mdp.reset_box_pose_and_mass,
    params={
        # G1 arms are shorter than Themis's: spawn the box closer.
        "x_range": (0.7, 1.2),
        "y_range": (0.0, 0.0),
        "yaw_range": (0.0, 0.0),
        "mass_range": (3.0, 15.0),
        "z_clearance": 0.03,
        "box_name": "box",
        "geom_name": "box_geom",
    },
)
EVENTS["reset_box_friction"] = EventTermCfg(
    mode="reset",
    func=mdp.randomize_box_friction_curriculum,
    params={
        # Start slippery; the push_box_friction curriculum raises it.
        "default_friction_range": (0.05, 0.05),
        "box_name": "box",
        "geom_name": "box_geom",
    },
)
