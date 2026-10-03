"""Observation declarations — locomotion_vel base + privileged box channels.

Actor is unchanged except the phase deadzone (push commands are small); the
critic additionally sees box pose/velocity/size and hand-box contact.
"""

import copy

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from ...locomotion_vel.env_configs.observation_cfgs import (
    ACTOR_TERMS as _VEL_ACTOR_TERMS,
    CRITIC_TERMS as _VEL_CRITIC_TERMS,
)
from .. import mdp
from .command_cfgs import PUSH_CMD_THRESHOLD

ACTOR_TERMS: dict[str, ObservationTermCfg] = copy.deepcopy(_VEL_ACTOR_TERMS)
ACTOR_TERMS["phase"].params["command_threshold"] = PUSH_CMD_THRESHOLD

CRITIC_TERMS: dict[str, ObservationTermCfg] = copy.deepcopy(_VEL_CRITIC_TERMS)
CRITIC_TERMS["box_pose_rel"] = ObservationTermCfg(
    func=mdp.box_pose_rel_priv,
    params={"box_name": "box", "robot_name": "robot"},
)
CRITIC_TERMS["box_lin_vel"] = ObservationTermCfg(
    func=mdp.box_lin_vel_priv,
    params={"box_name": "box", "robot_name": "robot"},
)
CRITIC_TERMS["box_size"] = ObservationTermCfg(
    func=mdp.box_size_priv,
    params={"box_name": "box", "geom_name": "box_geom"},
)
CRITIC_TERMS["hand_box_contact"] = ObservationTermCfg(
    func=mdp.hand_box_contact_priv,
    params={
        "lhand_sensor": "lhand_box_contact",
        "rhand_sensor": "rhand_box_contact",
    },
)

OBSERVATIONS: dict[str, ObservationGroupCfg] = {
    "actor": ObservationGroupCfg(
        terms=ACTOR_TERMS,
        concatenate_terms=True,
        enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
        terms=CRITIC_TERMS,
        concatenate_terms=True,
        enable_corruption=False,
    ),
}
