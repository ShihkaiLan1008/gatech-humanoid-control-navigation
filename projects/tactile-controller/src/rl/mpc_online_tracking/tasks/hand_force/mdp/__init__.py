"""hand_force task mdp — stage-2 hand command (pose + mask + force), the
loco-manip MPC consumer, live obs slots, and force rewards."""

from .commands import (
    HandForceEePoseCommand,
    HandForceEePoseCommandCfg,
    ReachPoolHandForceEePoseCommand,
    ReachPoolHandForceEePoseCommandCfg,
)
from .mpc_command import HandForceMPCCommand, HandForceMPCCommandCfg
from .observations import hand_contact_force_command, hand_ext_force_meas
from .rewards import (
    HandForceRate,
    contact_alignment,
    hand_contact_hold,
    hand_force_l1,
    hand_force_tracking,
)

__all__ = [
    "HandForceEePoseCommand",
    "HandForceEePoseCommandCfg",
    "ReachPoolHandForceEePoseCommand",
    "ReachPoolHandForceEePoseCommandCfg",
    "HandForceMPCCommand",
    "HandForceMPCCommandCfg",
    "HandForceRate",
    "contact_alignment",
    "hand_contact_force_command",
    "hand_contact_hold",
    "hand_ext_force_meas",
    "hand_force_l1",
    "hand_force_tracking",
]
