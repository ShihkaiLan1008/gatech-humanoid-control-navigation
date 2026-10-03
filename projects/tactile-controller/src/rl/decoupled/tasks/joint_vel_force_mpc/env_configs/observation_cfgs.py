"""Observations — IDENTICAL layout to joint_vel_mpc (strict=True resume).

The base task shipped every group with zero-filled reserve slots exactly for
this moment: ``hand_contact_force_cmd`` (8 = per hand mask(1)+force_b(3)/40)
in all four groups and ``hand_ext_force`` (6, measured, /40) in the critics.
Here the reserves go LIVE with the same dims — shapes never change, so the
trained joint_vel_mpc checkpoint resumes strict=True. Only func/params are
swapped; clip/history/noise attributes carry over from the base declaration.

Measured contact force stays OUT of the actor (standing user rule): the actor
slots carry the COMMAND tuple; ``hand_ext_force`` exists only in the critics.
"""

import copy

from rl.decoupled import mdp
from rl.decoupled.tasks.joint_vel_mpc.env_configs.observation_cfgs import (
    OBSERVATIONS as _BASE_OBSERVATIONS,
)

OBSERVATIONS = copy.deepcopy(_BASE_OBSERVATIONS)

_EE = ("left_ee_cmd", "right_ee_cmd")

for _group in OBSERVATIONS.values():
    _terms = _group.terms
    if "hand_contact_force_cmd" in _terms:
        _terms["hand_contact_force_cmd"].func = mdp.hand_contact_force_command
        _terms["hand_contact_force_cmd"].params = {"command_names": _EE}
    if "hand_ext_force" in _terms:
        _terms["hand_ext_force"].func = mdp.hand_ext_force_meas
        _terms["hand_ext_force"].params = {"command_names": _EE}
