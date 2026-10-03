"""Observations — IDENTICAL layout to joint_vel_mpc (strict=True resume).

Same zero-reserve activation as the cube task: ``hand_contact_force_cmd``
(8 = per hand mask(1)+force_cmd_b(3)/50) goes live in all four groups and
``hand_ext_force`` (6) in the critics — here it carries the SPRING force
(the ground-truth applied load) instead of a contact-sensor reading. Shapes
never change, so joint_vel_mpc checkpoints resume strict=True.
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
