"""Terminations: joint_foot_mpc's verbatim."""
import copy

from rl.decoupled.tasks.joint_foot_mpc.env_configs.termination_cfgs import (
    TERMINATIONS as _BASE,
)

TERMINATIONS = copy.deepcopy(_BASE)
