"""Curriculum: joint_foot_mpc's verbatim (EE nominal-blend anneal only)."""
import copy

from rl.decoupled.tasks.joint_foot_mpc.env_configs.curriculum_cfgs import (
    CURRICULUMS as _BASE,
)

CURRICULUMS = copy.deepcopy(_BASE)
