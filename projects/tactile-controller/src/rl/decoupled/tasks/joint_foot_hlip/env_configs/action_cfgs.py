"""Actions: joint_foot_mpc's joint-position terms verbatim."""
import copy

from rl.decoupled.tasks.joint_foot_mpc.env_configs.action_cfgs import (
    JOINT_POS_ACTIONS as _BASE,
)

JOINT_POS_ACTIONS = copy.deepcopy(_BASE)
