"""Action declarations — 15-DoF (12 legs + 3 waist) joint-position action plus
a ZERO-DIM nominal PD hold on the 14 arm joints.

The frozen upper term (`FrozenJointPositionAction`) owns the arm actuators as
a real action term — target = the HOME-keyframe default every step, zero
policy dims — replacing the old hack of the command term writing arm targets
at reset. It is the exact seam the joint tasks' `upper_body_joint_pos` fills.

The robot cfg is the HOME keyframe (env_cfgs), whose defaults ARE the TO
half-sitting nominal — `use_default_offset=True` gives the explicit-offset
semantics of the successful runs.
"""

import re

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg

from rl.g1_assets.g1_constants import G1_ACTION_SCALE
from rl.decoupled.mdp import (
    FrozenJointPositionActionCfg,
    LOWER_JOINT_NAMES,
    UPPER_JOINT_NAMES,
)


def _scale_for(joint_names) -> dict[str, float]:
    return {
        pattern: scale
        for pattern, scale in G1_ACTION_SCALE.items()
        if any(re.fullmatch(pattern, j) for j in joint_names)
    }


JOINT_POS_ACTIONS: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=tuple(LOWER_JOINT_NAMES),
        scale=_scale_for(LOWER_JOINT_NAMES),
        use_default_offset=True,
    ),
    "upper_body_hold": FrozenJointPositionActionCfg(
        entity_name="robot",
        actuator_names=tuple(UPPER_JOINT_NAMES),
        scale=_scale_for(UPPER_JOINT_NAMES),
        use_default_offset=True,
    ),
}
