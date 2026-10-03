"""Action declarations — two joint-position terms, 29 DoF total.

`lower_body_joint_pos` (12 legs + 3 waist) then `upper_body_joint_pos` (14
arms). The declaration ORDER is the layout of the concatenated action vector,
and lower-first is what makes it coincide with the robot's natural joint order
(legs 0-11, waist 12-14, arms 15-28) — see the note on `preserve_order` below.
"""

import re

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg

from rl.g1_assets.g1_constants import G1_ACTION_SCALE
from rl.decoupled.mdp import LOWER_JOINT_NAMES, UPPER_JOINT_NAMES


def _scale_for(joint_names: tuple[str, ...]) -> dict[str, float]:
    """The `G1_ACTION_SCALE` entries that belong to this term's joints.

    Both directions matter to mjlab's `resolve_matching_names_values`:
    a pattern matching NONE of the term's targets raises, and a target matched
    by NO pattern silently keeps scale 1.0. So assert an exact cover here rather
    than discover either failure at training time.
    """
    scales = {
        pattern: scale
        for pattern, scale in G1_ACTION_SCALE.items()
        if any(re.fullmatch(pattern, j) for j in joint_names)
    }
    uncovered = [
        j for j in joint_names if not any(re.fullmatch(p, j) for p in scales)
    ]
    if uncovered:
        raise ValueError(f"G1_ACTION_SCALE does not cover {uncovered}")
    return scales


LOWER_ACTION_SCALE = _scale_for(LOWER_JOINT_NAMES)
UPPER_ACTION_SCALE = _scale_for(UPPER_JOINT_NAMES)


JOINT_POS_ACTIONS: dict[str, ActionTermCfg] = {
    "lower_body_joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=tuple(LOWER_JOINT_NAMES),
        scale=LOWER_ACTION_SCALE,
        use_default_offset=True,
    ),
    "upper_body_joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=tuple(UPPER_JOINT_NAMES),
        scale=UPPER_ACTION_SCALE,
        use_default_offset=True,
    ),
}
