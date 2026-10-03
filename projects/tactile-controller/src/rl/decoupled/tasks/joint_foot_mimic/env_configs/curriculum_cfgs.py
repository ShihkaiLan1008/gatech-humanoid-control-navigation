"""Curriculum declarations — REFINE-DP ramps the smoothness regularizers
(accuracy first, smoothness later). Steps are `env.common_step_counter` (one
per policy step of the vectorized env; ~24 per training iteration).

Stage-2 moved 100k -> 250k and stage-3 400k -> 600k (run-5 forensics: feet err
bottomed at 0.053 by iter ~2.5k, then REGRESSED starting exactly at the 100k
ramp boundary — action_rate x50 + joint_acc x10 landed on a gait that was
still consolidating; the standalone survived the same ramp but without the
co-training arm disturbance).

`ee_target_blend_*`: the upper-body amplitude curriculum — EE pool draws start
90 % blended toward the fixed nominal tip (small motions, upper still learns
tracking) and anneal CONTINUOUSLY to the full pool while the lower discovers
stepping; see `_BLEND_STAGES` for the curve.
"""

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg

from rl.decoupled import mdp




_BLEND_STAGES = [
    {"step":       0, "value": 0.900},
    {"step":  40_000, "value": 0.900},
    {"step":  60_000, "value": 0.867},
    {"step":  80_000, "value": 0.822},
    {"step": 100_000, "value": 0.763},
    {"step": 120_000, "value": 0.684},
    {"step": 140_000, "value": 0.578},
    {"step": 160_000, "value": 0.438},
    {"step": 180_000, "value": 0.250},
    {"step": 200_000, "value": 0.000},
]


def _blend_anneal(command_name: str) -> CurriculumTermCfg:
    return CurriculumTermCfg(
        func=mdp.command_attr_curriculum,
        params={
            "command_name": command_name,
            "attr": "nominal_blend",
            "interp": True,
            "stages": _BLEND_STAGES,
        },
    )


CURRICULUMS: dict[str, CurriculumTermCfg] = {
    "ee_target_blend_left": _blend_anneal("left_ee_cmd"),
    "ee_target_blend_right": _blend_anneal("right_ee_cmd"),
}
