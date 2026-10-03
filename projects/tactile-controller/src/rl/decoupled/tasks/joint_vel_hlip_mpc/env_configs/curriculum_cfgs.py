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
