"""Curriculum declarations — walk-half velocity stages + box-friction ramp."""

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp

from .. import mdp

CURRICULUM: dict[str, CurriculumTermCfg] = {
    "command_vel": CurriculumTermCfg(
        func=mjlab_vel_mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {
                    "step": 0,
                    "lin_vel_x": (-0.5, 0.75),
                    "lin_vel_y": (-0.5, 0.5),
                    "ang_vel_z": (-0.75, 0.75),
                },
            ],
        },
    ),
    "push_box_friction": CurriculumTermCfg(
        func=mdp.push_box_mass_size_curriculum,
        params={
            "mass_size_stages": [
                {"step": 0, "friction_range": (0.05, 0.05)},
                {"step": 3000, "friction_range": (0.10, 0.10)},
                {"step": 5000, "friction_range": (0.25, 0.25)},
                {"step": 7000, "friction_range": (0.50, 0.50)},
            ],
        },
    ),
}
