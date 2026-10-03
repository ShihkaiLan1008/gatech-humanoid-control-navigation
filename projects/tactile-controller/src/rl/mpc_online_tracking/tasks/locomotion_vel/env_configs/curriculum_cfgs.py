"""Curriculum declarations — single-stage velocity curriculum (Themis flat)."""

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.velocity import mdp as mjlab_vel_mdp

CURRICULUM: dict[str, CurriculumTermCfg] = {
    "command_vel": CurriculumTermCfg(
        func=mjlab_vel_mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {
                    "step": 0,
                    "lin_vel_x": (-0.5, 1.0),
                    "lin_vel_y": (-0.75, 0.75),
                    "ang_vel_z": (-1.0, 1.0),
                },
            ],
        },
    ),
}
