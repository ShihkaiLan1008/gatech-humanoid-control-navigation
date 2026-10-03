"""Curriculum declarations — REFINE-DP ramps the smoothness regularizers
(accuracy first, smoothness later): action smoothness -0.001 -> -0.5 and joint
acceleration -1e-7 -> -1e-5. Steps are `env.common_step_counter` (one per
policy step of the vectorized env; ~24 per training iteration)."""

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg

CURRICULUM: dict[str, CurriculumTermCfg] = {
    "action_rate_ramp": CurriculumTermCfg(
        func=envs_mdp.reward_curriculum,
        params={
            "reward_name": "action_rate_l2",
            "stages": [
                {"step": 0, "weight": -0.001},
                {"step": 100_000, "weight": -0.05},
                {"step": 400_000, "weight": -0.5},
            ],
        },
    ),
    "joint_acc_ramp": CurriculumTermCfg(
        func=envs_mdp.reward_curriculum,
        params={
            "reward_name": "joint_acc_l2",
            "stages": [
                {"step": 0, "weight": -1.0e-7},
                {"step": 100_000, "weight": -1.0e-6},
                {"step": 400_000, "weight": -1.0e-5},
            ],
        },
    ),
}
