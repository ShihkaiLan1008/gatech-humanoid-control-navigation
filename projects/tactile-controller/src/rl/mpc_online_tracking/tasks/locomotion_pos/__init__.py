"""locomotion_pos — SE(2) pose-goal locomotion (CURRENT GOAL step 1).

The "twist" slot is fed by a world-latched SE(2) pelvis-target attractor
(``mdp.commands.Se2PoseVelocityCommand``) instead of sampled velocities;
obs/action are identical to locomotion_vel so its checkpoint drops in for
command-layer validation (validated 2026-07-25: 35/35 goals, 0 falls —
``scripts/rl/test/goal_probe.py``).
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import locomotion_pos_env_cfg
from .rl_cfg import locomotion_pos_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-MPC-Guided-Locomotion-Goal-G1",
    env_cfg=locomotion_pos_env_cfg(),
    play_env_cfg=locomotion_pos_env_cfg(play=True),
    rl_cfg=locomotion_pos_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
