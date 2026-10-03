"""wall_following — maze play env for the sparse-command stack.

A per-env maze of upright wall segments (non-axis-aligned, generous
corridors, re-randomized every reset) + the user's SE(2) FSM: stand / random
walk / approach-a-wall → follow it (side-step hops along the face, facing
it). Contact windows exist ONLY while following a wall and are materialized
by the REAL walls (no press cubes) — "no wall, no contact". Obs layout is
hand_force-identical, so trained hand_force checkpoints play here unchanged.

Play-first: registered like any task, but see CLAUDE.md before training here
(Se2-attractor-as-training-distribution rule + the velocity-payment gap).
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import wall_following_env_cfg
from .rl_cfg import wall_following_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-MPC-Guided-Wall-Following-G1",
    env_cfg=wall_following_env_cfg(),
    play_env_cfg=wall_following_env_cfg(play=True),
    rl_cfg=wall_following_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
