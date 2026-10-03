"""wall_press — LEGACY (W0 stage plan, superseded 2026-07-25 by CURRENT GOAL).

Kept as reference code: the wall scene / press schedule / world-axis
hand-force machinery may be cribbed when the hand-pose + contact-mask + force
commands land, but this task is NOT the plan — see CLAUDE.md → CURRENT GOAL.
Frozen single-file layout on purpose (no env_configs split for dead code).
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .wall_press_mdp import (
    g1_mpc_wall_press_env_cfg,
    g1_mpc_wall_press_runner_cfg,
)

register_mjlab_task(
    task_id="Mjlab-MPC-Guided-Wall-Press-G1",
    env_cfg=g1_mpc_wall_press_env_cfg(),
    play_env_cfg=g1_mpc_wall_press_env_cfg(play=True),
    rl_cfg=g1_mpc_wall_press_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
