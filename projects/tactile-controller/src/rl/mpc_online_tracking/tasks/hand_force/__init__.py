"""hand_force — CURRENT GOAL hand layer, stage 2 (contact mask + force live).

hand_pose v5.2 machinery (sampled twists, yaw-align boxes, IK mimic) plus the
full sparse command tuple: per-hand contact mask and commanded press force,
materialized by per-hand press cubes that ride the commands (no wall, no
world latch, no velocity gating — see g1_assets/press_cube.py). The MPC
upgrades to the 18-D loco-manip QP consuming mask/force, so its CoM/GRF/
foot-placement reward references are press-consistent. Obs layout is
hand_pose-identical: stage-1 checkpoints RESUME.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import hand_force_env_cfg
from .rl_cfg import hand_force_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-MPC-Guided-Hand-Force-G1",
    env_cfg=hand_force_env_cfg(),
    play_env_cfg=hand_force_env_cfg(play=True),
    rl_cfg=hand_force_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
