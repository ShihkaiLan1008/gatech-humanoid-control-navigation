"""hand_pose — CURRENT GOAL hand layer, stage 1 (mask ≡ 0).

SE(2) pose-goal locomotion (locomotion_pos machinery) + dual-hand
``wrist_yaw_link`` SE(3) pose tracking sampled in the pelvis-riding annular
reach cylinder, with wrist force DR (constant + impulse) and the stage-2
mask/force command slots already in the obs layout (frozen at zero). The MPC
stays the feet-only locomotion QP. Fresh training required (obs +22 dims vs
locomotion_pos).
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import hand_pose_env_cfg
from .rl_cfg import hand_pose_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-MPC-Guided-Hand-Pose-G1",
    env_cfg=hand_pose_env_cfg(),
    play_env_cfg=hand_pose_env_cfg(play=True),
    rl_cfg=hand_pose_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
