"""joint_vel_force_mpc — hand FORCE on top of the vel-tracking joint stack
(CURRENT PLAN: the force/MPC layer on the real-stable joint_vel_mpc baseline).

Same two-PPO architecture and OBS LAYOUT as joint_vel_mpc (the zero-reserve
slots go live, shapes unchanged → resume its checkpoints strict=True); adds
press cubes materializing per-hand pose+mask+force commands, the measured
hand_force reward family (upper group), and the ``loco_mpc`` command upgraded
to the 18-D loco-manip ``cd`` QP (hand slots gated on mask ∧ debounced
contact — the press-bracing channel for the lower group).
"""

from mjlab.tasks.registry import register_mjlab_task

from rl.decoupled.runners import JointOnPolicyRunner

from .env_cfgs import joint_vel_force_mpc_env_cfg
from .rl_cfgs import joint_vel_force_mpc_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Vel-Force-MPC-G1",
    env_cfg=joint_vel_force_mpc_env_cfg(),
    play_env_cfg=joint_vel_force_mpc_env_cfg(play=True),
    rl_cfg=joint_vel_force_mpc_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)
