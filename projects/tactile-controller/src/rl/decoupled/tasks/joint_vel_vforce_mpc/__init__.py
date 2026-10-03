"""joint_vel_vforce_mpc — hand FORCE via the VIRTUAL SPRING riding the EE
reference (Portela-style force field), no contact geometry. Same two-PPO
architecture and OBS LAYOUT as joint_vel_mpc (the zero-reserve slots go
live, shapes unchanged → resume its checkpoints strict=True); the
``loco_mpc`` QP gates its hand slots on the command state. Independent of
the cube task (joint_vel_force_mpc) — the two are never chained.
"""

from mjlab.tasks.registry import register_mjlab_task

from rl.decoupled.runners import JointOnPolicyRunner

from .env_cfgs import joint_vel_vforce_mpc_env_cfg
from .rl_cfgs import joint_vel_vforce_mpc_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Vel-VForce-MPC-G1",
    env_cfg=joint_vel_vforce_mpc_env_cfg(),
    play_env_cfg=joint_vel_vforce_mpc_env_cfg(play=True),
    rl_cfg=joint_vel_vforce_mpc_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)
