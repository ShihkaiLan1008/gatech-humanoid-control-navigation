"""joint_vel_hlip_mpc — the MPC baseline (joint_vel_mpc's obs, rewards and DR)
with the H-LIP foothold planner replacing the Raibert schedule: committed
footholds and the orbit CoM flow feed the centroidal MPC.
"""

from mjlab.tasks.registry import register_mjlab_task

from rl.decoupled.runners import JointOnPolicyRunner

from .env_cfgs import joint_vel_hlip_mpc_env_cfg
from .rl_cfgs import joint_vel_hlip_mpc_ppo_runner_cfg, joint_vel_hlip_mpc_vforce_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Vel-HLIP-MPC-G1",
    env_cfg=joint_vel_hlip_mpc_env_cfg(),
    play_env_cfg=joint_vel_hlip_mpc_env_cfg(play=True),
    rl_cfg=joint_vel_hlip_mpc_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

# VForce: the virtual-spring hand-force layer on the same chain (obs layout
# unchanged -> this task's checkpoints resume strict=True).
register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Vel-HLIP-MPC-G1-VForce",
    env_cfg=joint_vel_hlip_mpc_env_cfg(vforce=True),
    play_env_cfg=joint_vel_hlip_mpc_env_cfg(play=True, vforce=True),
    rl_cfg=joint_vel_hlip_mpc_vforce_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)
