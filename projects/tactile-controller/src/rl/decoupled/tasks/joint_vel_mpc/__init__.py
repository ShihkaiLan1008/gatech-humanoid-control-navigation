"""joint_mpc — the decoupled joint policy + the schedule-fed loco-manip MPC +
hand contact/force commands (CURRENT PLAN step 4: hand FORCE on top of the
decoupled stack).

Same two-PPO architecture as joint_policy; adds press cubes materializing
per-hand pose+mask+force commands, the measured hand_force reward family
(upper group), and the ``loco_mpc`` QP whose contact schedule/footholds/x_ref
come FROM the step command (``mpc_*`` rewards = the press-bracing channel,
lower group).
"""

from mjlab.tasks.registry import register_mjlab_task

from rl.decoupled.runners import JointOnPolicyRunner

from .env_cfgs import joint_vel_mpc_env_cfg
from .rl_cfgs import joint_policy_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Vel-MPC-G1",
    env_cfg=joint_vel_mpc_env_cfg(),
    play_env_cfg=joint_vel_mpc_env_cfg(play=True),
    rl_cfg=joint_policy_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

