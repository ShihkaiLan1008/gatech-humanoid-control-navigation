"""joint_policy — decoupled upper+lower policies trained TOGETHER in one env.

Two independent PPO stacks (`JointOnPolicyRunner`): 15-DoF lower foot-placement
imitation + 14-DoF upper tactile-tip SE(3) tracking, per-group reward streams
via `RewardGroupManager`, shared group broadcast (termination).
"""

from mjlab.tasks.registry import register_mjlab_task

from rl.decoupled.runners import JointOnPolicyRunner

from .env_cfgs import joint_vel_env_cfg
from .rl_cfgs import joint_policy_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Vel-G1",
    env_cfg=joint_vel_env_cfg(),
    play_env_cfg=joint_vel_env_cfg(play=True, stand_still=False),
    rl_cfg=joint_policy_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)
