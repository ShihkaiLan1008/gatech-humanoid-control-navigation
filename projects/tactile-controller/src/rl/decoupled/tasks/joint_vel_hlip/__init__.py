"""joint_vel_hlip — the decoupled velocity line trained with robot_rl's
LIP-CLF recipe: twist + gait clock -> orbit-step foothold planner -> H-LIP
reference (orbit CoM, Bezier swing, pelvis/foot orientation, waist) priced by
one CLF; the lower actor observes twist + clock only.
"""

from mjlab.tasks.registry import register_mjlab_task

from rl.decoupled.runners import JointOnPolicyRunner

from .env_cfgs import joint_vel_hlip_env_cfg
from .rl_cfgs import joint_vel_hlip_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Vel-HLIP-G1",
    env_cfg=joint_vel_hlip_env_cfg(),
    play_env_cfg=joint_vel_hlip_env_cfg(play=True),
    rl_cfg=joint_vel_hlip_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)
