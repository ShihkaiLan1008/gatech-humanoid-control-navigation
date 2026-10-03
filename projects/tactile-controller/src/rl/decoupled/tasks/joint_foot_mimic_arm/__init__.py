"""joint_foot_mimic_arm — joint_foot_mimic's clf_v3 lower body with the upper
body commanded from the ee_qpos dataset (per hand: tip pose + reference joint
positions, `EEQposeCommand`). Two independent PPO stacks (`JointOnPolicyRunner`).
"""

from mjlab.tasks.registry import register_mjlab_task

from rl.decoupled.runners import JointOnPolicyRunner

from .env_cfgs import joint_env_cfg
from .rl_cfgs import joint_foot_mimic_arm_vforce_ppo_runner_cfg, joint_policy_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-Mimic-Arm-G1",
    env_cfg=joint_env_cfg(),
    play_env_cfg=joint_env_cfg(play=True),
    rl_cfg=joint_policy_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-Mimic-Arm-G1-VForce",
    env_cfg=joint_env_cfg(force=True),
    play_env_cfg=joint_env_cfg(play=True, force=True),
    rl_cfg=joint_foot_mimic_arm_vforce_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

# PLAY-ONLY: the -VForce interface driven by the operator (gizmo tip goals,
# contact checkbox / force slider / virtual-force switch per hand, manual
# footsteps).  Same experiment name as -VForce, so play resolves its checkpoints.
register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-Mimic-Arm-G1-VForce-Manual",
    env_cfg=joint_env_cfg(force=True),
    play_env_cfg=joint_env_cfg(play=True, force=True, manual=True),
    rl_cfg=joint_foot_mimic_arm_vforce_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

# The same task with gliding EE draws (joint-space min-jerk transitions, 10 %
# jumps).  Same experiment name, so play resolves the base checkpoints.
register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-Mimic-Arm-G1-Glide",
    env_cfg=joint_env_cfg(glide=True),
    play_env_cfg=joint_env_cfg(play=True, glide=True),
    rl_cfg=joint_policy_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)
