"""joint_policy — decoupled upper+lower policies trained TOGETHER in one env.

Two independent PPO stacks (`JointOnPolicyRunner`): 15-DoF lower foot-placement
imitation + 14-DoF upper tactile-tip SE(3) tracking, per-group reward streams
via `RewardGroupManager`, shared group broadcast (termination).
"""

from mjlab.tasks.registry import register_mjlab_task

from rl.decoupled.runners import JointOnPolicyRunner

from .env_cfgs import joint_env_cfg
from .rl_cfgs import joint_policy_ppo_runner_cfg, joint_foot_mimic_vforce_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-Mimic-G1",
    env_cfg=joint_env_cfg(),
    play_env_cfg=joint_env_cfg(play=True),
    rl_cfg=joint_policy_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

# PLAY-ONLY: Raibert velocity-to-footstep planner shim over the same policy
# interface (same experiment name, so play resolves this task's checkpoints).
register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-Mimic-G1-Planner",
    env_cfg=joint_env_cfg(),
    play_env_cfg=joint_env_cfg(play=True, planner=True),
    rl_cfg=joint_policy_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

# PLAY-ONLY: foot placements assigned by the operator in viser (drag a marker
# inside its coverage box, then `step left` / `step right`).  Same experiment
# name, so play resolves this task's checkpoints.
register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-Mimic-G1-Manual",
    env_cfg=joint_env_cfg(),
    play_env_cfg=joint_env_cfg(play=True, manual=True),
    rl_cfg=joint_policy_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

# Same task on the foot_croco clips + torque/GRF tracking (critic +35 dims).
# Same experiment name; its -Planner / -Manual play variants carry the croco
# critic layout so its checkpoints load strict.
register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-Mimic-G1-Croco",
    env_cfg=joint_env_cfg(croco=True),
    play_env_cfg=joint_env_cfg(play=True, croco=True),
    rl_cfg=joint_policy_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-Mimic-G1-Croco-Planner",
    env_cfg=joint_env_cfg(croco=True),
    play_env_cfg=joint_env_cfg(play=True, planner=True, croco=True),
    rl_cfg=joint_policy_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-Mimic-G1-Croco-Manual",
    env_cfg=joint_env_cfg(croco=True),
    play_env_cfg=joint_env_cfg(play=True, manual=True, croco=True),
    rl_cfg=joint_policy_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-Mimic-G1-VForce",
    env_cfg=joint_env_cfg(force=True),
    play_env_cfg=joint_env_cfg(play=True, force=True),
    rl_cfg=joint_foot_mimic_vforce_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-Mimic-G1-Croco-VForce",
    env_cfg=joint_env_cfg(croco=True, force=True),
    play_env_cfg=joint_env_cfg(play=True, croco=True, force=True),
    rl_cfg=joint_foot_mimic_vforce_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

# PLAY-ONLY: footsteps planned from drawn world-frame SE(2) goals
# (`PositionFootstepCommand`, position route).  Same experiment names.
register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-Mimic-G1-Goal",
    env_cfg=joint_env_cfg(),
    play_env_cfg=joint_env_cfg(play=True, goal=True),
    rl_cfg=joint_policy_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-Mimic-G1-Croco-Goal",
    env_cfg=joint_env_cfg(croco=True),
    play_env_cfg=joint_env_cfg(play=True, goal=True, croco=True),
    rl_cfg=joint_policy_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

# PLAY-ONLY: the VForce interface driven by the operator (gizmo tip goals +
# contact checkbox / force slider per hand, manual footsteps).  Same
# experiment names as -VForce / -Croco-VForce, so play resolves their checkpoints.
register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-Mimic-G1-VForce-Manual",
    env_cfg=joint_env_cfg(force=True),
    play_env_cfg=joint_env_cfg(play=True, manual=True, force=True),
    rl_cfg=joint_foot_mimic_vforce_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-Mimic-G1-Croco-VForce-Manual",
    env_cfg=joint_env_cfg(croco=True, force=True),
    play_env_cfg=joint_env_cfg(play=True, manual=True, croco=True, force=True),
    rl_cfg=joint_foot_mimic_vforce_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)
