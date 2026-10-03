"""joint_foot_mpc — the foothold-conditioned line (docs/mpc_rl.md).

Lower command = stance-relative foothold queue on the shared gait clock
(standing = hold mode); the CD MPC consumes the committed queue (clock-only
sigma + support-centroid crude x_ref) and its anticipatory plan pays the
lower policy through the ``mpc_*`` family; goal-only foothold income (swing
kernel + touchdown event) on top.  Upper body = EE reach, unchanged.
"""

from mjlab.tasks.registry import register_mjlab_task

from rl.decoupled.runners import JointOnPolicyRunner

from .env_cfgs import joint_foot_mpc_env_cfg, joint_foot_vforce_mpc_env_cfg
from .rl_cfgs import joint_foot_mpc_ppo_runner_cfg, joint_foot_vforce_mpc_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-MPC-G1",
    env_cfg=joint_foot_mpc_env_cfg(),
    play_env_cfg=joint_foot_mpc_env_cfg(play=True),
    rl_cfg=joint_foot_mpc_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

# PLAY-ONLY: footholds assigned by the operator in viser (drag a marker inside
# its coverage box, then `step left` / `step right` or `walk`).  Same
# experiment name, so play resolves this task's checkpoints.
register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-MPC-G1-Manual",
    env_cfg=joint_foot_mpc_env_cfg(),
    play_env_cfg=joint_foot_mpc_env_cfg(play=True, manual=True),
    rl_cfg=joint_foot_mpc_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

# A + ONE variable: the orbit-clip joint-posture prior (swing leg + waist),
# data/foot_lip_orbit_v1 on the same clock; obs unchanged, so A's checkpoints
# resume strict. Same experiment name.
register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-MPC-G1-Prior",
    env_cfg=joint_foot_mpc_env_cfg(posture_prior=True),
    play_env_cfg=joint_foot_mpc_env_cfg(play=True, posture_prior=True),
    rl_cfg=joint_foot_mpc_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-MPC-G1-Prior-Manual",
    env_cfg=joint_foot_mpc_env_cfg(posture_prior=True),
    play_env_cfg=joint_foot_mpc_env_cfg(play=True, manual=True, posture_prior=True),
    rl_cfg=joint_foot_mpc_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-VForce-MPC-G1",
    env_cfg=joint_foot_vforce_mpc_env_cfg(),
    play_env_cfg=joint_foot_vforce_mpc_env_cfg(play=True),
    rl_cfg=joint_foot_vforce_mpc_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-VForce-MPC-G1-Manual",
    env_cfg=joint_foot_vforce_mpc_env_cfg(),
    play_env_cfg=joint_foot_vforce_mpc_env_cfg(play=True, manual=True),
    rl_cfg=joint_foot_vforce_mpc_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)