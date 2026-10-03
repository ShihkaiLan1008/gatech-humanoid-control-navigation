"""joint_foot_hlip — the foothold-conditioned line with a closed-form H-LIP
reference in place of the centroidal MPC.

Lower command = the same sampled stance-relative foothold queue
(``FootHoldCommand``); ``FootHoldHLIPCommand`` rides its windows and supplies
a CoM-proxy reference on the step's H-LIP orbit (re-anchored from the measured
state at every issue), its orbital energy, and an optional CLF. Rewards =
CLF / proxy tracking + an orbital-energy floor + the goal-only foothold
family. Upper body = EE reach, unchanged.
"""

from mjlab.tasks.registry import register_mjlab_task

from rl.decoupled.runners import JointOnPolicyRunner

from .env_cfgs import joint_foot_hlip_env_cfg
from .rl_cfgs import joint_foot_hlip_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Decoupled-Joint-Foot-HLIP-G1",
    env_cfg=joint_foot_hlip_env_cfg(),
    play_env_cfg=joint_foot_hlip_env_cfg(play=True),
    rl_cfg=joint_foot_hlip_ppo_runner_cfg(),
    runner_cls=JointOnPolicyRunner,
)
