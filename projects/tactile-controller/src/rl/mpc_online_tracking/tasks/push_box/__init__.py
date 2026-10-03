"""push_box — MPC-guided loco-manipulation (box pushing; vel-command era).

Trained OK in the source repo (checkpoint at
``g1_mpc_rl/logs/rsl_rl/g1_mpc_velocity/pushbox``).
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import push_box_env_cfg
from .rl_cfg import push_box_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-MPC-Guided-Loco-manipulation-G1",
    env_cfg=push_box_env_cfg(),
    play_env_cfg=push_box_env_cfg(play=True),
    rl_cfg=push_box_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
