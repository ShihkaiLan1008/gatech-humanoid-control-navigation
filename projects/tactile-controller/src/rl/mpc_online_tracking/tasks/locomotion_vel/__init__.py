"""locomotion_vel — MPC-guided velocity locomotion (vel-command era baseline).

The base task the other tasks derive their declarations from. Trained OK in
the source repo (checkpoint at
``g1_mpc_rl/logs/rsl_rl/g1_mpc_velocity/locomotion``).
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import locomotion_vel_env_cfg
from .rl_cfg import locomotion_vel_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-MPC-Guided-Locomotion-G1",
    env_cfg=locomotion_vel_env_cfg(),
    play_env_cfg=locomotion_vel_env_cfg(play=True),
    rl_cfg=locomotion_vel_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
