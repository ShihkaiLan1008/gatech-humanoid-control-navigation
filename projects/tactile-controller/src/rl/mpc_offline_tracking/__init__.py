from mjlab.tasks.registry import register_mjlab_task
from .runner import MotionTrackingOnPolicyRunner

from .env_cfgs import unitree_g1_mpc_tracking_env_cfg
from .rl_cfg import unitree_g1_mpc_tracking_ppo_runner_cfg

register_mjlab_task(
    task_id="Unitree-G1-MPC-Tracking",
    env_cfg=unitree_g1_mpc_tracking_env_cfg(),
    play_env_cfg=unitree_g1_mpc_tracking_env_cfg(play=True),
    rl_cfg=unitree_g1_mpc_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
