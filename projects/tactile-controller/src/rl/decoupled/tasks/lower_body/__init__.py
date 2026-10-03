"""lower_body — foot-placement tracking policy (REFINE-DP pi_lower).

15-DoF action (legs + waist), arms PD-held; motion imitation on the
data/foot_placement single-step clips; command = (swing indicator, countdown,
target swing-foot pose in the stance-foot frame).
"""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import lower_body_env_cfg
from .rl_cfg import lower_body_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Decoupled-Lower-G1",
    env_cfg=lower_body_env_cfg(),
    play_env_cfg=lower_body_env_cfg(play=True),
    rl_cfg=lower_body_ppo_runner_cfg(),
)


register_mjlab_task(
    task_id="Mjlab-Decoupled-Lower-G1-Planner",
    env_cfg=lower_body_env_cfg(),
    play_env_cfg=lower_body_env_cfg(play=True, planner=True),
    rl_cfg=lower_body_ppo_runner_cfg(),
)
