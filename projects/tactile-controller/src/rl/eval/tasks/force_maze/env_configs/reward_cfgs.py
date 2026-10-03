"""Rewards — the DECOUPLED force line's groups, imported UNCHANGED.

Nothing trains here; the reward streams are kept as EVAL METRICS (the
hand-force family + gait/mpc bracing terms read against the real walls), and
the JointOnPolicyRunner needs the two reward groups to exist to map its
per-body stacks. Weights are irrelevant in play.
"""

from rl.decoupled.tasks.joint_vel_force_mpc.env_configs.reward_cfgs import (
    LOWER_REWARDS,  # noqa: F401
    UPPER_REWARDS,  # noqa: F401
)
