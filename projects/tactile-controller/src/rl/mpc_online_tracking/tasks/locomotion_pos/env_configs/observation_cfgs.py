"""Observation declarations — identical to locomotion_vel BY DESIGN (final).

DECISION (2026-07-25): the actor's 3-dim "command" slot carries the
attractor's v_des permanently — the policy is a goal-blind vel-follower and
the goal lives only in the planner layer. Goal-error obs was rejected: under
planner-parameter DR (speed_scale etc.) the goal is not a sufficient
statistic for v_des, and the twist-mix coverage envs have no goal at all.
If anticipatory behavior is ever needed, ADD goal dims alongside v_des
(never replace it) together with the planner params.
"""

from ...locomotion_vel.env_configs.observation_cfgs import (  # noqa: F401
    ACTOR_TERMS,
    CRITIC_TERMS,
    OBSERVATIONS,
)
