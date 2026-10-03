"""Reward declarations — identical to locomotion_vel (goal-seeking enters
through the MPC plan: the attractor's v_des drives the MPC references, and the
mpc_* tracking rewards carry it; no separate goal reward needed at this
stage)."""

from ...locomotion_vel.env_configs.reward_cfgs import REWARDS  # noqa: F401
