"""Termination declarations — identical to hand_pose (no contact-specific
kill: the riding cube cannot leave the reach envelope, and a missed press is
a reward problem, not a failure state)."""

from ...hand_pose.env_configs.termination_cfgs import TERMINATIONS  # noqa: F401
