"""Event declarations — identical to hand_pose (base pushes etc.; cube
placement is NOT an event — the hand command term writes the mocap pose every
step)."""

from ...hand_pose.env_configs.event_cfgs import (  # noqa: F401
    EVENTS,
    FOOT_GEOMS,
    NON_CONTACT_GEOMS,
    RANDOMIZE_TERRAIN_EVENT,
)
