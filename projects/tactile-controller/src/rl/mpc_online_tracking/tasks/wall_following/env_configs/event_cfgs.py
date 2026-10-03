"""Event declarations — hand_force verbatim (maze placement is NOT an event:
the FSM twist command owns the wall map and re-poses the walls at reset)."""

from ...hand_force.env_configs.event_cfgs import (  # noqa: F401
    EVENTS,
    RANDOMIZE_TERRAIN_EVENT,
)
