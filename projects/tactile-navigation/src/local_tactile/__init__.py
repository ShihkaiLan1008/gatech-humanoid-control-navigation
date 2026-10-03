"""Persistent intermittent tactile navigation (LOCOMOTION xor CONTACT).

One MuJoCo environment, one reset, no teleportation.  ``LOCOMOTION`` walks the
G1 sideways along a wall it is facing with both hands parked; ``CONTACT``
(``STABILIZE`` -> ``PROBE``) stops the base, waits for both feet and a low base
velocity, and runs the local-geometry ATE probe cycle.  The route layer
combines the ATE belief with the known floorplan to continue, turn, or ask for
another probe.
"""

from .ate import LocalGeometryATE, ProbeCandidate, ProbeStance
from .floorplan import (
    Association,
    Floorplan,
    RouteDecision,
    RouteOutcome,
    RouteState,
    Vertex,
    WallEdge,
)
from .fsm import ExclusionAudit, Mode, StabilityGate

__all__ = [
    "Association",
    "ExclusionAudit",
    "Floorplan",
    "LocalGeometryATE",
    "Mode",
    "ProbeCandidate",
    "ProbeStance",
    "RouteDecision",
    "RouteOutcome",
    "RouteState",
    "StabilityGate",
    "Vertex",
    "WallEdge",
]
