"""The GLOBAL <-> LOCAL contract, and the firewall that keeps truth out of it.

Two dataclasses carry everything that crosses the boundary in each direction.
They exist as types rather than as loose kwargs so the boundary can be TESTED:
``test_interface.py`` asserts that ``LocalReport`` -- the only thing the global
side consumes -- carries no simulator state, and that the belief-update entry
point accepts nothing else.

THE GROUND-TRUTH FIREWALL.  The simulator knows the true base pose, the true
wall poses, and where every contact really landed.  None of it may reach
planning, association, or the belief; all of it is needed for evaluation.  The
mechanism is structural rather than disciplinary:

* ``LocalReport`` has no field that could carry it.  A realized contact is a
  point in the ROBOT'S BODY FRAME (which is what forward kinematics gives a real
  robot) plus the odometry increment that produced the body frame.  Converting
  that to a world point requires a pose the local side does not have.
* ``TruthLog`` is write-only from the executive's side and is handed to the
  evaluator, never to the belief.  ``BeliefInput`` is a frozen view that
  physically cannot reference it.
* ``assert_no_truth_leak`` walks a report and raises on any field name matching
  the truth vocabulary, so a future field named ``true_*`` or ``world_*`` fails
  a test rather than silently working.

The one legitimate use of exact geometry in planning is the KNOWN FLOORPLAN --
the map is given, that is the premise of the whole program.  Analytic ray-casts
against the map are used to CHOOSE a station (would touching here distinguish my
hypotheses?).  They are never used as a probe OUTCOME: the belief updates only
from what the arm actually felt.  ``ProbeOutcome`` is constructed only inside
the executive's PRESS handler, from measured force and measured wrist pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "RouteDecision", "GlobalDirective", "ProbeOutcome", "LocalReport",
    "ExecStatus", "TruthLog", "assert_no_truth_leak", "TRUTH_VOCABULARY",
]


class RouteDecision(str, Enum):
    CONTINUE = "CONTINUE"
    TURN = "TURN"
    PROBE = "PROBE"
    ARRIVE = "ARRIVE"
    ABORT = "ABORT"


# ── global -> local ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GlobalDirective:
    """What the global planner asks the physical pipeline to do next.

    Everything here is expressed in the MAP frame, because that is the frame the
    global belief lives in.  The executive converts to the world frame through
    the belief's own pose estimate -- never through the simulator's.
    """

    route_edge: int
    """Which floorplan edge the robot is following right now."""

    waypoint_map_xy: tuple[float, float]
    """Locomotion target, map frame.  ``None`` semantics are not used: a
    directive always names somewhere to be, even if it is where you are."""

    waypoint_yaw_map: float
    """Heading to hold on arrival -- the followed edge's facing yaw."""

    station: str | None = None
    """Name of the affordance-graph station to probe at, if this directive is a
    probe directive.  Reachability is guaranteed at graph construction, and
    re-checked against the measured standoff band before any press."""

    station_map_xy: tuple[float, float] | None = None
    hypotheses: tuple[dict, ...] = ()
    """Active route/landmark hypotheses: ``mass``, ``s``, ``xy`` per mode.  The
    executive navigates by ONE of these (the tracked mode), which is why they
    cross the boundary rather than a single pose."""

    probe_budget: int = 0
    """Probes the local side may spend at this station before returning control.
    Bounded per station AND per episode; the executive enforces both."""

    allowed_decisions: tuple[RouteDecision, ...] = ()
    """What the planner will accept next.  TURN absent means the branch gate is
    not met and the executive must not commit, however good the geometry looks."""

    turn_to_edge: int | None = None
    turn_yaw_map: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_edge": self.route_edge,
            "waypoint_map_xy": list(self.waypoint_map_xy),
            "waypoint_yaw_map": self.waypoint_yaw_map,
            "station": self.station,
            "n_hypotheses": len(self.hypotheses),
            "hypothesis_masses": [round(float(h["mass"]), 3) for h in self.hypotheses],
            "probe_budget": self.probe_budget,
            "allowed_decisions": [d.value for d in self.allowed_decisions],
            "turn_to_edge": self.turn_to_edge,
            "reason": self.reason,
        }


# ── local -> global ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProbeOutcome:
    """One executed probe, as the ROBOT measured it.

    ``p_body`` and ``normal_body`` are body-frame -- forward kinematics plus the
    force sensor, nothing else.  ``swept_free_body`` is the sampled path the
    wrist actually swept without touching anything, also body-frame, and is
    present for BOTH outcomes: a contact certifies the space in front of it free
    up to the touch, which is real negative information and would be thrown away
    by recording contacts only.
    """

    kind: str                                   # "contact" | "swept_free" | "invalid"
    p_body: tuple[float, float] | None
    normal_body: tuple[float, float] | None
    swept_free_body: tuple[tuple[float, float], ...]
    peak_force_n: float
    fan_offset_m: float
    station: str | None = None
    invalid_reason: str = ""

    @property
    def informative(self) -> bool:
        """An invalid probe is a CONTROLLER event, not a measurement.  It must
        not reach the belief -- a sweep that stopped short of the mapped face
        says nothing about geometry (ported from the intermittent pipeline,
        where letting it through corrupted the terminus estimate)."""
        return self.kind in ("contact", "swept_free")


@dataclass(frozen=True)
class ExecStatus:
    ok: bool = True
    owner: str | None = None                    # planner|estimator|ate|controller|route
    reason: str = ""
    peak_force_n: float = 0.0
    shield_aborts: int = 0


@dataclass(frozen=True)
class LocalReport:
    """Everything the global side is allowed to see after a local action.

    Deliberately impoverished.  There is no base pose here, no wall pose, no
    contact point in world coordinates -- a real robot walking a real corridor
    has none of those, and neither does this.
    """

    odom_delta_body: tuple[float, float]
    """Translation since the last report, in the body frame at the start of the
    interval.  This is the odometry increment the belief's motion model eats."""

    odom_delta_yaw: float
    probes: tuple[ProbeOutcome, ...] = ()
    ate_belief: dict = field(default_factory=dict)
    """The LOCAL geometry belief (terminus mean/variance, decision, entropy) --
    a summary, not a pose."""

    status: ExecStatus = field(default_factory=ExecStatus)
    station: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "odom_delta_body": list(self.odom_delta_body),
            "odom_delta_yaw": self.odom_delta_yaw,
            "n_probes": len(self.probes),
            "n_contacts": sum(1 for p in self.probes if p.kind == "contact"),
            "n_swept_free": sum(1 for p in self.probes if p.kind == "swept_free"),
            "n_invalid": sum(1 for p in self.probes if p.kind == "invalid"),
            "peak_force_n": self.status.peak_force_n,
            "ate_belief": dict(self.ate_belief),
            "station": self.station,
            "ok": self.status.ok,
            "owner": self.status.owner,
            "reason": self.status.reason,
        }


# ── the firewall ─────────────────────────────────────────────────────────────────


TRUTH_VOCABULARY = ("true_", "truth", "world_xy", "sim_", "gt_", "oracle",
                    "base_pos_w", "root_", "_w_", "wall_pose")
"""Field-name fragments that would indicate simulator state.  A report carrying
one fails ``assert_no_truth_leak``.  Crude on purpose: the point is that adding
a leaking field has to be a deliberate act that breaks a test."""


def assert_no_truth_leak(payload: Any, *, path: str = "LocalReport") -> None:
    """Raise if any field name in a nested payload matches the truth vocabulary."""
    if isinstance(payload, dict):
        items = payload.items()
    elif hasattr(payload, "__dataclass_fields__"):
        items = ((k, getattr(payload, k)) for k in payload.__dataclass_fields__)
    else:
        return
    for key, value in items:
        low = str(key).lower()
        for bad in TRUTH_VOCABULARY:
            if bad in low:
                raise AssertionError(f"ground-truth leak: {path}.{key}")
        if isinstance(value, (dict,)) or hasattr(value, "__dataclass_fields__"):
            assert_no_truth_leak(value, path=f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                assert_no_truth_leak(item, path=f"{path}.{key}[{i}]")


@dataclass
class TruthLog:
    """Simulator ground truth, for EVALUATION ONLY.

    Written by the executive, read by the evaluator and the plots.  Never passed
    to the belief, the planner, or the association gate -- and the belief-update
    entry point takes ``LocalReport``, so it could not accept this even by
    mistake.
    """

    base_xy_w: list = field(default_factory=list)
    base_yaw_w: list = field(default_factory=list)
    time_s: list = field(default_factory=list)
    mode: list = field(default_factory=list)
    contact_xy_w: list = field(default_factory=list)
    wrong_turn: bool = False
    wrong_turn_reason: str = ""

    def sample(self, *, t: float, xy, yaw: float, mode: str) -> None:
        self.time_s.append(float(t))
        self.base_xy_w.append([float(xy[0]), float(xy[1])])
        self.base_yaw_w.append(float(yaw))
        self.mode.append(str(mode))

    def contact(self, xy) -> None:
        self.contact_xy_w.append([float(xy[0]), float(xy[1])])

    def as_arrays(self) -> dict:
        return {
            "t": np.asarray(self.time_s, float),
            "xy": np.asarray(self.base_xy_w, float).reshape(-1, 2),
            "yaw": np.asarray(self.base_yaw_w, float),
            "mode": list(self.mode),
            "contacts": np.asarray(self.contact_xy_w, float).reshape(-1, 2),
        }
