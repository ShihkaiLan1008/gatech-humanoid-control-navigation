"""What the G1's probe fan can actually measure from a given stance.

The Batch-2 affordance model casts rays at BEARINGS from the body -- right for a
generic sensor, wrong for this robot.  The G1 faces the wall, strafes along it,
and presses its hand STRAIGHT FORWARD at a sequence of lateral offsets: the fan
is a set of PARALLEL rays, displaced along the travel axis, each capped at the
measured 0.45 m reach.  Modelling it any other way would let the planner select
stations whose value it cannot collect.

Every prediction here is analytic geometry against the KNOWN MAP, and it is used
for one purpose only: choosing WHERE TO STAND.  Probe outcomes that reach the
belief come from the arm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from global_nav.floorplan import Floorplan
from global_nav.geometry import raycast

__all__ = ["REACH_M", "G1Station", "ray_disagreement"]

REACH_M = 0.45
"""Measured hand_force_v2 reach.  A press ray longer than this is not a probe
the robot can execute, so the planner may not count on it."""

RANGE_DISCRIMINATION_M = 0.08
"""Two predicted contact ranges count as distinguishable past this.  Set from
the measured contact precision -- FK noise ~0.015 m and a 2 N threshold on a
6 N press -- with a wide margin, not from what would make a scene work."""


def ray_disagreement(a: dict, b: dict) -> float:
    """How distinguishable two predicted press outcomes are, in [0, 1].

    Contact-versus-free is 1.0 and is the channel that matters here: a fan that
    straddles a wall terminus under one hypothesis and sits on flat wall under
    the other is the strongest evidence this robot can collect, and it costs the
    same press as a useless one.
    """
    if a["hit"] != b["hit"]:
        return 1.0
    if not a["hit"]:
        return 0.0
    if abs(a["range"] - b["range"]) > RANGE_DISCRIMINATION_M:
        return 0.9
    return 0.0


@dataclass(frozen=True)
class G1Station:
    """A stance the robot can stop at, plus the fan it would press from there.

    ``xy_map`` is the BASE position; ``facing_map`` the unit direction the robot
    faces (and presses along); ``travel_map`` the unit direction the fan offsets
    displace along.
    """

    name: str
    leg: int
    s_m: float
    xy_map: tuple[float, float]
    facing_map: tuple[float, float]
    travel_map: tuple[float, float]
    fan_offsets_m: tuple[float, ...]
    reach_m: float = REACH_M
    purpose: str = "wall"
    face_offset_m: float = 0.0
    """How far the face this station presses stands PROUD of the leg's wall
    plane.  The approach bias is computed against the wall plane; a landmark
    face 0.08 m closer puts a 0.32 m stance only 0.24 m from the face -- inside
    the parked wrist's 0.26 m envelope -- and the wrist was measured clipping
    the box at 42.5 N while walking between stances.  The approach subtracts
    this offset so the stance keeps the same margin to the face it presses."""

    @property
    def xy(self) -> np.ndarray:
        return np.asarray(self.xy_map, float).reshape(2)

    @property
    def facing(self) -> np.ndarray:
        return np.asarray(self.facing_map, float).reshape(2)

    @property
    def travel(self) -> np.ndarray:
        return np.asarray(self.travel_map, float).reshape(2)

    @property
    def lateral_sign(self) -> float:
        """Body-frame sign of a forward fan offset.

        The hand's body +y is the robot's LEFT, but a fan offset is defined
        along the leg's TRAVEL direction, and the robot may be travelling to its
        right.  Getting this backwards points the whole fan the wrong way down
        the wall -- measured: every press then landed on wall material and no
        probe could ever report swept-free, so the terminus was unobservable and
        the branch gate never opened.
        """
        f, u = self.facing, self.travel
        return float(np.sign(f[0] * u[1] - f[1] * u[0]) or 1.0)

    def body_offset(self, travel_offset_m: float) -> float:
        return self.lateral_sign * float(travel_offset_m)

    def ray_origins(self, delta_along: float = 0.0) -> np.ndarray:
        """Where each press ray starts, with the map shifted by ``delta_along``
        along the robot's travel axis (a registration hypothesis)."""
        base = self.xy - delta_along * self.travel
        return np.array([base + o * self.travel for o in self.fan_offsets_m])

    def predict(self, plan: Floorplan, delta_along: float = 0.0) -> list[dict]:
        out = []
        for origin in self.ray_origins(delta_along):
            d, n, i = raycast(origin, self.facing, self.reach_m, plan.bodies)
            if d is None:
                out.append({"hit": False, "range": self.reach_m, "signature": None})
            else:
                out.append({"hit": True, "range": float(d),
                            "signature": plan.bodies[i].signature})
        return out

    def reachable(self, plan: Floorplan, delta_along: float = 0.0,
                  margin_m: float = 0.02) -> bool:
        """Is at least one ray a press this robot could complete?

        A ray that contacts must do so inside the reach; a ray that sweeps free
        must clear the mapped face by ``margin_m`` to certify anything.  A
        station where every ray fails both is not a probe station.
        """
        for ray in self.predict(plan, delta_along):
            if ray["hit"] and ray["range"] <= self.reach_m - margin_m:
                return True
            if not ray["hit"]:
                return True
        return False

    def discrimination(self, plan: Floorplan, delta_a: float, delta_b: float,
                       jitter_m: float = 0.08) -> float:
        """Worst-case disagreement between the fan under two registrations.

        Worst case over a jitter ball because a hypothesis carries its own
        spread and a raycast flips exactly at a slab edge -- Batch 2 measured a
        planner betting a branch commit on the third decimal of a mode estimate.
        """
        here = self.predict(plan, delta_a)
        worst = 1.0
        for shift in (-jitter_m, 0.0, jitter_m):
            there = self.predict(plan, delta_b + shift)
            worst = min(worst, max((ray_disagreement(x, y)
                                    for x, y in zip(here, there)), default=0.0))
        return worst

    def to_dict(self) -> dict:
        return {"name": self.name, "leg": self.leg, "s_m": self.s_m,
                "xy_map": list(self.xy_map), "purpose": self.purpose,
                "fan_offsets_m": list(self.fan_offsets_m)}
