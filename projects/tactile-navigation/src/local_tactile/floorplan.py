"""Known floorplan, map registration, and route state.

Pure geometry and decision logic — no torch, no simulator, no privileged
simulator state.  This module is what "the known floorplan and route state
should determine whether to continue, turn, or probe again" means concretely:

* the map's SHAPE is exact — edge tangents, standoffs, which edge succeeds
  which, and therefore the signed turn angle at every junction;
* the map's REGISTRATION along the edge currently being followed is NOT — the
  along-edge coordinate of each wall terminus is a Gaussian prior.  Closing
  that prior is the entire job of the tactile probes.

Nothing here knows the word "left".  The turn a junction requires is
``wrap(facing_yaw(next_edge) - facing_yaw(this_edge))``, read off the map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# 95 % gate of a chi-squared with one degree of freedom.
CHI2_1DOF_95 = 3.841459


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _unit(vector: Any) -> np.ndarray:
    v = np.asarray(vector, dtype=float).reshape(2)
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        raise ValueError("direction vector is degenerate")
    return v / norm


@dataclass(frozen=True)
class WallEdge:
    """One followed wall face and the robot path that parallels it.

    ``s`` is the along-edge coordinate of the ROBOT PATH, zero at ``origin_xy``
    and increasing along ``tangent_xy`` (the direction of travel).  The wall
    face sits ``standoff_m`` away along ``normal_xy``; the robot faces the wall,
    so its commanded heading on this edge is ``facing_yaw``.
    """

    index: int
    origin_xy: tuple[float, float]
    tangent_xy: tuple[float, float]
    normal_xy: tuple[float, float]
    standoff_m: float
    material_start_m: float
    material_end_m: float
    name: str = ""

    @property
    def tangent(self) -> np.ndarray:
        return _unit(self.tangent_xy)

    @property
    def normal(self) -> np.ndarray:
        return _unit(self.normal_xy)

    @property
    def origin(self) -> np.ndarray:
        return np.asarray(self.origin_xy, dtype=float).reshape(2)

    @property
    def facing_yaw(self) -> float:
        n = self.normal
        return math.atan2(float(n[1]), float(n[0]))

    def path_point(self, s_m: float) -> np.ndarray:
        return self.origin + float(s_m) * self.tangent

    def face_point(self, s_m: float) -> np.ndarray:
        return self.path_point(s_m) + self.standoff_m * self.normal

    def path_coordinate(self, point_xy: Any) -> float:
        p = np.asarray(point_xy, dtype=float).reshape(2)
        return float(np.dot(p - self.origin, self.tangent))

    def lateral_coordinate(self, point_xy: Any) -> float:
        """Signed distance of a point from the path line, positive toward the
        wall.  ``standoff_m`` is the value at the face."""
        p = np.asarray(point_xy, dtype=float).reshape(2)
        return float(np.dot(p - self.origin, self.normal))


@dataclass(frozen=True)
class Vertex:
    """A wall terminus in the map.

    ``to_edge`` is the successor edge of the route when this terminus is a
    junction, and ``None`` when it is a dead end (a terminus the robot may
    legitimately measure but must not turn at).
    """

    index: int
    edge: int
    s_m: float
    to_edge: int | None
    name: str = ""


@dataclass(frozen=True)
class Floorplan:
    """Rigid map.  World = map translated by ``delta`` along ``shift_axis``."""

    edges: tuple[WallEdge, ...]
    vertices: tuple[Vertex, ...]

    def __post_init__(self) -> None:
        for position, edge in enumerate(self.edges):
            if edge.index != position:
                raise ValueError("edges must be indexed 0..N-1 in order")
        for vertex in self.vertices:
            if not 0 <= vertex.edge < len(self.edges):
                raise ValueError(f"vertex {vertex.index} references an unknown edge")
            if vertex.to_edge is not None and not 0 <= vertex.to_edge < len(self.edges):
                raise ValueError(f"vertex {vertex.index} references an unknown successor")

    @property
    def shift_axis(self) -> np.ndarray:
        """Direction of the unknown map-to-odometry registration offset."""
        return self.edges[0].tangent

    def registration_gain(self, edge_index: int) -> float:
        """How much the registration offset moves ALONG this edge.

        The offset is a rigid translation of the map, so it shifts an edge's
        arclength only by its component on that edge's tangent: 1 on the edge
        the offset is defined along, 0 on a perpendicular edge (there it moves
        the path line sideways instead).
        """
        return float(np.dot(self.shift_axis, self.edge(edge_index).tangent))

    def edge(self, index: int) -> WallEdge:
        return self.edges[index]

    def vertex(self, index: int) -> Vertex:
        for vertex in self.vertices:
            if vertex.index == index:
                return vertex
        raise KeyError(index)

    def vertices_on(self, edge_index: int) -> tuple[Vertex, ...]:
        return tuple(v for v in self.vertices if v.edge == edge_index)

    def turn_angle_rad(self, vertex_index: int) -> float:
        """Signed yaw change the map requires at a junction.  Sign comes from
        the two edges' facing yaws — it is never assumed."""
        vertex = self.vertex(vertex_index)
        if vertex.to_edge is None:
            raise ValueError(f"vertex {vertex_index} is a dead end, not a junction")
        return wrap_angle(
            self.edge(vertex.to_edge).facing_yaw - self.edge(vertex.edge).facing_yaw
        )

    def turn_point(self, vertex_index: int, delta_m: float = 0.0) -> np.ndarray:
        """World point where the two path lines of a junction meet."""
        vertex = self.vertex(vertex_index)
        if vertex.to_edge is None:
            raise ValueError(f"vertex {vertex_index} is a dead end, not a junction")
        a = self.edge(vertex.edge)
        b = self.edge(vertex.to_edge)
        shift = float(delta_m) * self.shift_axis
        p, r = a.origin + shift, a.tangent
        q, s = b.origin + shift, b.tangent
        denom = float(r[0] * s[1] - r[1] * s[0])
        if abs(denom) < 1e-9:
            raise ValueError("junction edges are parallel; no unique turn point")
        d = q - p
        t = float(d[0] * s[1] - d[1] * s[0]) / denom
        return p + t * r

    def path_point_w(self, edge_index: int, s_m: float, delta_m: float = 0.0) -> np.ndarray:
        return self.edge(edge_index).path_point(s_m) + float(delta_m) * self.shift_axis

    def face_point_w(self, edge_index: int, s_m: float, delta_m: float = 0.0) -> np.ndarray:
        return self.edge(edge_index).face_point(s_m) + float(delta_m) * self.shift_axis


# ── association ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Association:
    status: str  # "associated" | "no_candidate" | "ambiguous"
    vertex_index: int | None
    mahalanobis2: float
    gated_vertices: tuple[int, ...]
    scores: tuple[tuple[int, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "vertex_index": self.vertex_index,
            "mahalanobis2": self.mahalanobis2,
            "gated_vertices": list(self.gated_vertices),
            "scores": [{"vertex": v, "mahalanobis2": d} for v, d in self.scores],
        }


def associate_terminus(
    plan: Floorplan,
    *,
    edge_index: int,
    s_hat_m: float,
    s_sigma_m: float,
    delta_m: float,
    delta_sigma_m: float,
    gate_chi2: float = CHI2_1DOF_95,
    ahead_of_s_m: float | None = None,
) -> Association:
    """Match a MEASURED wall terminus to a map vertex of the followed edge.

    The measurement and the map are both uncertain along the edge, so the gate
    is a one-dimensional Mahalanobis test on the along-edge coordinate.
    Candidates behind the robot are dropped: a terminus the robot has already
    walked past cannot be the one it is approaching.
    """
    scores: list[tuple[int, float]] = []
    gated: list[int] = []
    gain = plan.registration_gain(edge_index)
    variance_map = (float(delta_sigma_m) * gain) ** 2
    variance_obs = float(s_sigma_m) ** 2
    for vertex in plan.vertices_on(edge_index):
        s_world = vertex.s_m + gain * float(delta_m)
        if ahead_of_s_m is not None and s_world < float(ahead_of_s_m):
            continue
        d2 = (float(s_hat_m) - s_world) ** 2 / max(variance_map + variance_obs, 1e-12)
        scores.append((vertex.index, d2))
        if d2 <= gate_chi2:
            gated.append(vertex.index)
    scores.sort(key=lambda row: (row[1], row[0]))
    if len(gated) == 1:
        return Association("associated", gated[0], dict(scores)[gated[0]], tuple(gated), tuple(scores))
    if not gated:
        return Association("no_candidate", None, float("inf"), (), tuple(scores))
    return Association("ambiguous", None, dict(scores)[gated[0]], tuple(sorted(gated)), tuple(scores))


# ── route state ─────────────────────────────────────────────────────────────


class RouteDecision(str, Enum):
    CONTINUE_EDGE = "CONTINUE_EDGE"
    TURN = "TURN"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    ROUTE_FAILURE = "ROUTE_FAILURE"
    ASSOCIATION_FAILURE = "ASSOCIATION_FAILURE"


@dataclass(frozen=True)
class RouteOutcome:
    decision: RouteDecision
    reason: str
    vertex_index: int | None = None
    turn_angle_rad: float | None = None
    next_edge: int | None = None
    association: Association | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "vertex_index": self.vertex_index,
            "turn_angle_rad": self.turn_angle_rad,
            "turn_angle_deg": (
                None if self.turn_angle_rad is None else math.degrees(self.turn_angle_rad)
            ),
            "next_edge": self.next_edge,
            "association": None if self.association is None else self.association.to_dict(),
        }


@dataclass
class RouteState:
    """Which edge is being followed, and how the map is registered to odometry."""

    plan: Floorplan
    edge_index: int = 0
    delta_m: float = 0.0
    delta_sigma_m: float = 0.30
    visited_vertices: list[int] = field(default_factory=list)

    @property
    def edge(self) -> WallEdge:
        return self.plan.edge(self.edge_index)

    def route_vertex(self) -> Vertex | None:
        """The junction the route expects next on the current edge."""
        for vertex in self.plan.vertices_on(self.edge_index):
            if vertex.to_edge is not None:
                return vertex
        return None

    def expected_terminus_s_m(self) -> float:
        vertex = self.route_vertex()
        if vertex is None:
            return float("inf")
        return vertex.s_m + self.plan.registration_gain(self.edge_index) * self.delta_m

    def expected_terminus_sigma_m(self) -> float:
        return abs(self.plan.registration_gain(self.edge_index)) * self.delta_sigma_m

    def fuse_terminus(self, *, vertex_index: int, s_hat_m: float, s_sigma_m: float) -> dict[str, float]:
        """Re-register the map from a measured terminus (scalar Kalman step)."""
        vertex = self.plan.vertex(vertex_index)
        h = self.plan.registration_gain(vertex.edge)
        if abs(h) < 1e-9:
            raise ValueError("this vertex carries no information about the registration")
        prior_delta, prior_var = self.delta_m, self.delta_sigma_m**2
        innovation = float(s_hat_m) - (vertex.s_m + h * prior_delta)
        gain = prior_var * h / max(h * h * prior_var + float(s_sigma_m) ** 2, 1e-12)
        self.delta_m = prior_delta + gain * innovation
        self.delta_sigma_m = math.sqrt(max((1.0 - gain * h) * prior_var, 0.0))
        return {
            "innovation_m": innovation,
            "gain": gain,
            "delta_m": self.delta_m,
            "delta_sigma_m": self.delta_sigma_m,
        }

    def advance_to(self, vertex_index: int) -> None:
        vertex = self.plan.vertex(vertex_index)
        if vertex.to_edge is None:
            raise ValueError("cannot advance through a dead end")
        self.visited_vertices.append(vertex_index)
        self.edge_index = vertex.to_edge

    # ── decision ────────────────────────────────────────────────────────────

    def decide(
        self,
        belief: dict[str, Any],
        *,
        stance_s_m: float,
        pose_variance_tol_m2: float,
        gate_chi2: float = CHI2_1DOF_95,
    ) -> RouteOutcome:
        """Route decision from the ATE belief plus the map.

        ``belief`` is a :meth:`LocalGeometryATE.snapshot` dict.
        """
        if not belief["decided"]:
            return RouteOutcome(
                RouteDecision.NEED_MORE_EVIDENCE, "identity_belief_undecided"
            )
        if belief["decision"] == "CONTINUES":
            return RouteOutcome(
                RouteDecision.CONTINUE_EDGE, "wall_continues_past_probe_fan"
            )
        if belief["terminus_variance_m2"] > pose_variance_tol_m2:
            return RouteOutcome(
                RouteDecision.NEED_MORE_EVIDENCE, "terminus_pose_variance_above_tolerance"
            )
        s_hat = float(belief["terminus_mean_m"])
        s_sigma = math.sqrt(max(float(belief["terminus_variance_m2"]), 1e-12))
        association = associate_terminus(
            self.plan,
            edge_index=self.edge_index,
            s_hat_m=s_hat,
            s_sigma_m=s_sigma,
            delta_m=self.delta_m,
            delta_sigma_m=self.delta_sigma_m,
            gate_chi2=gate_chi2,
            ahead_of_s_m=float(stance_s_m),
        )
        if association.status == "no_candidate":
            return RouteOutcome(
                RouteDecision.ASSOCIATION_FAILURE,
                "measured_terminus_matches_no_map_vertex",
                association=association,
            )
        if association.status == "ambiguous":
            return RouteOutcome(
                RouteDecision.NEED_MORE_EVIDENCE,
                "measured_terminus_matches_several_map_vertices",
                association=association,
            )
        vertex = self.plan.vertex(int(association.vertex_index))
        if vertex.to_edge is None:
            return RouteOutcome(
                RouteDecision.ROUTE_FAILURE,
                "measured_terminus_is_a_dead_end_not_a_junction",
                vertex_index=vertex.index,
                association=association,
            )
        expected = self.route_vertex()
        if expected is None or expected.index != vertex.index:
            return RouteOutcome(
                RouteDecision.ROUTE_FAILURE,
                "associated_vertex_is_not_the_next_route_junction",
                vertex_index=vertex.index,
                association=association,
            )
        return RouteOutcome(
            RouteDecision.TURN,
            "map_junction_associated_from_realized_contacts",
            vertex_index=vertex.index,
            turn_angle_rad=self.plan.turn_angle_rad(vertex.index),
            next_edge=vertex.to_edge,
            association=association,
        )


# ── constructor for the validation scene ────────────────────────────────────


def l_corner_floorplan(
    *,
    origin_xy: Any,
    travel_xy: Any,
    facing_xy: Any,
    standoff_m: float,
    first_wall_half_length_m: float,
    second_wall_half_length_m: float,
    wall_thickness_m: float,
) -> Floorplan:
    """Two-edge convex 90-degree corner, expressed in the MAP frame.

    ``origin_xy``/``travel_xy``/``facing_xy`` describe edge 0's path: the robot
    starts at the origin, walks along ``travel``, and faces ``facing`` (the
    wall).  Edge 0's wall material ends at ``+first_wall_half_length_m``; the
    second wall stands off that terminus, perpendicular, on the far side, so
    the robot walks past the corner and turns to face back at it.

    The map's registration to odometry is NOT part of this geometry — it lives
    in :attr:`RouteState.delta_m`, whose non-zero prior is what makes the
    tactile probes necessary.
    """
    origin = np.asarray(origin_xy, dtype=float).reshape(2)
    u = _unit(travel_xy)
    v = _unit(facing_xy)
    if abs(float(np.dot(u, v))) > 1e-6:
        raise ValueError("travel and facing directions must be perpendicular")
    terminus_s = float(first_wall_half_length_m)

    edge0 = WallEdge(
        index=0,
        origin_xy=tuple(origin),
        tangent_xy=tuple(u),
        normal_xy=tuple(v),
        standoff_m=float(standoff_m),
        material_start_m=-float(first_wall_half_length_m),
        material_end_m=terminus_s,
        name="wall_0",
    )
    # Edge 1's face lies on the plane ``s = terminus_s``; the robot follows it
    # from the far side, so its path line sits one standoff beyond the corner
    # and its facing normal points back along -u.
    edge1_origin = origin + (terminus_s + float(standoff_m)) * u
    edge1 = WallEdge(
        index=1,
        origin_xy=tuple(edge1_origin),
        tangent_xy=tuple(v),
        normal_xy=tuple(-u),
        standoff_m=float(standoff_m),
        material_start_m=float(standoff_m) + float(wall_thickness_m) / 2.0,
        material_end_m=(
            float(standoff_m)
            + float(wall_thickness_m) / 2.0
            + 2.0 * float(second_wall_half_length_m)
        ),
        name="wall_1",
    )
    vertices = (
        Vertex(index=0, edge=0, s_m=-float(first_wall_half_length_m), to_edge=None, name="wall_0_start"),
        Vertex(index=1, edge=0, s_m=terminus_s, to_edge=1, name="corner"),
        Vertex(
            index=2,
            edge=1,
            s_m=edge1.material_end_m,
            to_edge=None,
            name="wall_1_end",
        ),
    )
    return Floorplan(edges=(edge0, edge1), vertices=vertices)
