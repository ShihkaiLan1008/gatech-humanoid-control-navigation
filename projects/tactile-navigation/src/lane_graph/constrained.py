"""Distance-and-rotation planning under stepwise SE(2) constraints.

This is the global planning model.  It does not execute a physical press.  A
TOUCH_FOLLOW edge merely says that, at a safe standoff, a conceptual tactile
observation is available after each planning step:

* the belief is a 3x3 (x, y, yaw) covariance; motion adds tunable x/y and yaw
  process noise and propagates yaw error into lateral position through the
  motion Jacobian, so a persistent heading error grows position uncertainty
  with distance squared and yaw becomes correlated with position;
* a wall observation reduces x/y covariance only along the wall normal and
  reduces yaw variance;
* a landmark observation reduces both x/y axes and yaw variance;
* the pre-contact and post-contact state of every step satisfies both bounds;
* a start outside the bounds must localize first: until the belief first
  satisfies both bounds the effective limit is the initial sigma plus a
  configured grace, and the goal is accepted only once localized;
* the objective among feasible routes is translation distance plus a tunable
  metre-equivalent charge for commanded rotation.

If the strict problem is infeasible, both uncertainty bounds are relaxed by a
configured factor and the search is repeated.  The relaxation level is part of
the result so a caller can never mistake a relaxed plan for a strict one.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from global_nav.geometry import raycast

from .uncertainty import UncertaintyConfig, bin_key, floor_axes
from .uncertainty import DEFAULT as UNCERTAINTY

__all__ = [
    "ConstrainedPlan", "position_sigma_max", "plan_constrained",
    "replay_path", "belief_step", "trace_row", "pos_block", "to_se2",
    "PlannerBelief", "motion_position_sigma",
]


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def pos_block(S) -> np.ndarray:
    """The x/y block of a 2x2 or 3x3 (x, y, yaw) covariance."""
    S = np.asarray(S, float)
    return S[:2, :2] if S.shape == (3, 3) else S.reshape(2, 2)


def to_se2(cov2, yaw_var: float) -> np.ndarray:
    """Block-diagonal 3x3 (x, y, yaw) covariance from an uncorrelated start."""
    S = np.zeros((3, 3))
    S[:2, :2] = np.asarray(cov2, float).reshape(2, 2)
    S[2, 2] = float(yaw_var)
    return S


def position_sigma_max(P) -> float:
    """Largest directional one-sigma position uncertainty."""
    B = pos_block(P)
    w = np.linalg.eigvalsh(0.5 * (B + B.T))
    return float(math.sqrt(max(float(w[-1]), 0.0)))


def _kalman(S, H, R: float) -> np.ndarray:
    """Joseph-free covariance update for one scalar-per-row measurement block."""
    S = np.asarray(S, float)
    H = np.atleast_2d(np.asarray(H, float))
    inn = H @ S @ H.T + np.eye(H.shape[0]) * float(R)
    K = S @ H.T @ np.linalg.inv(inn)
    S2 = S - K @ H @ S
    return 0.5 * (S2 + S2.T)


def _floor_se2(S, min_axis_sigma: float, min_yaw_sigma: float) -> np.ndarray:
    """Inflate any position axis or the yaw variance that fell below its floor."""
    S = np.asarray(S, float).copy()
    B = 0.5 * (S[:2, :2] + S[:2, :2].T)
    w, V = np.linalg.eigh(B)
    for k in range(2):
        if w[k] < min_axis_sigma ** 2:
            S[:2, :2] += (min_axis_sigma ** 2 - w[k]) * np.outer(V[:, k], V[:, k])
    if S[2, 2] < min_yaw_sigma ** 2:
        S[2, 2] = min_yaw_sigma ** 2
    return S


def _fuse_position(P, g, sigma_obs: float):
    P = 0.5 * (np.asarray(P, float) + np.asarray(P, float).T)
    g = np.asarray(g, float).reshape(2)
    n = float(np.linalg.norm(g))
    if n < 1e-12:
        return P
    g = g / n
    Pg = P @ g
    den = float(g @ Pg) + float(sigma_obs) ** 2
    return 0.5 * ((P - np.outer(Pg, Pg) / den)
                  + (P - np.outer(Pg, Pg) / den).T)


def _fuse_yaw(var: float, sigma_obs: float, floor_sigma: float) -> float:
    obs_var = float(sigma_obs) ** 2
    if var <= 0.0:
        return float(floor_sigma) ** 2
    post = 1.0 / (1.0 / var + 1.0 / obs_var)
    return max(post, float(floor_sigma) ** 2)


@dataclass
class _State:
    P: np.ndarray
    """3x3 (x, y, yaw) covariance; ``yaw_var`` mirrors ``P[2, 2]``."""
    yaw_var: float
    heading: float | None
    resolved: bool
    localized: bool = True
    """Both bounds have been satisfied at least once after a step."""


def _pre_limits(belief, cfg: UncertaintyConfig, pos_limit: float, yaw_limit: float):
    """(initially localized, pre-localization position limit, pre-localization yaw limit).

    Nobody starts localized: the flag is earned by the first step whose
    post-observation belief satisfies both bounds.  Until then the limit is
    ``max(bound, initial sigma + grace)``, which equals the bound itself for a
    start comfortably inside it, so such starts behave exactly as before; a
    start at or just inside the bound gets the same small allowance as one
    outside it instead of failing on its first step of growth.
    """
    sigma0 = position_sigma_max(np.asarray(belief.cov, float).reshape(2, 2))
    yaw0 = float(getattr(belief, "yaw_sigma", 0.0))
    # An uncertain along-track axis must not grant cross-track grace.
    pre_pos = (pos_limit if cfg.position_constraint == "cross_track"
               else max(pos_limit, sigma0 + cfg.initial_position_grace_m))
    return (False, pre_pos,
            max(yaw_limit, yaw0 + cfg.initial_yaw_grace_rad))


@dataclass
class PlannerBelief:
    """The Gaussian summary both planners start from: a 2x2 position
    covariance, a yaw sigma, and the mass outside the leading registration
    hypothesis.  graph3.Belief3 also satisfies this interface where it
    carries a covariance."""
    cov: np.ndarray
    yaw_sigma: float = math.radians(3.0)
    minor: float = 0.0

    def __post_init__(self):
        self.cov = np.asarray(self.cov, float).reshape(2, 2)


@dataclass
class ConstrainedPlan:
    path: list[int] | None
    distance_m: float
    objective_cost: float
    feasible: bool
    relaxation_level: int
    position_threshold_m: float
    yaw_threshold_rad: float
    explored_states: int
    trace: list[dict] = field(default_factory=list)
    search_trace: list[dict] = field(default_factory=list)
    reason: str = ""

    @property
    def relaxed(self) -> bool:
        return self.relaxation_level > 0


def _polyline(g, e, entry: int):
    if e.poly is None:
        P = np.asarray([g.nodes[entry].xy,
                        g.nodes[e.b if e.a == entry else e.a].xy], float)
        N = None
    else:
        P = np.asarray(e.poly, float)
        N = None if e.normal is None else np.asarray(e.normal, float)
        if entry == e.b:
            P = P[::-1]
            if N is not None:
                N = N[::-1]
    return P, N


def _interp(P, s, at):
    return np.array([np.interp(at, s, P[:, 0]), np.interp(at, s, P[:, 1])])


def _edge_steps(g, e, entry: int, cfg: UncertaintyConfig):
    """Yield one record per configured planning step along an edge."""
    P, N = _polyline(g, e, entry)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] <= 1e-12:
        return
    marks = np.arange(0.0, s[-1], cfg.planning_step_m)
    if not len(marks) or marks[0] != 0.0:
        marks = np.r_[0.0, marks]
    if marks[-1] < s[-1] - 1e-12:
        marks = np.r_[marks, s[-1]]

    for a, b in zip(marks[:-1], marks[1:]):
        p0, p1 = _interp(P, s, a), _interp(P, s, b)
        d = p1 - p0
        ds = float(np.linalg.norm(d))
        if ds <= 1e-12:
            continue
        tangent = d / ds
        normal = None
        if N is not None:
            nx = np.interp(b, s, N[:, 0])
            ny = np.interp(b, s, N[:, 1])
            normal = np.array([nx, ny], float)
            nn = float(np.linalg.norm(normal))
            normal = normal / nn if nn > 1e-12 else None
        commanded_yaw = (math.atan2(normal[1], normal[0])
                         if e.kind == "TOUCH_FOLLOW" and normal is not None
                         else math.atan2(tangent[1], tangent[0]))
        yield p1, ds, tangent, normal, commanded_yaw


def _conceptual_contact(g, point, normal, cfg: UncertaintyConfig):
    """Return ``(wall|landmark|none, normal, standoff)`` at one route step."""
    if normal is None or g.scene is None:
        return "none", normal, None
    bodies = tuple(g.scene.plan.bodies)
    origin = np.asarray(point, float) + cfg.conceptual_reach_from_lane_m * normal
    cast = cfg.safe_standoff_m + cfg.safe_standoff_tolerance_m + 0.25
    distance, hit_normal, hit_i = raycast(origin, normal, cast, bodies)
    if distance is None:
        return "none", normal, None
    distance = float(distance)
    if abs(distance - cfg.safe_standoff_m) > cfg.safe_standoff_tolerance_m:
        return "none", normal, distance

    body = bodies[int(hit_i)]
    name = str(getattr(body, "name", ""))
    kind = "wall" if name.startswith("wall") else "landmark"

    # A named map feature at the hit point is also a landmark even if its
    # collision body was compiled as a wall slab.
    hit = origin + distance * normal
    for feature in getattr(g.scene.plan, "features", ()):
        if float(np.linalg.norm(hit - np.asarray(feature.position, float))) <= 0.25:
            kind = "landmark"
            break
    n = np.asarray(hit_normal if hit_normal is not None else normal, float)
    nn = float(np.linalg.norm(n))
    return kind, (n / nn if nn > 1e-12 else normal), distance


def motion_position_sigma(P, tangent, mode="cross_track") -> float:
    """One-sigma position error across the commanded translation (metres)."""
    if mode == "max_axis":
        return position_sigma_max(P)
    if mode != "cross_track":
        raise ValueError(f"unknown position constraint: {mode}")
    d = np.asarray(tangent, float).reshape(2)
    length = float(np.linalg.norm(d))
    if length < 1e-12:
        raise ValueError("cross-track uncertainty needs a translation direction")
    n = np.array([-d[1], d[0]]) / length
    return math.sqrt(max(float(n @ pos_block(P) @ n), 0.0))


def belief_step(P, yaw_var, heading, resolved, localized, ds, tangent, commanded_yaw,
                contact, contact_normal, cfg: UncertaintyConfig,
                pos_limit: float, yaw_limit: float,
                pre_pos_limit: float | None = None, pre_yaw_limit: float | None = None,
                constraint_tangent=None):
    """One planning step of the shared belief model.

    Grows the belief by ``ds`` of motion along ``tangent`` with the commanded
    heading, checks the bounds, credits ``contact`` (``wall``/``landmark``/
    ``none``) along ``contact_normal``, checks again and updates the
    localized flag.  Returns ``(P, yaw_var, heading, resolved, localized, info)``
    where ``info`` carries every number a trace row or a search needs.  The
    roadmap planner and the grid planner both call this, so they cannot drift
    apart in their arithmetic.
    """
    S = np.asarray(P, float)
    if S.shape != (3, 3):
        S = to_se2(S, yaw_var)
    S = S.copy()
    # Feedback may restore a route's transverse error without changing its
    # travel direction. Propagation still uses the ACTUAL displacement below.
    reference = tangent if constraint_tangent is None else constraint_tangent
    source_pos_sigma = motion_position_sigma(S, reference, cfg.position_constraint)
    pre_pos_limit = pos_limit if pre_pos_limit is None else max(pos_limit, pre_pos_limit)
    pre_yaw_limit = yaw_limit if pre_yaw_limit is None else max(yaw_limit, pre_yaw_limit)
    frac = ds / max(cfg.planning_step_m, 1e-9)

    # The turn happens first: heading drift proportional to the commanded
    # rotation, plus the per-step yaw process noise.
    turn = 0.0 if heading is None else abs(_wrap(commanded_yaw - heading))
    q_yaw = cfg.yaw_growth_per_step_rad * math.sqrt(frac)
    S[2, 2] += q_yaw * q_yaw + (cfg.drift_yaw_per_rad * turn) ** 2
    heading = commanded_yaw

    # Then the motion: a persistent heading error displaces the robot
    # laterally by ds * error, so yaw and position become correlated.  The
    # Jacobian carries that correlation forward; over a long straight the
    # lateral variance therefore grows with distance squared, as it should.
    F = np.array([[1.0, 0.0, -ds * tangent[1]],
                  [0.0, 1.0, ds * tangent[0]],
                  [0.0, 0.0, 1.0]])
    q = cfg.position_growth_per_step_m * math.sqrt(frac)
    S = F @ S @ F.T + np.diag([q * q, q * q, 0.0])
    yaw_var = float(S[2, 2])

    # Until the belief has satisfied both bounds once, the looser
    # localize-first limits apply (identical to the bounds for a start
    # that is already inside them).
    eff_pos = pos_limit if localized else pre_pos_limit
    eff_yaw = yaw_limit if localized else pre_yaw_limit
    # The bound is a maximum over the whole step, not merely the belief
    # after a convenient observation.  Check the motion endpoint before
    # crediting the conceptual touch, then check again afterwards.
    pre_pos_sigma = motion_position_sigma(S, reference, cfg.position_constraint)
    pre_yaw_sigma = math.sqrt(max(yaw_var, 0.0))
    pre_ok = (max(source_pos_sigma, pre_pos_sigma) <= eff_pos + 1e-12
              and pre_yaw_sigma <= eff_yaw + 1e-12)
    S_pre = S.copy()

    if contact == "wall":
        # One wall: range along the normal, and the surface orientation as a
        # yaw observation.  Through the yaw/position correlation the range
        # measurement alone already tightens yaw.
        n = np.asarray(contact_normal, float).reshape(2)
        n = n / max(float(np.linalg.norm(n)), 1e-12)
        S = _kalman(S, [n[0], n[1], 0.0], cfg.wall_position_obs_sigma_m ** 2)
        S = _kalman(S, [0.0, 0.0, 1.0], cfg.wall_yaw_obs_sigma_rad ** 2)
    elif contact == "landmark":
        if cfg.landmark_two_axes or contact_normal is None:
            # A landmark anchors both map axes and yaw.
            S = _kalman(S, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], cfg.landmark_position_obs_sigma_m ** 2)
        else:
            # A point contact on the landmark's surface: range along its
            # local normal; a second contact with a different normal gives
            # the other axis.
            n = np.asarray(contact_normal, float).reshape(2)
            n = n / max(float(np.linalg.norm(n)), 1e-12)
            S = _kalman(S, [n[0], n[1], 0.0], cfg.landmark_position_obs_sigma_m ** 2)
        S = _kalman(S, [0.0, 0.0, 1.0], cfg.landmark_yaw_obs_sigma_rad ** 2)
        resolved = True

    S = _floor_se2(S, cfg.min_axis_sigma_m, cfg.min_yaw_sigma_rad)
    P = S
    yaw_var = float(S[2, 2])
    pos_sigma = motion_position_sigma(S, reference, cfg.position_constraint)
    yaw_sigma = math.sqrt(max(yaw_var, 0.0))
    ok = (pre_ok and pos_sigma <= eff_pos + 1e-12
          and yaw_sigma <= eff_yaw + 1e-12)
    if (not localized and pos_sigma <= pos_limit + 1e-12
            and yaw_sigma <= yaw_limit + 1e-12):
        localized = True
    info = dict(turn=turn, pre_pos_sigma=pre_pos_sigma, pre_yaw_sigma=pre_yaw_sigma,
                pos_sigma=pos_sigma, yaw_sigma=yaw_sigma,
                source_pos_sigma=source_pos_sigma,
                position_constraint=cfg.position_constraint,
                motion_tangent=[float(v) for v in tangent],
                constraint_tangent=[float(v) for v in reference],
                max_axis_sigma=position_sigma_max(S),
                max_pos_sigma=max(source_pos_sigma, pre_pos_sigma, pos_sigma),
                max_yaw_sigma=max(pre_yaw_sigma, yaw_sigma),
                eff_pos=eff_pos, eff_yaw=eff_yaw, ok=ok, S_pre=S_pre)
    return P, yaw_var, heading, resolved, localized, info


def trace_row(P, heading, info, *, edge, edge_kind, step, distance, point, contact,
              standoff, localized, pos_limit, yaw_limit):
    """The trace record both planners emit per step."""
    return {
        "edge": int(edge), "edge_kind": edge_kind, "step": int(step),
        "distance_m": float(distance), "x": float(point[0]), "y": float(point[1]),
        "position_sigma_m": info["pos_sigma"], "heading_rad": float(heading),
        "position_constraint": info["position_constraint"],
        "motion_tangent": info["motion_tangent"],
        "source_position_sigma_m": info["source_pos_sigma"],
        "max_axis_sigma_m": info["max_axis_sigma"],
        "turn_rad": float(info["turn"]),
        "position_sigma_x_m": math.sqrt(max(float(P[0, 0]), 0.0)),
        "position_sigma_y_m": math.sqrt(max(float(P[1, 1]), 0.0)),
        "yaw_sigma_rad": info["yaw_sigma"], "yaw_sigma_deg": math.degrees(info["yaw_sigma"]),
        "pre_contact_position_sigma_m": info["pre_pos_sigma"],
        "pre_contact_yaw_sigma_rad": info["pre_yaw_sigma"],
        "max_position_sigma_m": info["max_pos_sigma"],
        "max_yaw_sigma_rad": info["max_yaw_sigma"],
        "max_yaw_sigma_deg": math.degrees(info["max_yaw_sigma"]),
        "cov": [[float(v) for v in row] for row in pos_block(P)],
        "cov_pre": [[float(v) for v in row] for row in pos_block(info["S_pre"])],
        "cov3": [[float(v) for v in row] for row in np.asarray(P, float)] if np.asarray(P).shape == (3, 3) else None,
        "yaw_position_corr": (float(P[2, 1]) / max(math.sqrt(max(float(P[1, 1]) * float(P[2, 2]), 0.0)), 1e-12)
                              if np.asarray(P).shape == (3, 3) else 0.0),
        "position_threshold_m": pos_limit, "yaw_threshold_rad": yaw_limit,
        "position_limit_effective_m": info["eff_pos"],
        "yaw_limit_effective_rad": info["eff_yaw"],
        "localized": bool(localized), "contact": contact, "standoff_m": standoff,
        "feasible": bool(info["ok"]),
    }


def _advance(g, e, entry: int, state: _State, cfg: UncertaintyConfig,
             pos_limit: float, yaw_limit: float, collect: bool = False,
             distance0: float = 0.0, contact_steps: frozenset[int] | None = None,
             pre_pos_limit: float | None = None, pre_yaw_limit: float | None = None):
    P = np.asarray(state.P, float).copy()
    yaw_var = float(state.yaw_var)
    heading = state.heading
    resolved = bool(state.resolved)
    localized = bool(state.localized)
    rows = []
    distance = float(distance0)
    turn_total = 0.0

    for step_i, (point, ds, tangent, normal, commanded_yaw) in enumerate(
            _edge_steps(g, e, entry, cfg), start=1):
        contact, contact_normal, standoff = ("none", normal, None)
        if e.kind == "TOUCH_FOLLOW" and (contact_steps is None or step_i in contact_steps):
            contact, contact_normal, standoff = _conceptual_contact(g, point, normal, cfg)
        P, yaw_var, heading, resolved, localized, info = belief_step(
            P, yaw_var, heading, resolved, localized, ds, tangent, commanded_yaw,
            contact, contact_normal, cfg, pos_limit, yaw_limit, pre_pos_limit, pre_yaw_limit)
        turn_total += info["turn"]
        distance += ds
        if collect:
            rows.append(trace_row(P, heading, info, edge=e.eid, edge_kind=e.kind, step=step_i,
                                  distance=distance, point=point, contact=contact,
                                  standoff=standoff, localized=localized,
                                  pos_limit=pos_limit, yaw_limit=yaw_limit))
        if not info["ok"]:
            return None, rows, turn_total

    return _State(P, yaw_var, heading, resolved, localized), rows, turn_total


def _state_key(nid: int, state: _State, cfg: UncertaintyConfig):
    yaw_sigma = math.sqrt(max(state.yaw_var, 0.0))
    yaw_bin = min(int(yaw_sigma / max(cfg.yaw_bin_rad, 1e-9)),
                  cfg.max_yaw_bins)
    if state.heading is None:
        heading_bin = -1
    else:
        n = max(int(cfg.heading_bins), 1)
        heading_bin = int((state.heading % (2.0 * math.pi))
                          / (2.0 * math.pi) * n) % n
    return (nid, bin_key(pos_block(state.P), cfg), yaw_bin, heading_bin, state.resolved,
            state.localized)


def _search(g, src: int, dst: int, belief, cfg: UncertaintyConfig,
            pos_limit: float, yaw_limit: float, yaw0: float | None):
    yaw_sigma = float(getattr(belief, "yaw_sigma", 0.0))
    localized0, pre_pos, pre_yaw = _pre_limits(belief, cfg, pos_limit, yaw_limit)
    state0 = _State(to_se2(belief.cov, yaw_sigma * yaw_sigma),
                    yaw_sigma * yaw_sigma,
                    None if yaw0 is None else float(yaw0),
                    float(getattr(belief, "minor", 0.0)) <= 0.10,
                    localized0)

    tie = 0
    goal_xy = np.asarray(g.nodes[dst].xy, float)

    def heuristic(nid):
        # Translation-only lower bound.  Ignoring turns, uncertainty and
        # obstacles can only make it smaller, so it is admissible.
        return float(np.linalg.norm(np.asarray(g.nodes[nid].xy, float) - goal_xy))

    pq = [(heuristic(src), 0.0, 0.0, tie, src, state0, ())]
    best = {}
    explored = 0
    search_trace = []
    while pq:
        _f, objective, distance, _, nid, state, path = heapq.heappop(pq)
        key = _state_key(nid, state, cfg)
        if best.get(key, math.inf) <= objective:
            continue
        best[key] = objective
        explored += 1
        if cfg.max_explored_states and explored > cfg.max_explored_states:
            return None, math.inf, math.inf, explored, search_trace
        h = heuristic(nid)
        search_trace.append({
            "node": int(nid), "path": [int(eid) for eid in path],
            "objective_cost": float(objective), "distance_m": float(distance),
            "heuristic_m": float(h), "estimated_total_cost": float(objective + h),
            "position_sigma_m": position_sigma_max(state.P),
            "position_sigma_x_m": math.sqrt(
                max(float(state.P[0, 0]), 0.0)),
            "position_sigma_y_m": math.sqrt(
                max(float(state.P[1, 1]), 0.0)),
            "yaw_sigma_rad": math.sqrt(max(state.yaw_var, 0.0)),
            "resolved": bool(state.resolved),
            "localized": bool(state.localized),
        })
        if nid == dst and state.localized and (
                state.resolved or not cfg.require_landmark_for_mode_resolution):
            return list(path), distance, objective, explored, search_trace

        for eid in g.adj[nid]:
            # An immediate edge reversal is a zero-progress sensing manoeuvre,
            # not a global route.  Without this guard the search can traverse
            # a tactile edge out-and-back only to count the same conceptual
            # observation twice.  Deliberate local re-observation belongs in
            # the execution/local-planning layer.
            if path and eid == path[-1]:
                continue
            e = g.edges[eid]
            nxt = e.b if e.a == nid else e.a
            advanced, _rows, turn = _advance(
                g, e, nid, state, cfg, pos_limit, yaw_limit,
                pre_pos_limit=pre_pos, pre_yaw_limit=pre_yaw)
            if advanced is None:
                continue
            # Feasibility was decided above.  The objective cannot soften it.
            step_cost = e.length + cfg.turn_cost_per_rad * turn
            objective2 = objective + step_cost
            distance2 = distance + e.length
            tie += 1
            heapq.heappush(pq, (objective2 + heuristic(nxt), objective2,
                                distance2, tie, nxt, advanced, path + (eid,)))
    return None, math.inf, math.inf, explored, search_trace


def replay_path(g, path, belief, cfg: UncertaintyConfig | None = None,
                src: int | None = None, yaw0: float | None = None,
                position_threshold_m: float = math.inf,
                yaw_threshold_rad: float = math.inf,
                probe_schedule: frozenset[tuple[int, int]] | None = None):
    """Replay uncertainty; optional probes are (path-edge index, step index).

    None retains dense conceptual sensing. An empty set disables all probes.
    Edge occurrence indices distinguish multiple traversals of the same edge.
    """
    cfg = cfg or UNCERTAINTY
    if src is None:
        return []
    localized0, pre_pos, pre_yaw = _pre_limits(
        belief, cfg, position_threshold_m, yaw_threshold_rad)
    state = _State(to_se2(belief.cov, float(getattr(belief, "yaw_sigma", 0.0)) ** 2),
                   float(getattr(belief, "yaw_sigma", 0.0)) ** 2,
                   yaw0, float(getattr(belief, "minor", 0.0)) <= 0.10,
                   localized0)
    rows, nid, distance = [], int(src), 0.0
    for edge_index, eid in enumerate(path or ()):
        e = g.edges[eid]
        state2, edge_rows, _turn = _advance(
            g, e, nid, state, cfg, position_threshold_m, yaw_threshold_rad,
            collect=True, distance0=distance,
            pre_pos_limit=pre_pos, pre_yaw_limit=pre_yaw,
            contact_steps=(None if probe_schedule is None else frozenset(
                step for index, step in probe_schedule if index == edge_index)))
        for row in edge_rows:
            row["edge_index"] = edge_index
        rows.extend(edge_rows)
        if edge_rows:
            distance = edge_rows[-1]["distance_m"]
        if state2 is None:
            break
        state = state2
        nid = e.b if e.a == nid else e.a
    return rows


def plan_constrained(g, src: int, dst: int, belief,
                     cfg: UncertaintyConfig | None = None,
                     yaw0: float | None = None) -> ConstrainedPlan:
    """Minimize distance plus rotation subject to per-step SE(2) bounds."""
    cfg = cfg or UNCERTAINTY
    if cfg.planning_step_m <= 0.0:
        raise ValueError("planning_step_m must be positive")
    if cfg.position_threshold_m <= 0.0 or cfg.yaw_threshold_rad <= 0.0:
        raise ValueError("uncertainty thresholds must be positive")
    pos_relax = (cfg.relaxation_factor
                 if cfg.position_relaxation_factor is None
                 else cfg.position_relaxation_factor)
    yaw_relax = (cfg.relaxation_factor
                 if cfg.yaw_relaxation_factor is None
                 else cfg.yaw_relaxation_factor)
    if pos_relax < 1.0 or yaw_relax < 1.0:
        raise ValueError("position/yaw relaxation factors must be >= 1")

    total_explored = 0
    search_history = []
    for level in range(max(0, int(cfg.max_relaxations)) + 1):
        pos_limit = cfg.position_threshold_m * pos_relax ** level
        yaw_limit = cfg.yaw_threshold_rad * yaw_relax ** level
        path, distance, objective, explored, attempt_trace = _search(
            g, src, dst, belief, cfg, pos_limit, yaw_limit, yaw0)
        for row in attempt_trace:
            row["relaxation_level"] = level
            row["position_threshold_m"] = pos_limit
            row["yaw_threshold_rad"] = yaw_limit
        search_history.extend(attempt_trace)
        total_explored += explored
        if path is not None:
            trace = replay_path(g, path, belief, cfg, src, yaw0,
                                pos_limit, yaw_limit)
            return ConstrainedPlan(
                path=path, distance_m=float(distance),
                objective_cost=float(objective), feasible=True,
                relaxation_level=level, position_threshold_m=pos_limit,
                yaw_threshold_rad=yaw_limit, explored_states=total_explored,
                trace=trace, search_trace=search_history,
                reason=("strict" if level == 0
                        else "relaxed uncertainty bounds"))

    last_level = max(0, int(cfg.max_relaxations))
    return ConstrainedPlan(
        path=None, distance_m=math.inf, objective_cost=math.inf,
        feasible=False, relaxation_level=int(cfg.max_relaxations),
        position_threshold_m=cfg.position_threshold_m * pos_relax ** last_level,
        yaw_threshold_rad=cfg.yaw_threshold_rad * yaw_relax ** last_level,
        explored_states=total_explored, trace=[], search_trace=search_history,
        reason="no route satisfies even the maximally relaxed bounds")
