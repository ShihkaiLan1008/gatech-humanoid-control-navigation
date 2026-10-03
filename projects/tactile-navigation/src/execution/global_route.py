"""Global grid route -> PF-based route follower -> single-touch EIG.

By default the planner minimizes travel outside contact reach before
distance/rotation; ``route_objective=distance_and_rotation`` selects the
distance/rotation objective under the same bounds.
Dense contacts are the default prediction schedule; pruning is explicit opt-in.
Execution keeps the route fixed and chooses touches by EIG minus time cost.
A scheduled contact never bypasses that gate. Predicted contact updates are
planning assumptions, not guarantees of successful execution observations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from lane_graph.constrained import PlannerBelief, belief_step, motion_position_sigma, position_sigma_max
from lane_graph.grid_constrained import (GridBackend, build_grid, commanded_heading,
                                         plan_grid)
from lane_graph.tradeoff import (Timing, evaluate_route, feasible, prune_schedule,
                                 prune_schedule_balanced)
from lane_graph.uncertainty import DEFAULT, UncertaintyConfig, parse_overrides

from .punch_eig import spec_at, arm_geometry, TRAVEL_M
from .punch_eig_policy import Segment, _wrap
from .punch_eig_step import ClearedArmEIG, LAMBDA_TIME
from .punch_rules import TactileIO
from .route_guard import BeliefRouteGuard, local_connector

__all__ = ["RouteLeg", "PlanStation", "GlobalRoute", "filter_consistent_config",
           "plan_global", "legs_from_path", "route_record", "plot_route",
           "SingleTouchSelector", "RouteFollowPolicy"]


@dataclass
class RouteLeg(Segment):
    """A straight leg of the global route.  ``yaw_cmd`` is the heading the
    base holds on it; the local policy reads it through ``yaw``."""
    yaw_cmd: float = 0.0
    touch: bool = False
    normal: tuple | None = None

    @property
    def yaw(self) -> float:            # overrides Segment.yaw (face the wall)
        return self.yaw_cmd


@dataclass
class PlanStation:
    move_index: int
    leg: int
    s: float                           # arclength within the leg
    g: float                           # cumulative route arclength
    xy: tuple
    kind: str                          # wall | landmark
    normal: tuple | None
    body: int | None = None


@dataclass
class GlobalRoute:
    grid: object
    plan: object
    cfg: UncertaintyConfig
    position_bound_m: float
    yaw_bound_rad: float
    schedule: frozenset
    metrics: dict
    trace: list
    legs: list
    stations: list
    yaw0: float
    prior: PlannerBelief = field(repr=False, default=None)


PAIR_BASELINE_M = 0.30
"""Lateral distance between the two guard fists (`punch_actions.PARALLEL_GUARD`):
the baseline a pair of contacts on one face measures its orientation over."""


def minimum_clearance_target(xy, target, minimum_m):
    """Extend a short correction along its direction; _goto still gates it."""
    xy, target = np.asarray(xy, float), np.asarray(target, float)
    delta = target - xy
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        return None
    return xy + delta * max(1., float(minimum_m) / length)


def filter_consistent_config(loc_cfg, overrides=()) -> UncertaintyConfig:
    """The planner's growth AND contact model from the filter's numbers.

    The conceptual contact assumes range sigma ``sigma_contact_m`` and paired
    face-orientation sigma ``sqrt(2) * sigma_contact_m / baseline``. Runtime
    selects individual EIG touches and checks the resulting measured belief.
    """
    step = DEFAULT.planning_step_m
    sig_c = float(loc_cfg.sigma_contact_m)
    yaw_obs = math.sqrt(2.0) * sig_c / PAIR_BASELINE_M
    cfg = DEFAULT.merged(
        position_constraint="cross_track",
        grid_resolution_m=.10, max_move_m=.375, contact_standoff_m=.30,
        goal_sigma_multiplier=math.sqrt(-2 * math.log(.05)),
        position_growth_per_step_m=float(loc_cfg.sigma_along_per_m) * step,
        yaw_growth_per_step_rad=float(loc_cfg.sigma_yaw_per_m) * step,
        drift_yaw_per_rad=float(loc_cfg.sigma_yaw_per_rad),
        wall_position_obs_sigma_m=sig_c, landmark_position_obs_sigma_m=sig_c,
        wall_yaw_obs_sigma_rad=yaw_obs, landmark_yaw_obs_sigma_rad=yaw_obs,
        landmark_two_axes=False,
        max_relaxations=0,
        yaw_bin_rad=math.radians(5.0), sigma_bin_m=0.05, angle_bins=3,
        max_explored_states=300000)
    return parse_overrides(list(overrides), cfg) if overrides else cfg


def _cumulative(legs):
    """Route arclength at the start of each leg, continuous across turns
    (the same construction `StepEigPolicy` uses)."""
    L = [0.0]
    for k in range(len(legs) - 1):
        a, b = legs[k], legs[k + 1]
        end_pt = a.pt + a.s_end * a.along
        s_b, _ = b.coords(end_pt)
        L.append(L[k] + a.s_end - s_b)
    return L


def legs_from_path(grid, path, schedule, cfg: UncertaintyConfig, trace=None):
    """Group the cell path into legs of constant travel direction and
    commanded heading; place the scheduled contacts on them."""
    moves = []
    for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
        pa, pb = grid.cell_xy(a), grid.cell_xy(b)
        d = pb - pa
        L = float(np.linalg.norm(d))
        head = math.atan2(d[1], d[0])
        yaw = commanded_heading(grid, b, head, cfg)
        cs = grid.contacts.get(b, ())
        moves.append(dict(i=i, a=pa, b=pb, L=L, head=head, yaw=yaw, contacts=cs))

    legs, groups = [], []
    for m in moves:
        key = (round(m["head"], 3), round(m["yaw"], 3))
        if groups and groups[-1][0] == key:
            groups[-1][1].append(m)
        else:
            groups.append((key, [m]))
    for k, (key, ms) in enumerate(groups):
        along = np.array([math.cos(ms[0]["head"]), math.sin(ms[0]["head"])])
        touch_cs = [c for m in ms for c in m["contacts"]]
        if touch_cs:
            c = min(touch_cs, key=lambda c: c["distance"])
            n_in = np.asarray(c["normal"], float)
            normal = tuple(float(v) for v in n_in)
        else:
            n_in = np.array([-along[1], along[0]])
            normal = None
        legs.append(RouteLeg(pt=ms[0]["a"].copy(), along=along, n_in=n_in,
                             s_end=float(sum(m["L"] for m in ms)),
                             kind="stop" if k == len(groups) - 1 else "turn",
                             label=f"leg{k + 1}", yaw_cmd=float(ms[0]["yaw"]),
                             touch=bool(touch_cs), normal=normal))
    L = _cumulative(legs)
    move_leg = {m["i"]: k for k, (_key, ms) in enumerate(groups) for m in ms}
    stations = []
    for i, _step in sorted(schedule):
        m = moves[i]
        k = move_leg[i]
        s, _off = legs[k].coords(m["b"])
        cs = m["contacts"]
        if not cs:
            continue
        lm = [c for c in cs if c["kind"] == "landmark"]
        c = min(lm or cs, key=lambda c: c["distance"])
        predicted = trace[i] if trace is not None else {}
        normal = predicted.get("contact_normal", c["normal"])
        stations.append(PlanStation(move_index=i, leg=k, s=float(s), g=float(L[k] + s),
                                    xy=tuple(float(v) for v in m["b"]), kind=c["kind"],
                                    normal=tuple(float(v) for v in normal),
                                    body=predicted.get("contact_body", c.get("body"))))
    return legs, stations


def plan_global(scene, yaw0: float, cfg: UncertaintyConfig, position_bound_m: float,
                yaw_bound_rad: float, prior_sigma_xy_m: float | tuple[float, float] = 0.03,
                prior_sigma_theta_rad: float = 0.02, timing: Timing = Timing(),
                log=None, thin: str = "dense", thin_margin: float = 0.15,
                path=None, align_start: bool = False) -> GlobalRoute | None:
    """thin: 'balanced' keeps the required contacts spread so the belief
    stays thin_margin below the bound everywhere; 'time' is the greedy
    route-order deletion (fewest contacts, clustered where the bound binds);
    'dense' keeps every available contact.  path replays a given cell
    path instead of planning."""
    """Plan under the bounds, thin the contacts, and lay the route out for
    the local policy.  None when no route satisfies the bounds."""
    log = log or (lambda *a, **k: None)
    cfg = cfg.merged(position_threshold_m=position_bound_m, yaw_threshold_rad=yaw_bound_rad)
    grid = build_grid(scene, cfg)
    sigma_xy = np.broadcast_to(np.asarray(prior_sigma_xy_m, float), (2,))
    prior = PlannerBelief(cov=np.diag(sigma_xy ** 2),
                          yaw_sigma=prior_sigma_theta_rad, minor=0.0)
    if path is not None:
        from lane_graph.grid_constrained import GridPlan, free_travel_distance
        path = [tuple(c) for c in path]
        plan = GridPlan(path, grid.path_length(path), grid.path_length(path), True, 0,
                        position_bound_m, yaw_bound_rad, 0, reason="replayed",
                        strategy="replay",
                        free_distance_m=sum(free_travel_distance(grid, a, b)
                                            for a, b in zip(path, path[1:])))
    else:
        plan = plan_grid(grid, prior, cfg, yaw0)
    log("GLOBAL_PLAN", feasible=plan.feasible, distance_m=round(plan.distance_m, 2)
        if plan.feasible else None, explored=plan.explored_states, reason=plan.reason,
        requested_strategy=cfg.route_strategy, strategy=plan.strategy,
        strategy_stats=plan.strategy_stats)
    if not plan.feasible:
        return None
    if align_start and len(plan.path) > 1:
        # Choose the spawn yaw from the selected route, then replay its belief
        # from that initial heading. This changes setup, not runtime progress.
        delta = grid.cell_xy(plan.path[1]) - grid.cell_xy(plan.path[0])
        initial_yaw = commanded_heading(grid, plan.path[1], math.atan2(delta[1], delta[0]), cfg)
        log("START_HEADING", scene_yaw_deg=math.degrees(yaw0),
            spawn_yaw_deg=math.degrees(initial_yaw), mode="route")
        yaw0 = float(initial_yaw)
    backend = GridBackend(grid)
    cache = {}

    def evaluate(schedule):
        if schedule not in cache:
            cache[schedule] = evaluate_route(backend, plan.path, prior, cfg, yaw0, timing, schedule)
        return cache[schedule]

    dense_metrics, dense_trace = evaluate(None)
    y_deg = math.degrees(yaw_bound_rad)
    dense_sched = frozenset((r["edge_index"], r["step"]) for r in dense_trace if r["contact"] != "none")
    best = None
    if thin == "balanced":
        sched = prune_schedule_balanced(evaluate, dense_trace, position_bound_m, y_deg, thin_margin)
        if sched is not None:
            best = (sched, evaluate(sched)[0])
        else:
            log("GLOBAL_THIN", note=f"dense contact cannot hold {1 - thin_margin:.0%} of the bound; "
                                    "falling back to route-order thinning")
    if thin == "time" or (thin == "balanced" and best is None):
        for reverse in (False, True):
            sched = prune_schedule(evaluate, dense_trace, position_bound_m, y_deg, reverse)
            metrics, _ = evaluate(sched)
            if feasible(metrics, position_bound_m, y_deg) and (
                    best is None or metrics["estimated_time_s"] < best[1]["estimated_time_s"]):
                best = (sched, metrics)
    if best is None:                      # dense, or thinning found nothing better
        best = (dense_sched, dense_metrics)
    schedule, metrics = best
    _m, trace = evaluate(schedule)
    if schedule != dense_sched:
        last = trace[-1]
        margin = (np.linalg.norm(np.array([last["x"], last["y"]]) - grid.goal_xy)
                  + cfg.goal_sigma_multiplier * position_sigma_max(last["cov"]))
        if margin > grid.goal_radius + 1e-9:
            log("GLOBAL_THIN", note="thinned contacts fail goal confidence; retaining dense contacts")
            schedule, metrics, trace = dense_sched, dense_metrics, dense_trace
    legs, stations = legs_from_path(grid, plan.path, schedule, cfg, trace)
    if align_start and not feasible(metrics, position_bound_m, y_deg):
        log("GLOBAL_PLAN_REJECTED", reason="aligned initial heading fails belief bounds")
        return None
    log("GLOBAL_ROUTE", legs=len(legs), stations=len(stations),
        distance_m=round(metrics["distance_m"], 2), probes=metrics["probe_count"],
        est_time_s=round(metrics["estimated_time_s"], 1),
        peak_sigma_m=round(metrics["peak_position_sigma_m"], 3))
    return GlobalRoute(grid=grid, plan=plan, cfg=cfg, position_bound_m=position_bound_m,
                       yaw_bound_rad=yaw_bound_rad, schedule=schedule, metrics=metrics,
                       trace=trace, legs=legs, stations=stations, yaw0=yaw0, prior=prior)


def route_record(route: GlobalRoute) -> dict:
    """What the run record keeps of the plan (JSON-serialisable)."""
    return {
        "planner": "grid", "position_bound_m": route.position_bound_m,
        "initial_yaw_rad": route.yaw0,
        "position_constraint": route.cfg.position_constraint,
        "route_objective": route.cfg.route_objective,
        "requested_strategy": route.cfg.route_strategy,
        "strategy": route.plan.strategy,
        "strategy_stats": route.plan.strategy_stats,
        "free_distance_m": route.plan.free_distance_m,
        "open_boundaries": list(route.grid.spec.open_boundaries),
        "yaw_bound_deg": math.degrees(route.yaw_bound_rad),
        "grid": dict(resolution_m=route.grid.res, max_move_m=route.cfg.max_move_m,
                     contact_standoff_m=route.cfg.contact_standoff_m,
                     contact_band_m=route.cfg.contact_band_m,
                     free_cells=int(route.grid.free.sum()), touch_cells=len(route.grid.contacts)),
        "model": dict(position_growth_per_step_m=route.cfg.position_growth_per_step_m,
                      yaw_growth_per_step_deg=math.degrees(route.cfg.yaw_growth_per_step_rad),
                      drift_yaw_per_rad=route.cfg.drift_yaw_per_rad),
        "metrics": {k: v for k, v in route.metrics.items() if k != "profile"},
        "explored_states": route.plan.explored_states,
        "path_xy": [route.grid.cell_xy(c).tolist() for c in route.plan.path],
        "legs": [dict(label=l.label, kind=l.kind, pt=l.pt.tolist(), along=l.along.tolist(),
                      n_in=l.n_in.tolist(), s_end=l.s_end, yaw_cmd=l.yaw_cmd, touch=l.touch,
                      normal=l.normal) for l in route.legs],
        "stations": [dict(move_index=s.move_index, leg=s.leg, s=s.s, g=s.g, xy=list(s.xy),
                          kind=s.kind, normal=s.normal, body=s.body) for s in route.stations],
        "predicted": [dict(distance_m=r["distance_m"], x=r["x"], y=r["y"],
                           sigma_m=r["position_sigma_m"], yaw_sigma_deg=r["yaw_sigma_deg"],
                           motion_tangent=r["motion_tangent"], cov_xy=r["cov"],
                           source_sigma_m=r["source_position_sigma_m"],
                           pre_contact_sigma_m=r["pre_contact_position_sigma_m"],
                           max_axis_sigma_m=r["max_axis_sigma_m"],
                           contact=r["contact"]) for r in route.trace],
    }


def plot_route(route: GlobalRoute, scene, out_png) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    g = route.grid
    fig, ax = plt.subplots(figsize=(9, 7))
    for body in scene.plan.bodies:
        poly = np.asarray(body.verts)
        ax.fill(poly[:, 0], poly[:, 1], color="#d6d9de")
    regions = g.region_cells()
    for kind, color in (("wall", "#f3c9a6"), ("landmark", "#b9e2c4")):
        pts = np.asarray(regions[kind])
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], marker="s", s=6, c=color, lw=0, zorder=1)
    xy = np.asarray([g.cell_xy(c) for c in route.plan.path])
    ax.plot(xy[:, 0], xy[:, 1], color="#3167c6", lw=1.5, zorder=3)
    for i, r in enumerate(route.trace):
        if i % 4 == 0 or r["contact"] != "none":
            cov = np.asarray(r["cov"])
            w, v = np.linalg.eigh(0.5 * (cov + cov.T))
            w = np.sqrt(np.maximum(w, 0))
            ax.add_patch(Ellipse((r["x"], r["y"]), 4 * w[1], 4 * w[0],
                                 angle=np.degrees(np.arctan2(v[1, 1], v[0, 1])),
                                 facecolor="#3167c6", alpha=.18, edgecolor="#1d4a9a", lw=.5, zorder=2))
    for leg in route.legs:
        mid = leg.pt + 0.5 * leg.s_end * leg.along
        ax.annotate("", xy=mid + 0.35 * np.array([math.cos(leg.yaw), math.sin(leg.yaw)]),
                    xytext=mid, arrowprops=dict(arrowstyle="->", color="#555", lw=1))
    if route.stations:
        st = np.asarray([s.xy for s in route.stations])
        ax.scatter(st[:, 0], st[:, 1], c="#e16b2a", s=28, zorder=5,
                   label=f"{len(route.stations)} planned contacts")
    ax.scatter(*g.cell_xy(g.start), marker="s", color="green", s=40, zorder=6, label="start")
    ax.scatter(*g.goal_xy, marker="*", color="red", s=90, zorder=6, label="goal")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    m = route.metrics
    ax.set_title(f"{scene.name}: global route {m['distance_m']:.1f} m, {len(route.legs)} legs, "
                 f"{m['probe_count']} planned contacts, {route.cfg.position_constraint} bound {route.position_bound_m:g} m / "
                 f"{math.degrees(route.yaw_bound_rad):g} deg\n2-sigma predicted belief (blue); "
                 "arrows = commanded base heading; tan/green = wall/landmark touch regions", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


T_PUNCH_S = 2.3
"""One force-controlled punch, measured."""
MIN_TOUCH_IG_NATS = LAMBDA_TIME * T_PUNCH_S
"""A touch must be worth its own time in expected information."""


def _expected_ig(h, cov3=None, projection=None) -> float:
    """Expected information about pose, or a linear part of pose ``G x``.

    Each hit/miss branch already uses a rank-one Gaussian covariance
    update along H. Project that same update, including position/yaw
    correlations, rather than rewarding precision in unrelated coordinates.
    """
    if h.body_k == -1 or not np.isfinite(h.ig_hit) or not np.isfinite(h.ig_miss):
        return -math.inf
    hit, miss = h.ig_hit, h.ig_miss
    if projection is not None and h.H is not None:
        P, G = np.asarray(cov3, float), np.atleast_2d(projection)
        PH = P @ h.H
        var = float(h.H @ PH)
        if var <= 0.0:
            return 0.0
        cross = G @ PH
        fraction = float(np.clip(cross @ np.linalg.pinv(G @ P @ G.T) @ cross / var, 0., 1.))

        def projected(ig):
            reduction = fraction * -math.expm1(-2.0 * ig)
            return -0.5 * math.log(max(1.0 - reduction, 1e-15))

        hit, miss = projected(hit), projected(miss)
    return float(h.p_hit * hit + (1.0 - h.p_hit) * miss)


class SingleTouchSelector:
    """One touch at a time, chosen by expected information gain over BOTH
    hands and the whole angle band, against the local map.

    Each candidate is one hand pressing at one angle.  Its value is the
    expected information of that press: the map at the believed pose gives
    the margin to the surface along the press, the belief gives the
    probability of a hit, and both branches, contact and swept air, are
    priced.  Nothing here knows about pairs, wall roles or leg ends: a
    second fist on the same face wins when the face orientation is what the
    belief lacks, and a sideways reach wins near a wall end because the
    uncertain hit-or-miss there is what carries the information.
    An optional projection scores information about selected pose coordinates
    instead of the full pose; the observation model and action set are shared.
    """

    def __init__(self, plan, localizer, sigma_contact_m=0.018, n_grid=31,
                 log=None, on_eval=None, open_boundaries=(), reach_profile=None):
        self.eig = ClearedArmEIG(plan, sigma_contact_m, n_grid, reach_profile=reach_profile)
        self.loc = localizer
        self._log = log or (lambda *a, **k: None)
        self._on_eval = on_eval or (lambda ev, scores: None)
        self.last = None
        self.open_boundaries = tuple(open_boundaries)

    def choose_single(self, meta=None, projection=None):
        mu3, cov3 = self.loc.pose(), self.loc.cov3()
        full = (-40.0, 110.0)
        profile = self.eig.reach_profile
        bands = {h: full if profile is None else profile.band_deg(h) for h in ("left", "right")}
        ev = self.eig.evaluate(mu3, cov3, bands["left"], bands["right"], meta=dict(meta or {}))
        best = None
        for hand in ("left", "right"):
            for i, h in enumerate(ev.hands[hand]):
                if self.open_boundaries:
                    p0, direction, _ = arm_geometry(hand, h.alpha)
                    c, s = math.cos(mu3[2]), math.sin(mu3[2])
                    R = np.array([[c, -s], [s, c]])
                    ends = np.array([p0, p0 + TRAVEL_M * direction]) @ R.T + mu3[:2]
                    if profile is not None:
                        ends = np.asarray(self.eig.command_geometry(hand, h.alpha))[:, :2] @ R.T + mu3[:2]
                    if any(np.max(ends @ np.array([nx, ny])) > offset
                           for nx, ny, offset in self.open_boundaries):
                        continue
                v = _expected_ig(h, cov3, projection)
                if best is None or v > best[0]:
                    best = (v, hand, i, h)
        if best is None or not np.isfinite(best[0]):
            self._log("TOUCH", press=False, reason="no feasible touch")
            return None
        v, hand, i, h = best
        ev.best_ij = (i, ev.best_ij[1]) if hand == "left" else (ev.best_ij[0], i)
        margin = None if h.margin_m is None else float(h.margin_m)
        ev.meta.update({"single_hand": hand, "single_alpha_deg": round(math.degrees(h.alpha), 1),
                        "single_expected_ig": round(v, 4), "single_p_hit": round(float(h.p_hit), 3),
                        "single_pose_ig": round(_expected_ig(h), 4),
                        "information_projection": (None if projection is None
                                                   else np.asarray(projection).tolist()),
                        "single_margin_m": None if margin is None else round(margin, 3)})
        self.last = ev
        if profile is not None:
            ev.meta.update(profile.action_record(hand, h.alpha))
            ev.meta["candidate_probe_stroke_m"] = {
                side: [self.eig.reach_at(side, alpha) for alpha in ev.alphas[side]]
                for side in ("left", "right")}
        self._log("TOUCH", hand=hand, alpha_deg=round(math.degrees(h.alpha), 1),
                  p_hit=round(float(h.p_hit), 2), ig=round(v, 3),
                  margin_m=None if margin is None else round(margin, 2))
        self._on_eval(ev, [])
        spec = (spec_at(hand, h.alpha) if profile is None
                else spec_at(hand, h.alpha, reach_profile=profile))
        return dict(hand=hand, spec=spec, expected_ig=v, p_hit=float(h.p_hit),
                    alpha_deg=math.degrees(h.alpha), margin_m=margin)


class RouteFollowPolicy:
    """Motion from the plan and the estimate, touch from the local selector.

    At every station the next base target is the route point one step ahead
    of the BELIEVED arclength on the current leg (clamped to the leg end),
    reached as a body-frame displacement from the believed pose with the
    leg's commanded heading.  Then the station is touched one EIG-chosen
    action at a time until the belief is at the sigma the plan predicts for
    this arclength, or no touch is worth its time. At routine stations,
    follow-up touches earn information only about coordinates still above
    the planned targets.
    Before translation, predict
    joint position/yaw uncertainty across that move. In probe_then_continue
    mode, excessive position uncertainty triggers a bounded sensing batch,
    then one primitive may proceed with that exceedance recorded. Geometry
    and yaw checks remain mandatory; every subsequent primitive is checked anew.
    """

    MOVE_TIMEOUT_S = 15.0
    END_TOL_M = 0.10
    MAX_STATIONS = 400
    STEP_ALONG_M = 0.40
    MAX_LEG_M = 0.60
    WALL_CORRECTION_M = 0.10
    CLEARANCE_MIN_MOVE_M = 0.08
    WALL_CORRECTION_TOL_M = 0.05
    MAX_TOUCHES = 4
    """Touches per station before the policy moves on regardless."""
    TARGET_TOL = 1.10
    """The belief counts as at the planned sigma within this factor."""
    MIN_PROGRESS = 0.03
    """End the probing batch when neither position nor yaw sigma improves
    by this fraction. A miss must earn another probe just as a hit does."""

    def __init__(self, io: TactileIO, route: GlobalRoute, selector: SingleTouchSelector,
                 log=None, done_label: str = "route_complete", yaw0_map: float = 0.0,
                 station_tol_m: float = 0.25, step_m: float | None = None,
                 max_touches: int | None = None, target_sigma_m: float | None = None,
                 target_yaw_sigma_rad: float | None = None,
                 guard_mode: str = "belief", arrival_confidence: float = .95,
                 recovery_enabled: bool = True,
                 execution_pos_trigger_m: float | None = None,
                 execution_pos_policy: str = "probe_then_continue"):
        self.io = io
        self.sel = selector
        self.route = route
        self.travel = "route"
        self._log = log or (lambda *a, **k: None)
        self.done_label = done_label
        self._yaw_cmd = float(yaw0_map)
        self.segments = list(route.legs)
        self.L = _cumulative(self.segments)
        self.k = 0
        self.plan_stations = sorted(route.stations, key=lambda s: s.g)
        self.next_station = 0
        self.station_tol_m = float(station_tol_m)
        io.expect_travel = self._expect_travel
        io.creep_to_full = True
        self.stations: list[dict] = []
        if step_m is not None:
            self.STEP_ALONG_M = float(step_m)
        if max_touches is not None:
            self.MAX_TOUCHES = int(max_touches)
        self.target_sigma_m = target_sigma_m
        self.target_yaw_sigma_rad = target_yaw_sigma_rad
        """Fixed touch targets.  None: the target at a station is the sigma the
        plan predicts there (touch only where the plan counted on it).  A
        number: touch at every station until the belief is under it."""
        self.touches: list[dict] = []
        self.motion_checks: list[dict] = []
        if guard_mode not in ("distance", "contact", "belief"):
            raise ValueError(f"unknown route guard: {guard_mode}")
        self.guard_mode = guard_mode
        if execution_pos_trigger_m is None:
            execution_pos_trigger_m = route.position_bound_m
        if not math.isfinite(execution_pos_trigger_m) or execution_pos_trigger_m <= 0:
            raise ValueError("execution position trigger must be finite and positive")
        if execution_pos_policy not in ("strict", "probe_then_continue"):
            raise ValueError(f"unknown execution position policy: {execution_pos_policy}")
        self.execution_pos_trigger_m = float(execution_pos_trigger_m)
        self.execution_pos_policy = execution_pos_policy
        self.position_recoveries: list[dict] = []
        self.guard = BeliefRouteGuard(route, arrival_confidence)
        self.transitions: list[dict] = []
        self._endpoint_attempted = False
        self._arrival_retries = 0
        self._wall_recovering = False
        self.recovery_enabled = bool(recovery_enabled)

    def _station_due(self, g: float):
        """The first unserved station whose arclength the belief has reached."""
        while (self.next_station < len(self.plan_stations)
               and self.plan_stations[self.next_station].g < g - self.station_tol_m):
            st = self.plan_stations[self.next_station]
            self._log("STATION_PASSED", g=round(st.g, 2), kind=st.kind)
            self.next_station += 1
        if self.next_station < len(self.plan_stations):
            st = self.plan_stations[self.next_station]
            if st.g <= g + self.station_tol_m:
                return st
        return None

    def progress(self, xy) -> float:
        s, _ = self.segments[self.k].coords(xy)
        return self.L[self.k] + s

    def _expect_travel(self, spec):
        ev = self.sel.last
        if ev is None:
            return None
        from .punch_eig import DOME_R_M
        i = ev.best_ij[0] if spec.hand == "left" else ev.best_ij[1]
        hs = ev.hands[spec.hand][i]
        if hs.d_surface_m is None:
            return None
        profile = getattr(getattr(self.sel, "eig", None), "reach_profile", None)
        d = float(hs.d_surface_m) - (DOME_R_M if profile is None else 0.)
        return float(np.clip(d, 0.0, spec.travel_m)), float(hs.sigma_m)

    # ── what the plan expects the belief to be here ────────────────────────

    def _planned_sigma(self, g: float) -> tuple[float, float]:
        """(position sigma, yaw sigma) the global plan predicts at route
        arclength ``g`` after any contact it scheduled there, capped by the
        bounds.  This is the threshold the station touches toward."""
        rows = self.route.trace
        row = next((r for r in rows if r["distance_m"] >= g - 1e-6), rows[-1])
        pos = min(float(row["position_sigma_m"]), self.route.position_bound_m)
        yaw = min(float(row["yaw_sigma_rad"]), self.route.yaw_bound_rad)
        if self.target_sigma_m is not None:
            pos = min(pos, float(self.target_sigma_m))
        if self.target_yaw_sigma_rad is not None:
            yaw = min(yaw, float(self.target_yaw_sigma_rad))
        return pos, yaw

    def _belief_sigma(self, tangent) -> tuple[float, float]:
        loc = self.io.localizer
        return (motion_position_sigma(loc.cov_xy(), tangent, self.route.cfg.position_constraint),
                float(loc.sigma_yaw()))

    def _touch_until(self, seg, g: float, required: bool, guards: dict,
                     pending_target=None, accept=None, recovery=False) -> dict:
        """Touch, one EIG-chosen action at a time, until the belief is at the
        planned sigma for this arclength or no touch is worth its time."""
        target_pos, target_yaw = self._planned_sigma(g)
        if pending_target is not None:
            target_pos, target_yaw = self.execution_pos_trigger_m, self.route.yaw_bound_rad

        def sigma():
            if pending_target is None:
                return self._belief_sigma(seg.along)
            prediction = self._motion_prediction(pending_target, seg.yaw, recovery)
            return prediction["sigma_m"], prediction["yaw_sigma_rad"]

        uses_eig = getattr(self.sel, "uses_eig", True)
        if not uses_eig:
            self.sel.begin_batch()
        done, reason = [], "no touch worth its time"
        for n in range(self.MAX_TOUCHES):
            sig_pos, sig_yaw = sigma()
            tol = self.TARGET_TOL if pending_target is None else 1.0
            reached = (sig_pos <= target_pos * tol and sig_yaw <= target_yaw * tol)
            if accept is not None:
                reached = bool(accept())
            if reached and (n > 0 or not uses_eig):
                reason = "at target sigma"
                break
            # Ask the selector FIRST: a touch that pays for its time is thrown
            # even when the belief is already at the target, because a tight
            # belief is exactly what an unnoticed error hides behind (the
            # start pose most of all).  The target only decides when to stop
            # touching, never whether to start. Follow-ups must inform an
            # unmet target: sharpening an already sufficient wall distance
            # cannot justify more punches when only yaw is deficient.
            projection = None
            if uses_eig and n > 0 and pending_target is None and accept is None:
                rows = []
                if sig_pos > target_pos * tol:
                    if self.route.cfg.position_constraint == "cross_track":
                        rows.append([-seg.along[1], seg.along[0], 0.])
                    else:
                        values, vectors = np.linalg.eigh(self.io.localizer.cov_xy())
                        rows.extend([*v, 0.] for value, v in zip(values, vectors.T)
                                    if value > (target_pos * tol) ** 2)
                if sig_yaw > target_yaw * tol:
                    rows.append([0., 0., 1.])
                projection = np.asarray(rows).reshape(-1, 3)
            pick = self.sel.choose_single(meta={"state": f"touch:{seg.label}", "g": round(g, 3),
                                                "n": n, "required": required,
                                                "target_sigma_m": round(target_pos, 3),
                                                "target_yaw_sigma_deg": math.degrees(target_yaw)},
                                          projection=projection)
            if pick is None:
                reason = "no feasible touch"
                break
            if uses_eig and pick["expected_ig"] < MIN_TOUCH_IG_NATS:
                reason = ("no touch worth its time" if n == 0 else "remaining touches not worth their time")
                break
            hand, spec = pick["hand"], pick["spec"]
            guards[hand] = spec.guard
            res = self.io.punch(spec, guards, f"touch:{seg.label}")
            if getattr(res, "status", "complete") == "guard":
                # the force guard stopped the punch: not a contact, not a miss
                reason = "punch interrupted by guard"
                guards.update(self.io.parked_carry(guards) if hasattr(self.io, "parked_carry") else {})
                break
            sig_pos2, sig_yaw2 = sigma()
            rec = {"seg": seg.label, "g": g, "n": n, "hand": hand, "alpha_deg": pick["alpha_deg"],
                   "p_hit": pick["p_hit"], "expected_ig": pick["expected_ig"], "margin_m": pick["margin_m"],
                   "hit": bool(res.contact), "sigma_before": sig_pos, "sigma_after": sig_pos2,
                   "yaw_sigma_before_deg": math.degrees(sig_yaw), "yaw_sigma_after_deg": math.degrees(sig_yaw2),
                   "target_sigma_m": target_pos, "target_yaw_sigma_deg": math.degrees(target_yaw),
                   "information_projection": None if projection is None else projection.tolist(),
                   "required": required,
                   "before_translation": pending_target is not None}
            if res.contact:
                # Infer the observed surface from the posterior and measured
                # fingertip. Simulator body IDs must not drive the executor.
                p_body = self.io.punches[-1].get("p_body")
                if p_body is not None:
                    pose = self.io.localizer.pose()
                    c, s = math.cos(pose[2]), math.sin(pose[2])
                    point = pose[:2] + np.array([[c, -s], [s, c]]) @ np.asarray(p_body)[:2]
                    ds_ns = [b.signed_distance(point[None, :]) for b in self.route.grid.bodies]
                    body = int(np.argmin([abs(v[0][0]) for v in ds_ns]))
                    rec.update(surface_body=body, surface_normal=np.asarray(ds_ns[body][1][0]).tolist())
            done.append(rec)
            self.touches.append(rec)
            self._log("TOUCH_RESULT", kind=seg.label, n=n, hand=hand, hit=bool(res.contact),
                      sigma=f"{sig_pos:.3f}->{sig_pos2:.3f}", target=round(target_pos, 3),
                      yaw_sigma_deg=f"{math.degrees(sig_yaw):.1f}->{math.degrees(sig_yaw2):.1f}")
            pos_progress = sig_pos2 <= sig_pos * (1.0 - self.MIN_PROGRESS)
            yaw_progress = sig_yaw2 <= sig_yaw * (1.0 - self.MIN_PROGRESS)
            if projection is not None:
                pos_progress &= sig_pos > target_pos * tol
                yaw_progress &= sig_yaw > target_yaw * tol
            if uses_eig and not (pos_progress or yaw_progress):
                reason = "no progress: probe did not reduce uncertainty"
                break
        else:
            reason = "touch limit"
        return {"touches": done, "reason": reason, "target_sigma_m": target_pos,
                "target_yaw_sigma_deg": math.degrees(target_yaw)}

    # ── motion from the plan and the estimate ──────────────────────────────

    def _wall_tracking_tangent(self, xy, target):
        """Keep bounded wall-distance feedback in the active route frame.

        This includes the small finishing move to a leg endpoint. Arrival
        still has its PF probability guard; entering the next leg changes
        the reference normal before any movement on that leg is admitted.
        """
        seg = self.segments[self.k]
        if not seg.touch or seg.normal is None or abs(np.dot(seg.along, seg.normal)) > .05:
            return None
        d = np.asarray(target) - xy
        n = np.array([-seg.along[1], seg.along[0]])
        if abs(d @ n) > self.WALL_CORRECTION_M + 1e-9:
            return None
        before, after = (xy - seg.pt) @ n, (np.asarray(target) - seg.pt) @ n
        if abs(after) > abs(before) + 1e-9:
            return None
        s = float((np.asarray(target) - seg.pt) @ seg.along)
        return seg.along if -1e-9 <= s <= seg.s_end + 1e-9 else None

    def _motion_prediction(self, target_xy, yaw_map, recovery=False):
        loc = self.io.localizer
        xy = np.asarray(loc.pose()[:2], float)
        d = np.asarray(target_xy, float) - xy
        ds = float(np.linalg.norm(d))
        S = loc.cov3()
        rotating = ds < 1e-9
        pos_bound = getattr(self, "execution_pos_trigger_m", self.route.position_bound_m)
        reference = (self._wall_tracking_tangent(xy, target_xy)
                     if not rotating and hasattr(self, "segments") else None)
        *_, info = belief_step(S, float(S[2, 2]), self._yaw_cmd, True, True,
                              ds, np.array([1., 0.]) if rotating else d / ds,
                              yaw_map, "none", None, self.route.cfg,
                              math.inf if rotating else pos_bound, self.route.yaw_bound_rad,
                              constraint_tangent=reference)
        rec = dict(t=float(self.io.time_s), xy=xy.tolist(), target_xy=np.asarray(target_xy).tolist(),
                    distance_m=ds, tangent=None if rotating else (d / ds).tolist(), cov3=np.asarray(S).tolist(),
                    constraint_tangent=info["constraint_tangent"],
                    wall_correction=reference is not None and recovery,
                    constraint_frame="active_route" if reference is not None else "motion",
                    source_sigma_m=info["source_pos_sigma"], endpoint_sigma_m=info["pre_pos_sigma"],
                    sigma_m=info["max_pos_sigma"], yaw_sigma_rad=info["max_yaw_sigma"],
                    bound_m=pos_bound, planner_bound_m=self.route.position_bound_m,
                    position_allowed=bool(rotating or info["max_pos_sigma"] <= pos_bound + 1e-12),
                    yaw_allowed=bool(info["max_yaw_sigma"] <= self.route.yaw_bound_rad + 1e-12),
                    geometry_allowed=True, allowed=bool(info["ok"]))
        if not rotating and hasattr(self, "guard"):
            seg = self.segments[self.k]
            s, _ = seg.coords(target_xy)
            gap = self.guard.departure(xy, S, self._yaw_cmd, target_xy, self.L[self.k] + s, yaw_map,
                                       recovery, correction_tangent=reference,
                                       position_bound_m=pos_bound)
            rec["gap"] = gap
            if self.guard_mode == "belief":
                for key in ("position_allowed", "yaw_allowed", "geometry_allowed"):
                    rec[key] &= gap[key]
                rec["sigma_m"] = max(rec["sigma_m"], gap["sigma_m"])
                rec["yaw_sigma_rad"] = max(rec["yaw_sigma_rad"], gap["yaw_sigma_rad"])
        rec["allowed"] = all(rec[key] for key in ("position_allowed", "yaw_allowed", "geometry_allowed"))
        return rec

    def _to_body(self, target_xy):
        p = self.io.localizer.pose()
        d = np.asarray(target_xy, float) - p[:2]
        c, s = math.cos(p[2]), math.sin(p[2])
        return float(d @ np.array([c, s])), float(d @ np.array([-s, c]))

    def _goto(self, target_xy, yaw_map: float, guards: dict, kind: str) -> str:
        target_xy = np.asarray(target_xy, float)
        st = "done"
        recovery = kind.startswith("reconnect:")
        soft_position = self.guard_mode == "belief" and self.execution_pos_policy == "probe_then_continue"
        sensed = None
        while True:
            xy = np.asarray(self.io.localizer.pose()[:2])
            # Recompute both the body displacement and its subdivision after
            # observations: the PF can change position and yaw during probing.
            n = max(1, int(math.ceil(float(np.linalg.norm(target_xy - xy)) / self.MAX_LEG_M)))
            subtarget = xy + (target_xy - xy) / n
            turn = _wrap(yaw_map - self._yaw_cmd)
            if abs(turn) < 1e-3:
                turn = 0.0
            clear = (self.route.grid.segment_clear(xy, subtarget, allow_escape=True) if recovery
                     else self.route.grid.segment_clear(xy, subtarget))
            if not clear:
                return "route_obstructed"
            prediction = self._motion_prediction(subtarget, yaw_map, recovery)
            prediction["kind"] = kind
            prediction["issued"] = False
            prediction["position_override"] = False
            self.motion_checks.append(prediction)
            if sensed is not None:
                sensed.update(after_sigma_m=prediction["sigma_m"],
                              after_yaw_sigma_rad=prediction["yaw_sigma_rad"],
                              after_xy=xy.tolist(), sensing_end_t=float(self.io.time_s))
            if not prediction["geometry_allowed"]:
                return "route_obstructed"
            if not prediction["allowed"]:
                self._log("MOTION_UNCERTAINTY", kind=kind,
                          sigma_m=round(prediction["sigma_m"], 4),
                          bound_m=self.execution_pos_trigger_m)
                if soft_position and sensed is None and prediction["distance_m"] > 1e-9:
                    # Orient for sensing without translating, as in the
                    # original motion-recovery branch. A guard/timeout here
                    # is not permission to issue the pending translation.
                    if turn:
                        turn_status = self._goto(xy, yaw_map, guards, "turn:localize")
                        if turn_status != "done":
                            return turn_status
                    seg = self.segments[self.k]
                    sensed = dict(t=float(self.io.time_s), kind=kind,
                                  target_xy=target_xy.tolist(), before_xy=xy.tolist(),
                                  before_sigma_m=prediction["sigma_m"],
                                  before_yaw_sigma_rad=prediction["yaw_sigma_rad"],
                                  trigger_m=self.execution_pos_trigger_m, continued=False)
                    self.position_recoveries.append(sensed)
                    out = self._touch_until(seg, self.progress(self.io.localizer.pose()[:2]), True, guards,
                                            pending_target=subtarget, recovery=recovery,
                                            accept=lambda: self._motion_prediction(subtarget, yaw_map, recovery)["allowed"])
                    sensed.update(n_touches=len(out["touches"]), reason=out["reason"])
                    self._log("MOTION_SENSING", **{key: value for key, value in sensed.items() if key != "t"})
                    if out["reason"] == "punch interrupted by guard":
                        return "guard"
                    continue
                if not (soft_position and sensed is not None and prediction["yaw_allowed"]):
                    return "motion_uncertainty"
                prediction["position_override"] = True
                self._log("POSITION_BOUND_CONTINUE", kind=kind, sigma_m=prediction["sigma_m"],
                          trigger_m=self.execution_pos_trigger_m,
                          excess_m=max(0., prediction["sigma_m"] - self.execution_pos_trigger_m))
            fwd, left = self._to_body(subtarget)
            prediction["issued"] = True
            prediction["primitive_start_t"] = float(self.io.time_s)
            if sensed is not None:
                sensed.update(continued=True, position_override=prediction["position_override"])
            try:
                st = self.io.move(fwd, left, guards, kind, turn_rad=turn,
                                  timeout_s=20.0 if turn else self.MOVE_TIMEOUT_S)
                prediction["primitive_status"] = st
            finally:
                prediction.setdefault("primitive_status", "aborted")
                prediction.update(primitive_end_t=float(self.io.time_s),
                                  primitive_duration_s=float(self.io.time_s)-prediction["primitive_start_t"])
            if turn:
                self._yaw_cmd = _wrap(self._yaw_cmd + turn)
            if hasattr(self.io, "heading_command"):
                self._yaw_cmd = float(self.io.heading_command)
            sensed = None   # permission expires after exactly one primitive
            if st != "done" or n == 1:
                break
        if hasattr(self.io, "heading_command"):
            self._yaw_cmd = float(self.io.heading_command)   # the io's own heading after a cancel or a guard
        return st

    # ── stepping interface (a notebook can drive one station at a time) ────

    def start(self) -> None:
        """Settle, then face the first leg.  Call once before `step_station`."""
        from .punch_actions import PARALLEL_GUARD
        self._guards = getattr(self.io, "default_guards", lambda: dict(PARALLEL_GUARD))()
        self.io.settle(1.5, self._guards)
        turn0 = _wrap(self.segments[0].yaw - self._yaw_cmd)
        if abs(turn0) > 1e-3:
            self._log("TURN_TO_ROUTE", turn_deg=round(math.degrees(turn0), 1))
            status = self._goto(self.io.localizer.pose()[:2], self.segments[0].yaw, self._guards, "turn:start")
            if status != "done":
                self._start_failure = status
        self._n_stations = 0

    def _contact_satisfied(self):
        expected = [s for s in self.plan_stations if s.leg == self.k]
        if not expected:
            return True
        st = expected[-1]
        return any(t.get("surface_body") == st.body and
                   abs(t["g"] - st.g) <= self.route.grid.cfg.max_move_m and
                   np.dot(t.get("surface_normal", [0., 0.]), st.normal) >= math.cos(math.radians(20))
                   for t in self.touches)

    def step_station(self) -> str | None:
        """One station: advance one step along the route from the believed
        pose, then touch until the target.  Returns None to continue, or the
        outcome label when the route is complete or the station limit hit."""
        io = self.io
        guards = self._guards
        if getattr(self, "_start_failure", None):
            return self._start_failure
        if self._n_stations >= self.MAX_STATIONS:
            self._log("STATION_LIMIT", kind="route")
            return "station_limit"
        self._n_stations += 1
        seg = self.segments[self.k]
        s, off = seg.coords(io.localizer.pose()[:2])
        end_xy = seg.pt + seg.s_end * seg.along
        near_end = np.linalg.norm(io.localizer.pose()[:2] - end_xy) <= self.END_TOL_M
        # Intermediate legs describe a continuous route. Their endpoints are
        # tracking targets, not separate localization goals. The next _goto
        # still checks geometry and belief propagation to the next contact.
        check_arrival = self.guard_mode == "belief" and seg.kind == "stop"
        candidate = self._endpoint_attempted if check_arrival else near_end
        if candidate:
            arrival = self.guard.arrival(io.localizer, self.k) if check_arrival else None
            def satisfied():
                return (self.guard.arrival(io.localizer, self.k)["allowed"] if check_arrival
                        else self._contact_satisfied() if self.guard_mode == "contact" else True)
            if not satisfied():
                out = self._touch_until(seg, self.progress(io.localizer.pose()[:2]), True, guards, accept=satisfied)
                arrival = self.guard.arrival(io.localizer, self.k) if check_arrival else None
                self.transitions.append(dict(t=io.time_s, leg=self.k, mode=self.guard_mode,
                                             arrival=arrival, allowed=satisfied(),
                                             n_touches=len(out["touches"])))
                if not satisfied():
                    # A touch/miss may shift the posterior mean off the
                    # planned stance. Recenter locally under the same motion
                    # guard before concluding that arrival is impossible.
                    if (self.recovery_enabled and check_arrival and self._arrival_retries < 3 and
                            np.linalg.norm(io.localizer.pose()[:2] - end_xy) > .02):
                        self._arrival_retries += 1
                        status = self._goto(end_xy, seg.yaw, guards, "reconnect:arrival")
                        if status == "done":
                            return None
                    return "arrival_uncertain" if self.guard_mode == "belief" else "contact_unconfirmed"
            else:
                self.transitions.append(dict(t=io.time_s, leg=self.k, mode=self.guard_mode,
                                             arrival=arrival, allowed=True, n_touches=0))
            if seg.kind == "stop":
                self._log("ROUTE_END", kind=seg.label, s=round(s, 3))
                return self.done_label
            self.k += 1
            self._endpoint_attempted = False
            self._arrival_retries = 0
            self._wall_recovering = False
            seg = self.segments[self.k]
            self._log("LEG", kind=seg.label, touch=seg.touch, s_end=round(seg.s_end, 2),
                      yaw_deg=round(math.degrees(seg.yaw), 1))
            s, off = seg.coords(io.localizer.pose()[:2])
        # the next station: one step ahead along the leg, back on its line
        s_next = min(max(s, 0.0) + self.STEP_ALONG_M, seg.s_end)
        target = seg.pt + s_next * seg.along
        # Correct a displacement toward the nearest point on this fixed leg
        # before advancing along it. A local connector may go around obstacles.
        cross_off = float((io.localizer.pose()[:2] - seg.pt) @ np.array([-seg.along[1], seg.along[0]]))
        wall_parallel = seg.touch and seg.normal is not None and abs(np.dot(seg.along, seg.normal)) <= .05
        self._wall_recovering = bool(wall_parallel and
            (abs(cross_off) > self.route.grid.res / 2 or
             (self._wall_recovering and abs(cross_off) > self.WALL_CORRECTION_TOL_M)))
        kind = f"route:{seg.label}"
        recovery = self.recovery_enabled and (self._wall_recovering or abs(cross_off) > self.route.grid.res / 2)
        if recovery:
            reconnect = seg.pt + np.clip(s, 0., seg.s_end) * seg.along
            connector = local_connector(self.route.grid, io.localizer.pose()[:2], reconnect)
            if connector is None:
                return "reconnect_infeasible"
            target = next((p for p in connector if np.linalg.norm(p - io.localizer.pose()[:2]) > .02), reconnect)
            if self._wall_recovering:
                xy = np.asarray(io.localizer.pose()[:2])
                d = target - xy
                target = xy + d * min(1., self.WALL_CORRECTION_M / max(float(np.linalg.norm(d)), 1e-9))
            kind = "reconnect:wall" if self._wall_recovering else "reconnect:route"
            s_next = float(seg.coords(target)[0])
        st = self._goto(target, seg.yaw, guards, kind)
        if st == "motion_uncertainty":
            if self.guard_mode == "belief" and self.execution_pos_policy == "probe_then_continue":
                # _goto already attempted the bounded sensing batch. Remaining
                # yaw rejection is hard; positional exceedance alone continued.
                return st
            # Facing the planned surface is allowed without translating.
            turn = _wrap(seg.yaw - self._yaw_cmd)
            if abs(turn) > 1e-3:
                turn_status = self._goto(io.localizer.pose()[:2], seg.yaw, guards, "turn:localize")
                if turn_status != "done":
                    return turn_status
            out = self._touch_until(seg, self.progress(io.localizer.pose()[:2]), True, guards,
                                    pending_target=target, recovery=recovery,
                                    accept=lambda: self._motion_prediction(target, seg.yaw, recovery)["allowed"])
            self.stations.append(dict(seg=seg.label, target=target.tolist(), move=st,
                                      g=self.progress(io.localizer.pose()[:2]),
                                      n_touches=len(out["touches"]), stop_reason=out["reason"]))
            prediction = self._motion_prediction(target, seg.yaw, recovery)
            self.motion_checks.append(prediction)
            return None if prediction["allowed"] else "motion_uncertainty"
        if st == "route_obstructed":
            if not self.recovery_enabled:
                return st
            projection = seg.pt + np.clip(seg.coords(io.localizer.pose()[:2])[0], 0., seg.s_end) * seg.along
            connector = local_connector(self.route.grid, io.localizer.pose()[:2], projection)
            if connector:
                first = next((p for p in connector if np.linalg.norm(p - io.localizer.pose()[:2]) > .02), projection)
                first = minimum_clearance_target(io.localizer.pose()[:2], first,
                                                 self.CLEARANCE_MIN_MOVE_M)
                if first is None:
                    return "reconnect_infeasible"
                self._log("RECONNECT", target=first.tolist(), reason="restore route clearance",
                          minimum_move_m=self.CLEARANCE_MIN_MOVE_M)
                status = self._goto(first, seg.yaw, guards, "reconnect:clearance")
                if status == "done":
                    return None
                if status == "timeout":
                    # TactileIO has cancelled the primitive and predicted the
                    # PF from odometry. Choose again from that pose; the next
                    # _goto must pass geometry and belief admission afresh.
                    self._log("REPLAN", kind="reconnect:clearance", status=status)
                    return None
                self._log("RECONNECT_FAILED", reason="clearance move incomplete", status=status)
            return "reconnect_infeasible"
        if st != "done":
            if not self.recovery_enabled:
                return st
            # A passive contact interrupted the move (punching branch f1f5275):
            # the contact is already in the filter, the controller holds its
            # safety stop.  Walk with the parked carry and let the next station
            # be chosen from the corrected belief instead of touching here.
            self._guards = io.parked_carry(guards) if st == "guard" and hasattr(io, "parked_carry") else guards
            self._log("REPLAN", kind=f"route:{seg.label}", status=st)
            self.stations.append({"seg": seg.label, "target": [float(v) for v in target], "move": st,
                                  "g": self.progress(io.localizer.pose()[:2]), "required": False,
                                  "n_touches": 0, "stop_reason": f"move incomplete: {st}", "pressed": False,
                                  "landed": False, "sigma_m": io.localizer.sigma_m(),
                                  "sigma_yaw_deg": math.degrees(io.localizer.sigma_yaw())})
            return None
        s2, off2 = seg.coords(io.localizer.pose()[:2])
        g2 = self.progress(io.localizer.pose()[:2])
        due = self._station_due(g2)
        self._log("STATION", kind=seg.label, s=round(s2, 3), off=round(off2, 3), g=round(g2, 3),
                  s_end=round(seg.s_end, 3), target=[round(float(v), 3) for v in target],
                  move=st, required=due is not None)
        out = self._touch_until(seg, g2, due is not None, guards)
        landed = any(t["hit"] for t in out["touches"])
        rec = {"seg": seg.label, "target": [float(v) for v in target], "move": st, "g": g2,
               "off": off2, "required": due is not None,
               "station_g": None if due is None else due.g,
               "n_touches": len(out["touches"]), "stop_reason": out["reason"],
               "target_sigma_m": out["target_sigma_m"], "pressed": bool(out["touches"]),
               "landed": landed,
               "sigma_m": io.localizer.sigma_m(), "sigma_yaw_deg": math.degrees(io.localizer.sigma_yaw())}
        if due is not None:
            if landed:
                self.next_station += 1        # served: a contact was actually made
            else:
                # Not served.  The misses have already moved the belief (the
                # surface is farther than it thought), so the next route step,
                # taken from the corrected estimate, carries the base toward
                # it; the station stays due until a contact lands or the
                # believed arclength has passed it by the tolerance.
                self._log("STATION_UNSERVED", kind=seg.label, g=round(g2, 2), station_g=round(due.g, 2),
                          misses=len(out["touches"]))
        self.stations.append(rec)
        self._endpoint_attempted = bool(s_next >= seg.s_end - 1e-6)
        return None

    def run(self) -> str:
        self.start()
        while True:
            outcome = self.step_station()
            if outcome is not None:
                return outcome
