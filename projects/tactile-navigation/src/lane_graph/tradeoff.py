"""Offline selection among sampled routes and greedily pruned probe schedules.

Uses the existing conceptual-contact uncertainty model. Times are estimates,
all planned contacts are assumed successful, and limits are never relaxed.
This is candidate comparison, not joint route/hand-motion optimization.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import math
import time

from .constrained import plan_constrained, position_sigma_max, replay_path


class GraphBackend:
    """The roadmap planner behind the interface `run_sweep` drives."""
    name = "roadmap"

    def __init__(self, graph, src, dst):
        self.graph, self.src, self.dst = graph, src, dst

    def plan(self, belief, cfg, yaw0):
        return plan_constrained(self.graph, self.src, self.dst, belief, cfg, yaw0)

    def replay(self, path, belief, cfg, yaw0, schedule):
        return replay_path(self.graph, path, belief, cfg, self.src, yaw0, probe_schedule=schedule)

    def length(self, path):
        return float(sum(self.graph.edges[e].length for e in path))

    def describe(self):
        g = self.graph
        return dict(planner="roadmap", src=self.src, dst=self.dst,
                    start_xy=list(g.nodes[self.src].xy), goal_xy=list(g.nodes[self.dst].xy))


@dataclass(frozen=True)
class Timing:
    walk_mps: float = 0.45
    turn_radps: float = 0.9
    probe_seconds: float = 5.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0 or (name != "probe_seconds" and value == 0):
                raise ValueError(f"invalid timing parameter {name}: {value}")

    def components(self, distance, turn, probes):
        return dict(walk_time_s=distance / self.walk_mps,
                    turn_time_s=turn / self.turn_radps,
                    probe_time_s=probes * self.probe_seconds)


def localization(metrics, position_limit, yaw_limit_deg):
    """Localize-first feasibility of one evaluated route under a bound pair.

    Returns ``(feasible, localized_at_distance_m)``. The belief is localized
    at the first step whose post-observation sigmas satisfy both bounds; up to
    that step it may exceed neither ``max(bound, initial sigma + grace)``, and
    after it every pre- and post-observation sigma must stay inside the
    bounds. A start inside both bounds reports a localization distance of 0.
    """
    eps = 1e-10
    prof = metrics["profile"]
    pre_pos, post_pos = prof["pre_pos"], prof["post_pos"]
    pre_yaw, post_yaw = prof["pre_yaw_deg"], prof["post_yaw_deg"]
    s0, y0 = metrics["initial_position_sigma_m"], metrics["initial_yaw_sigma_deg"]
    first = next((i for i in range(len(post_pos))
                  if post_pos[i] <= position_limit + eps and post_yaw[i] <= yaw_limit_deg + eps), None)
    if first is None:
        return False, None
    grace_pos = max(position_limit, s0 + metrics["grace_position_m"]) + eps
    if metrics.get("position_constraint") == "cross_track":
        grace_pos = position_limit + eps
    grace_yaw = max(yaw_limit_deg, y0 + metrics["grace_yaw_deg"]) + eps
    before = list(zip(pre_pos[:first + 1], pre_yaw[:first + 1])) + list(zip(post_pos[:first], post_yaw[:first]))
    if any(p > grace_pos or y > grace_yaw for p, y in before):
        return False, None
    at = 0.0 if (first == 0 and s0 <= position_limit + eps and y0 <= yaw_limit_deg + eps) else prof["distance"][first]
    after = zip(pre_pos[first + 1:] + post_pos[first + 1:], pre_yaw[first + 1:] + post_yaw[first + 1:])
    if any(p > position_limit + eps or y > yaw_limit_deg + eps for p, y in after):
        return False, None
    return True, float(at)


def feasible(metrics, position_limit, yaw_limit_deg):
    return metrics["association_resolved"] and localization(metrics, position_limit, yaw_limit_deg)[0]


def pareto_indices(rows, keys):
    """Minimization; equal points survive, infeasible rows are caller-filtered."""
    return [i for i, a in enumerate(rows) if not any(
        all(b[k] <= a[k] + 1e-10 for k in keys)
        and any(b[k] < a[k] - 1e-10 for k in keys)
        for j, b in enumerate(rows) if i != j)]


def select_candidate(rows, position_limit, yaw_limit_deg, objective="time"):
    keys = {"time": ("estimated_time_s", "distance_m", "probe_count"),
            "distance": ("distance_m", "estimated_time_s", "probe_count"),
            "probes": ("probe_count", "estimated_time_s", "distance_m")}
    eligible = [r for r in rows if feasible(r, position_limit, yaw_limit_deg)]
    if objective not in keys:
        raise ValueError(f"unknown objective {objective}")
    return min(eligible, key=lambda r: tuple(r[k] for k in keys[objective])) if eligible else None


def evaluate_route(backend, path, belief, cfg, yaw0, timing, schedule=None):
    # Unlimited replay avoids mistaking a truncated failed route for a cheap one.
    trace = backend.replay(path, belief, cfg, yaw0, schedule)
    for row in trace:  # unlimited replay: per-row bound fields are meaningless
        for key in ("position_threshold_m", "yaw_threshold_rad", "feasible",
                    "position_limit_effective_m", "yaw_limit_effective_rad"):
            row.pop(key, None)
    contacts = [r for r in trace if r["contact"] != "none"]
    distance = backend.length(path)
    turn = sum(r["turn_rad"] for r in trace)
    components = timing.components(distance, turn, len(contacts))
    initial_pos = (trace[0]["source_position_sigma_m"] if trace
                   else position_sigma_max(belief.cov))
    initial_yaw = math.degrees(belief.yaw_sigma)
    resolved = (not cfg.require_landmark_for_mode_resolution
                or belief.minor <= 0.10 or any(r["contact"] == "landmark" for r in contacts))
    profile = dict(pre_pos=[max(r["source_position_sigma_m"], r["pre_contact_position_sigma_m"])
                            for r in trace],
                   post_pos=[r["position_sigma_m"] for r in trace],
                   pre_yaw_deg=[math.degrees(r["pre_contact_yaw_sigma_rad"]) for r in trace],
                   post_yaw_deg=[r["yaw_sigma_deg"] for r in trace],
                   distance=[r["distance_m"] for r in trace])
    metrics = dict(position_constraint=cfg.position_constraint,
                   distance_m=float(distance), commanded_turn_rad=float(turn),
                   probe_count=len(contacts), **components,
                   estimated_time_s=sum(components.values()),
                   peak_position_sigma_m=max([initial_pos] + [r["max_position_sigma_m"] for r in trace]),
                   peak_yaw_sigma_deg=max([initial_yaw] + [r["max_yaw_sigma_deg"] for r in trace]),
                   initial_position_sigma_m=initial_pos, initial_yaw_sigma_deg=initial_yaw,
                   grace_position_m=cfg.initial_position_grace_m,
                   grace_yaw_deg=math.degrees(cfg.initial_yaw_grace_rad),
                   association_resolved=bool(resolved), profile=profile)
    return metrics, trace


def prune_schedule(evaluate, dense_trace, position_limit, yaw_limit_deg, reverse=False):
    """Remove redundant contacts; every trial replays downstream uncertainty.

    The result is deletion-minimal for this order, not minimum-cardinality.
    """
    schedule = frozenset((r["edge_index"], r["step"]) for r in dense_trace if r["contact"] != "none")
    changed = True
    while changed:
        changed = False
        for action in sorted(schedule, reverse=reverse):
            trial = schedule - {action}
            metrics, _ = evaluate(trial)
            if feasible(metrics, position_limit, yaw_limit_deg):
                schedule = trial
                changed = True
    return schedule


def timing_sensitivity(candidates, position_limits, yaw_limits_deg, timing,
                       probe_seconds_values=(0.0, 2.0, 5.0, 10.0, 20.0)):
    """Re-select the fastest feasible candidate under alternative probe durations.

    Distance, turn and probe count are fixed per candidate, so this is a pure
    re-weighting of the objective; the uncertainty model is untouched.
    """
    rows = []
    for p in position_limits:
        for y in yaw_limits_deg:
            eligible = [r for r in candidates if feasible(r, p, y)]
            shortest = select_candidate(eligible, p, y, "distance")
            fewest = select_candidate(eligible, p, y, "probes")
            for probe_seconds in probe_seconds_values:
                alt = replace(timing, probe_seconds=probe_seconds)
                scored = []
                for r in eligible:
                    parts = alt.components(r["distance_m"], r["commanded_turn_rad"], r["probe_count"])
                    scored.append(dict(r, **parts, estimated_time_s=sum(parts.values())))
                best = select_candidate(scored, p, y, "time")
                rows.append(dict(position_limit_m=p, yaw_limit_deg=y, probe_seconds=probe_seconds,
                                 fastest_candidate_id=best["candidate_id"] if best else None,
                                 fastest_time_s=best["estimated_time_s"] if best else None,
                                 fastest_distance_m=best["distance_m"] if best else None,
                                 fastest_probe_count=best["probe_count"] if best else None,
                                 shortest_candidate_id=shortest["candidate_id"] if shortest else None,
                                 fewest_probe_candidate_id=fewest["candidate_id"] if fewest else None,
                                 fastest_is_shortest=bool(best and shortest and best["candidate_id"] == shortest["candidate_id"]),
                                 fastest_is_fewest_probes=bool(best and fewest and best["candidate_id"] == fewest["candidate_id"])))
    return rows


def slack(metrics, position_limit, yaw_limit_deg) -> float:
    """Smallest relative headroom to the bounds along the whole route
    (negative when violated); the profile the localize-first check uses."""
    prof = metrics["profile"]
    pos = prof["pre_pos"] + prof["post_pos"]
    yaw = prof["pre_yaw_deg"] + prof["post_yaw_deg"]
    if not pos:
        return 1.0
    return min(1.0 - max(pos) / position_limit, 1.0 - max(yaw) / yaw_limit_deg)


def prune_schedule_balanced(evaluate, dense_trace, position_limit, yaw_limit_deg, margin=0.0):
    """Thin the contacts while keeping them SPREAD OUT.

    Greedy deletion of whichever contact leaves the most headroom to the
    bounds, against limits shrunk by margin (a fraction), until no
    deletion is feasible.  Deleting in route order instead ()
    defers every contact until the belief sits on the bound and then needs
    one at every step: sparse, then dense.  This keeps the belief evenly
    below the bound so the executed run has headroom everywhere.
    O(n^2) replays; n is the dense contact count.
    """
    p_eff, y_eff = position_limit * (1.0 - margin), yaw_limit_deg * (1.0 - margin)
    schedule = frozenset((r["edge_index"], r["step"]) for r in dense_trace if r["contact"] != "none")
    if not feasible(evaluate(schedule)[0], p_eff, y_eff):
        return None                     # even dense contact cannot hold the shrunk bound
    while schedule:
        best = None
        for action in sorted(schedule):
            trial = schedule - {action}
            metrics, _ = evaluate(trial)
            if not feasible(metrics, p_eff, y_eff):
                continue
            s = slack(metrics, p_eff, y_eff)
            if best is None or s > best[0]:
                best = (s, trial)
        if best is None:
            break
        schedule = best[1]
    return schedule


def run_sweep(backend, belief, cfg, position_limits, yaw_limits_deg,
              turn_weights=(0.0, 0.5, 1.0), yaw0=0.0, timing=Timing(), progress=None):
    """Generate routes, prune under every bound pair, then cross-evaluate all.

    Sharing the candidate pool across bounds prevents separately sampled pools
    from manufacturing nonmonotone best-time results as limits loosen.
    """
    for values in (position_limits, yaw_limits_deg):
        if not values or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("uncertainty limits must be finite and positive")
    if not turn_weights or any(not math.isfinite(v) or v < 0 for v in turn_weights):
        raise ValueError("turn weights must be finite and nonnegative")
    if not math.isfinite(yaw0):
        raise ValueError("initial heading must be finite")
    cfg = replace(cfg, max_relaxations=0)
    sweep_start = time.perf_counter()
    attempts, routes = [], {}
    for p in position_limits:
        for y in yaw_limits_deg:
            for w in turn_weights:
                current = replace(cfg, position_threshold_m=p, yaw_threshold_rad=math.radians(y),
                                  turn_cost_per_rad=w)
                start = time.perf_counter()
                plan = backend.plan(belief, current, yaw0)
                attempts.append(dict(position_limit_m=p, yaw_limit_deg=y, turn_weight=w,
                                     feasible=plan.feasible, reason=plan.reason,
                                     explored_states=plan.explored_states,
                                     planning_seconds=time.perf_counter() - start))
                if plan.feasible:
                    routes.setdefault(tuple(tuple(x) if isinstance(x, (list, tuple)) else x
                                            for x in plan.path), len(routes))
            if progress:
                progress(f"bounds {p:g} m / {y:g} deg: {len(routes)} distinct routes")

    candidates, seen = [], set()
    for path, route_id in routes.items():
        cache = {}

        def evaluate(schedule):
            if schedule not in cache:
                cache[schedule] = evaluate_route(backend, list(path), belief, cfg, yaw0, timing, schedule)
            return cache[schedule]

        dense_metrics, dense_trace = evaluate(None)
        dense_schedule = frozenset((r["edge_index"], r["step"]) for r in dense_trace if r["contact"] != "none")
        cache[dense_schedule] = (dense_metrics, dense_trace)

        def add(schedule, policy, p=None, y=None):
            signature = (path, schedule)
            if signature in seen:
                return
            seen.add(signature)
            metrics, trace = evaluate(schedule)
            candidates.append(dict(candidate_id=f"c{len(candidates):04d}", route_id=route_id,
                                   policy=policy, generated_position_limit_m=p,
                                   generated_yaw_limit_deg=y, path=[list(x) if isinstance(x, tuple) else x for x in path],
                                   probe_schedule=[list(a) for a in sorted(schedule)],
                                   **metrics, trace=trace))

        add(dense_schedule, "dense")
        add(frozenset(), "no_probe")
        for p in position_limits:
            for y in yaw_limits_deg:
                if not feasible(dense_metrics, p, y):
                    continue
                for reverse in (False, True):
                    schedule = prune_schedule(evaluate, dense_trace, p, y, reverse)
                    add(schedule, "pruned_backward" if reverse else "pruned_forward", p, y)
        if progress:
            progress(f"route {route_id}: {len(candidates)} distinct action schedules so far")

    selections = []
    for p in position_limits:
        for y in yaw_limits_deg:
            eligible = [r for r in candidates if feasible(r, p, y)]
            for objective in ("distance", "time", "probes"):
                best = select_candidate(eligible, p, y, objective)
                dense = select_candidate([r for r in eligible if r["policy"] == "dense"], p, y, objective)
                selections.append(dict(position_limit_m=p, yaw_limit_deg=y, objective=objective,
                                       status="feasible_candidate" if best else "no_feasible_candidate_found",
                                       candidate_id=best["candidate_id"] if best else None,
                                       localized_at_distance_m=localization(best, p, y)[1] if best else None,
                                       dense_candidate_id=dense["candidate_id"] if dense else None,
                                       feasible_candidates=len(eligible)))
    sensitivity = timing_sensitivity(candidates, position_limits, yaw_limits_deg, timing)
    plan_times = [a["planning_seconds"] for a in attempts]
    compute = dict(sweep_wall_seconds=time.perf_counter() - sweep_start,
                   planning_attempts=len(attempts),
                   planning_seconds_total=float(sum(plan_times)),
                   planning_seconds_mean=float(sum(plan_times) / max(len(plan_times), 1)),
                   planning_seconds_max=float(max(plan_times, default=0.0)),
                   explored_states_mean=float(sum(a["explored_states"] for a in attempts) / max(len(attempts), 1)))
    valid = [r for r in candidates if r["association_resolved"]]
    front = pareto_indices(valid, ("distance_m", "estimated_time_s", "probe_count",
                                  "peak_position_sigma_m", "peak_yaw_sigma_deg"))
    return dict(timing=asdict(timing), uncertainty_config=asdict(cfg),
                position_limits=list(position_limits), yaw_limits_deg=list(yaw_limits_deg),
                turn_weights=list(turn_weights), yaw0_rad=yaw0,
                initial_covariance=belief.cov.tolist(), initial_yaw_sigma_rad=belief.yaw_sigma,
                initial_minor_mass=belief.minor, attempts=attempts, candidates=candidates,
                selections=selections, timing_sensitivity=sensitivity, compute=compute, pareto_candidate_ids=[valid[i]["candidate_id"] for i in front])
