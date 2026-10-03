"""The stressed benchmarks.  Each one is a wrong turn waiting to happen.

TWIN_JUNCTIONS -- aliasing all the way to the commit point.  Three identical
junction mouths at x = 4, 7, 10; the route takes the middle one; the outer two
are dead-end decoys.  The prior is 50/50 over two positions 3.0 m apart -- the
robot has just anchored on a wall terminus and the map offers two identical
places that terminus could be (Batch 1's ambiguity, now with consequences).  The
mouth spacing EQUALS the hypothesis spacing, so every north-wall probe is aliased:
gap answers gap, wall answers wall, all the way to the junction.  Touch that
follows the shortest path learns nothing; the no-touch robot turns into a decoy
about half the time BY CONSTRUCTION.  The single de-aliasing surface is a pillar
on the south wall -- unique, off the shortest path, and out of reach from the
centerline (0.55 m > 0.45 m): disambiguation must be paid for with a detour.
Probing where the pillar SHOULD be resolves both ways: contact kills the far
hypothesis, swept-free kills the near one.

DRIFT_GAUNTLET -- uncertainty outgrows the junction spacing.  A 12 m approach
with a loose along-corridor prior (the map's registration to the entry door is
poor) ending at two openings 1.6 m apart: the first is a dead end, the second is
the route.  Arrival sigma_s is ~1.0 m against a 0.8 m half-spacing, so committing
on dead reckoning picks the wrong opening for a meaningful fraction of seeds.  A
box on the south wall mid-corridor and the junction furniture itself are the only
along-route anchors; everything between is featureless and rank-1.

GAUNTLET_SLIP -- the same corridor, plus a 1.0 m un-odometered slip after the box
anchor.  The belief is tight and confidently wrong.  Rank-1 physics delays
detection -- side probes on a uniform wall explain themselves perfectly whatever
x is -- so nothing fires until the junction furniture disagrees, which is exactly
when SUSPECT_LOST -> RELOCALIZE must trigger, re-seed, and recover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .affordance import AffordanceGraph
from .floorplan import Floorplan, MapFeature, RouteEdge, rect

__all__ = ["Scenario", "twin_junctions", "drift_gauntlet", "SCENARIOS", "build_scenario"]

WALL_T = 0.15
HW = 0.8


@dataclass(frozen=True)
class Scenario:
    name: str
    plan: Floorplan
    graph: AffordanceGraph
    goal_xy: tuple[float, float]
    goal_node: str
    goal_radius_m: float
    prior_modes: tuple | None            # ((xy, theta, edge, mass), ...) or None
    prior_mean: tuple[float, float]
    prior_sigma_xy: tuple[float, float]
    prior_sigma_theta: float
    truth_of_seed: Callable
    reg_true: tuple[float, float] = (0.06, -0.04)
    slip: tuple[float, float, float] | None = None   # (s_trigger, dx, dy)
    sigma_branch_max_m: float = 0.45
    junction_spacing_m: float = 3.0
    drift_per_m: float = 0.06
    decoy_edges: tuple[int, ...] = ()
    route_junctions: tuple[str, ...] = ()            # branch names the route takes, in order


# ── twin_junctions ───────────────────────────────────────────────────────────────


def _twin_plan() -> Floorplan:
    gaps = ((3.2, 4.8), (6.2, 7.8), (9.2, 10.8))     # J1 decoy, J2 route, J3 decoy
    depth = {0: 3.8, 1: 4.2, 2: 3.8}                 # dead ends far beyond reach
    bodies = [
        rect(-0.75, 12.75, -0.95, -HW, "south"),
        rect(-0.75, -0.60, -HW, HW, "west_cap"),
        rect(12.60, 12.75, -HW, HW, "east_cap"),
        rect(-0.75, gaps[0][0], HW, HW + WALL_T, "north_0"),
        rect(gaps[0][1], gaps[1][0], HW, HW + WALL_T, "north_1"),
        rect(gaps[1][1], gaps[2][0], HW, HW + WALL_T, "north_2"),
        rect(gaps[2][1], 12.75, HW, HW + WALL_T, "north_3"),
        rect(5.2, 5.8, -HW, -0.50, "pillar", signature="pillar"),
    ]
    for k, (x0, x1) in enumerate(gaps):
        top = depth[k]
        bodies += [
            rect(x0 - WALL_T, x0, HW, top + WALL_T, f"B{k+1}_west"),
            rect(x1, x1 + WALL_T, HW, top + WALL_T, f"B{k+1}_east"),
            rect(x0 - WALL_T, x1 + WALL_T, top, top + WALL_T, f"B{k+1}_cap"),
        ]
    features = []
    for x0, x1 in gaps:                              # six aliased termini
        features += [(x0, HW), (x1, HW)]
    features = tuple(
        MapFeature(i, xy, "gap_edge", f"gap_edge_{i}") for i, xy in enumerate(features)
    ) + (
        MapFeature(6, (5.2, -0.50), "pillar_corner", "pillar_w"),
        MapFeature(7, (5.8, -0.50), "pillar_corner", "pillar_e"),
    )
    edges = (
        RouteEdge(0, (0.0, 0.0), (4.0, 0.0), HW, (1, 4), "A0"),
        RouteEdge(1, (4.0, 0.0), (7.0, 0.0), HW, (5, 2), "A1"),
        RouteEdge(2, (7.0, 0.0), (10.0, 0.0), HW, (6, 3), "A2"),
        RouteEdge(3, (10.0, 0.0), (12.3, 0.0), HW, (), "A3"),
        RouteEdge(4, (4.0, 0.0), (4.0, 3.6), HW, (), "B1_decoy"),
        RouteEdge(5, (7.0, 0.0), (7.0, 3.5), HW, (), "B2_route"),
        RouteEdge(6, (10.0, 0.0), (10.0, 3.6), HW, (), "B3_decoy"),
    )
    return Floorplan(tuple(bodies), features, edges, route=(0, 1, 5), name="twin_junctions")


def _twin_graph(plan: Floorplan) -> AffordanceGraph:
    g = AffordanceGraph(plan)
    ny = HW - 0.28                                    # north wall-adjacent lane
    center = [("c26", 2.6), ("c40", 4.0), ("c55", 5.5), ("c70", 7.0),
              ("c85", 8.5), ("c100", 10.0), ("c115", 11.5)]
    north = [("n30", 3.0), ("n40", 4.0), ("n50", 5.0), ("n62", 6.2), ("n70", 7.0),
             ("n78", 7.8), ("n85", 8.5), ("n100", 10.0)]
    for name, x in center:
        g.add_node(name, (x, 0.0), "junction" if x in (4.0, 7.0, 10.0) else "way")
    for name, x in north:
        g.add_node(name, (x, ny), "station")
    g.add_node("p55", (5.5, -0.20), "station")
    for k, x in (("1", 4.0), ("2", 7.0), ("3", 10.0)):
        g.add_node(f"b{k}a", (x, 1.6))
        g.add_node(f"b{k}b", (x, 3.4 if k == "2" else 3.3),
                   "goal" if k == "2" else "way")
    for (a, _), (b, _) in zip(center, center[1:]):
        g.connect(a, b, "transit")
    for (a, _), (b, _) in zip(north, north[1:]):
        g.connect(a, b, "wall")
    for a, b in (("c26", "n30"), ("c40", "n40"), ("c70", "n70"),
                 ("c85", "n85"), ("c100", "n100")):
        g.connect(a, b, "transit")
    g.connect("c55", "p55", "detour")
    for k, cn, nn in (("1", "c40", "n40"), ("2", "c70", "n70"), ("3", "c100", "n100")):
        g.connect(cn, f"b{k}a", "transit", branch_junction=f"B{k}")
        g.connect(nn, f"b{k}a", "transit", branch_junction=f"B{k}")
        g.connect(f"b{k}a", f"b{k}b", "transit")
    for name, _ in north:
        purpose = "mouth" if name in ("n40", "n70", "n100") else "wall"
        g.add_station(name, 0.0, "left_g1", purpose)
    g.add_station("p55", 0.0, "right_g1", "feature")
    return g


def twin_junctions() -> Scenario:
    plan = _twin_plan()
    modes = (((2.6, 0.0), 0.0, 0, 0.5), ((5.6, 0.0), 0.0, 1, 0.5))

    def truth_of_seed(seed: int, rng: np.random.Generator):
        m = modes[seed % 2][0]
        return (np.asarray(m, float) + rng.normal(0.0, 0.05, 2), 0.0)

    return Scenario(
        name="twin_junctions", plan=plan, graph=_twin_graph(plan),
        goal_xy=(7.0, 3.4), goal_node="b2b", goal_radius_m=0.70,
        prior_modes=modes, prior_mean=(4.1, 0.0), prior_sigma_xy=(0.12, 0.10),
        prior_sigma_theta=0.03, truth_of_seed=truth_of_seed,
        sigma_branch_max_m=0.60, junction_spacing_m=3.0,
        decoy_edges=(4, 6), route_junctions=("B2",),
    )


# ── drift_gauntlet ───────────────────────────────────────────────────────────────


def _gauntlet_plan() -> Floorplan:
    gaps = ((10.0, 11.2), (11.6, 12.8))               # D decoy, R route -- 1.6 m apart
    bodies = [
        rect(-0.75, 14.75, -0.95, -HW, "south"),
        rect(-0.75, -0.60, -HW, HW, "west_cap"),
        rect(14.60, 14.75, -HW, HW, "east_cap"),
        rect(-0.75, gaps[0][0], HW, HW + WALL_T, "north_0"),
        rect(gaps[0][1], gaps[1][0], HW, HW + WALL_T, "north_div"),
        rect(gaps[1][1], 14.75, HW, HW + WALL_T, "north_1"),
        rect(6.2, 6.8, -HW, -0.50, "box", signature="box"),
        # D decoy corridor
        rect(gaps[0][0] - WALL_T, gaps[0][0], HW, 3.55, "D_west"),
        rect(gaps[0][1], gaps[0][1] + WALL_T, HW, 3.55, "D_east"),
        rect(gaps[0][0] - WALL_T, gaps[0][1] + WALL_T, 3.40, 3.55, "D_cap"),
        # R route corridor
        rect(gaps[1][0] - WALL_T, gaps[1][0], HW, 3.75, "R_west"),
        rect(gaps[1][1], gaps[1][1] + WALL_T, HW, 3.75, "R_east"),
        rect(gaps[1][0] - WALL_T, gaps[1][1] + WALL_T, 3.60, 3.75, "R_cap"),
    ]
    features = tuple(
        MapFeature(i, xy, "gap_edge", f"gap_edge_{i}")
        for i, xy in enumerate([(x, HW) for gap in gaps for x in gap])
    ) + (
        MapFeature(4, (6.2, -0.50), "box_corner", "box_w"),
        MapFeature(5, (6.8, -0.50), "box_corner", "box_e"),
    )
    edges = (
        RouteEdge(0, (0.0, 0.0), (10.6, 0.0), HW, (1, 3), "A0"),
        RouteEdge(1, (10.6, 0.0), (12.2, 0.0), HW, (4, 2), "A1"),
        RouteEdge(2, (12.2, 0.0), (14.3, 0.0), HW, (), "A2"),
        RouteEdge(3, (10.6, 0.0), (10.6, 3.2), 0.6, (), "D_decoy"),
        RouteEdge(4, (12.2, 0.0), (12.2, 3.4), 0.6, (), "R_route"),
    )
    return Floorplan(tuple(bodies), features, edges, route=(0, 1, 4), name="drift_gauntlet")


def _gauntlet_graph(plan: Floorplan) -> AffordanceGraph:
    g = AffordanceGraph(plan)
    ny = HW - 0.28
    center = [("c05", 0.5), ("c30", 3.0), ("c65", 6.5), ("c90", 9.0),
              ("c106", 10.6), ("c122", 12.2), ("c135", 13.5)]
    # n135 is the de-aliasing station: the junction furniture itself (two
    # identical gaps 1.6 m apart) is aliased under +-1.6 m shifts BY DESIGN, so a
    # belief slipped by ~one spacing re-anchors one junction off and every probe
    # inside the cluster confirms it.  A wall reading at 13.4 differs from the
    # R-mouth free reading at 11.8, and that is the only lane evidence east of
    # the cluster that can break the tie.
    north = [("n10", 1.0), ("n30", 3.0), ("n50", 5.0), ("n70", 7.0),
             ("n90", 9.0), ("n100", 10.0), ("n106", 10.6), ("n114", 11.4),
             ("n122", 12.2), ("n128", 12.8), ("n135", 13.4)]
    for name, x in center:
        g.add_node(name, (x, 0.0), "junction" if x in (10.6, 12.2) else "way")
    for name, x in north:
        g.add_node(name, (x, ny), "station")
    g.add_node("p65", (6.5, -0.20), "station")
    g.add_node("d1", (10.6, 1.5))
    g.add_node("d2", (10.6, 3.0))
    g.add_node("r1", (12.2, 1.5))
    g.add_node("goal", (12.2, 2.9), "goal")
    for (a, _), (b, _) in zip(center, center[1:]):
        g.connect(a, b, "transit")
    for (a, _), (b, _) in zip(north, north[1:]):
        g.connect(a, b, "wall")
    for a, b in (("c05", "n10"), ("c30", "n30"), ("c90", "n90"),
                 ("c106", "n106"), ("c122", "n122"), ("c135", "n135")):
        g.connect(a, b, "transit")
    g.connect("c65", "p65", "detour")
    for cn, nn, first, jname in (("c106", "n106", "d1", "D"), ("c122", "n122", "r1", "R")):
        g.connect(cn, first, "transit", branch_junction=jname)
        g.connect(nn, first, "transit", branch_junction=jname)
    g.connect("d1", "d2", "transit")
    g.connect("r1", "goal", "transit")
    for name, x in north:
        purpose = "mouth" if name in ("n106", "n122") else "wall"
        g.add_station(name, 0.0, "left_g1", purpose)
    g.add_station("p65", 0.0, "right_g1", "feature")
    return g


def drift_gauntlet(slip: tuple[float, float, float] | None = None,
                   name: str = "drift_gauntlet",
                   prior_sigma_x: float = 0.60) -> Scenario:
    plan = _gauntlet_plan()
    sx = float(prior_sigma_x)

    def truth_of_seed(seed: int, rng: np.random.Generator):
        # the belief's prior spread IS the truth's draw: the map's registration to
        # the entry door is genuinely that uncertain.  The lower clip keeps the
        # start a body-radius clear of the west cap -- a draw at the cap face left
        # the robot wedged inside its own bump standoff, unable to take step one.
        dx = float(np.clip(rng.normal(0.0, sx), max(-1.9 * sx, -0.70), 2.6 * sx))
        dy = float(np.clip(rng.normal(0.0, 0.10), -0.25, 0.25))
        return (np.array([0.5 + dx, dy]), 0.0)

    return Scenario(
        name=name, plan=plan, graph=_gauntlet_graph(plan),
        goal_xy=(12.2, 2.9), goal_node="goal", goal_radius_m=0.70,
        prior_modes=None, prior_mean=(0.5, 0.0), prior_sigma_xy=(sx, 0.10),
        prior_sigma_theta=0.03, truth_of_seed=truth_of_seed,
        slip=slip, sigma_branch_max_m=0.32, junction_spacing_m=1.6,
        decoy_edges=(3,), route_junctions=("R",),
    )


SCENARIOS = ("twin_junctions", "drift_gauntlet", "gauntlet_slip")


def build_scenario(name: str) -> Scenario:
    if name == "twin_junctions":
        return twin_junctions()
    if name == "drift_gauntlet":
        return drift_gauntlet()
    if name == "gauntlet_slip":
        # TIGHT prior, then a 0.9 m un-odometered slip: the belief must be
        # confidently wrong, not merely wide -- a wide belief would explain the
        # junction contacts and absorb the slip without ever being "lost".
        #
        # 0.9 m is deliberately OFF the junction furniture's alias spectrum
        # (gap-edge pairwise deltas: 0.4, 1.2, 1.6 m).  A slip AT an alias
        # distance was measured to be near-unrecoverable from local evidence:
        # the belief re-anchors exactly one junction off, every cluster probe
        # confirms the alias, and relocalization "recovers" onto it.  That
        # failure mode is real and documented in the report; the validation
        # scenario tests the recovery machinery, not the impossible case.
        return drift_gauntlet(slip=(8.5, 0.9, 0.0), name="gauntlet_slip",
                              prior_sigma_x=0.12)
    raise KeyError(name)
