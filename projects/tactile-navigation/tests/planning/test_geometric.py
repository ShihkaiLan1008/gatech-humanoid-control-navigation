"""Geometric route generation on a known map (Subtask 5)."""

from __future__ import annotations

import numpy as np
import pytest

from lane_graph.maps_v3 import MapV3, Wall, clearance
from lane_graph.scene3 import build_v3_scene
from planning import OccupancyGrid, RouteConfig, plan_alternatives, plan_route

BOUNDS = (-0.5, -0.5, 6.5, 4.5)


def two_room(gap_m: float) -> MapV3:
    """Two rooms joined by one doorway of a chosen width."""
    y0, y1, xd = 0.0, 4.0, 3.0
    h = gap_m / 2.0
    walls = [Wall((0, y0), (6, y0)), Wall((0, y1), (6, y1)),
             Wall((0, y0), (0, y1)), Wall((6, y0), (6, y1)),
             Wall((xd, y0), (xd, 2.0 - h)), Wall((xd, 2.0 + h), (xd, y1))]
    return MapV3(name="twin_rooms", title="twin", walls=walls, bulges=[], prots=[],
                 rooms=[(0, y0, 6, y1)], start=(0.8, 2.0), goal=(5.2, 2.0),
                 goal_r=0.25, doors=[(xd, 2.0, "door")], landmarks=[],
                 topology="two rooms, one door", story="")


def cf(m: MapV3):
    return lambda P: clearance(P, m)


# ── routes exist on every shipped map, without hand authoring ────────────────────

@pytest.mark.parametrize("name", ["r1_gallery", "r2_lbend", "r3_tpinch"])
def test_route_is_found_on_every_shipped_map(name):
    sc = build_v3_scene(name)
    m = sc.spec
    xs = [v for r in m.rooms for v in (r[0], r[2])]
    ys = [v for r in m.rooms for v in (r[1], r[3])]
    bounds = (min(xs) - 0.5, min(ys) - 0.5, max(xs) + 0.5, max(ys) + 0.5)
    r = plan_route(cf(m), bounds, sc.start_map_xy, sc.goal_map_xy)
    assert r.found, r.reason
    assert r.min_clearance_m >= RouteConfig().robot_clearance_m - 1e-9
    assert np.allclose(r.points[0], sc.start_map_xy, atol=1e-6)
    assert np.allclose(r.points[-1], sc.goal_map_xy, atol=1e-6)


# ── the clearance contract is the safety contract ───────────────────────────────

def test_a_gap_the_robot_fits_through_is_routed():
    r = plan_route(cf(two_room(1.20)), BOUNDS, (0.8, 2.0), (5.2, 2.0))
    assert r.found
    assert r.min_clearance_m >= RouteConfig().robot_clearance_m


def test_a_gap_the_robot_does_not_fit_through_is_refused():
    """0.72 m between wall centrelines is 0.62 m of free width once the 0.10 m
    wall thickness is taken off -- less than 2 x 0.34 m.  Refusing is correct;
    squeezing through would be a route the base cannot execute."""
    r = plan_route(cf(two_room(0.72)), BOUNDS, (0.8, 2.0), (5.2, 2.0))
    assert not r.found
    assert "clearance" in r.reason


def test_no_route_never_returns_a_partial_path():
    r = plan_route(cf(two_room(0.40)), BOUNDS, (0.8, 2.0), (5.2, 2.0))
    assert not r.found and len(r.points) == 0 and r.length_m == 0.0


# ── the route is smooth, which is what the controller has to track ──────────────

def test_the_route_is_smooth_not_a_zig_zag():
    """Station hopping must not redefine the base trajectory.  Measured as the
    turn angle between consecutive segments: a grid path has 45/90 deg corners,
    a smoothed one does not."""
    r = plan_route(cf(two_room(1.40)), BOUNDS, (0.8, 2.0), (5.2, 2.0))
    d = np.diff(r.points, axis=0)
    d = d[np.linalg.norm(d, axis=1) > 1e-9]
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    turns = np.degrees(np.arccos(np.clip(np.sum(u[:-1] * u[1:], axis=1), -1, 1)))
    assert float(np.max(turns)) < 40.0, f"max turn {np.max(turns):.1f} deg"


def test_route_points_are_evenly_spaced():
    r = plan_route(cf(two_room(1.40)), BOUNDS, (0.8, 2.0), (5.2, 2.0))
    seg = np.linalg.norm(np.diff(r.points, axis=0), axis=1)
    assert seg.std() < 0.02 * max(seg.mean(), 1e-6) + 1e-6


# ── endpoints near walls are legitimate: the robot ends up near things ──────────

def test_a_goal_inside_the_inflation_band_snaps_to_free_space():
    m = two_room(1.40)
    grid = OccupancyGrid(cf(m), BOUNDS, RouteConfig())
    tight = (5.85, 2.0)                     # 0.15 m from the east wall
    assert not grid.free[grid.to_cell(tight)]
    assert grid.nearest_free(tight) is not None
    r = plan_route(cf(m), BOUNDS, (0.8, 2.0), tight)
    assert r.found


# ── several routes, so distance alone does not decide tactile behaviour ─────────

def test_alternatives_are_genuinely_different_routes():
    """A corridor and a detour, not the same corridor twice."""
    y0, y1 = 0.0, 6.0
    walls = [Wall((0, y0), (8, y0)), Wall((0, y1), (8, y1)),
             Wall((0, y0), (0, y1)), Wall((8, y0), (8, y1)),
             Wall((3.0, 2.2), (5.0, 2.2)), Wall((3.0, 3.8), (5.0, 3.8))]
    m = MapV3(name="two_ways", title="two ways", walls=walls, bulges=[], prots=[],
              rooms=[(0, y0, 8, y1)], start=(1.0, 3.0), goal=(7.0, 3.0),
              goal_r=0.25, doors=[], landmarks=[], topology="split", story="")
    routes = plan_alternatives(cf(m), (-0.5, -0.5, 8.5, 6.5), (1.0, 3.0), (7.0, 3.0), n=2)
    assert len(routes) >= 2
    a, b = routes[0], routes[1]
    mid_a = a.points[len(a.points) // 2]
    mid_b = b.points[len(b.points) // 2]
    assert float(np.linalg.norm(mid_a - mid_b)) > 0.8, "alternatives coincide"


# ── the planner must not silently produce something unusable ────────────────────

def test_an_absurd_bounding_box_is_refused_not_thrashed_on():
    with pytest.raises(ValueError, match="max_cells"):
        OccupancyGrid(cf(two_room(1.4)), (-500, -500, 500, 500), RouteConfig())


# ── the tactile preferences: touch band, then features ───────────────────────────

def _clear_along(m, pts):
    return clearance(np.asarray(pts, float), m)


def test_the_default_cost_pushes_a_route_out_of_the_touchable_band():
    """The preference the module shipped with is a wall REPULSION: maximal at
    `robot_clearance_m`, zero past `prefer_clearance_m` = 0.60 m.  With a
    0.45 m arm that is the exact inverse of what a tactile robot wants, and
    this pins it as the baseline the touch band is measured against."""
    m = two_room(1.4)
    r = plan_route(cf(m), BOUNDS, (0.8, 2.0), (5.2, 2.0), RouteConfig())
    assert r.found
    assert np.median(_clear_along(m, r.points)) > 0.60


def test_the_touch_band_moves_the_route_toward_the_surfaces():
    """The claim is comparative and holds on any map: under the touch band a
    route runs nearer to something than it does under the repulsion.  How
    MUCH nearer is geometry -- in a 6x4 hall with a central door, hugging a
    wall costs more detour than the saturated out-of-reach price is worth,
    and the planner is right to decline it."""
    m = two_room(1.4)
    a = plan_route(cf(m), BOUNDS, (0.8, 2.0), (5.2, 2.0), RouteConfig())
    cfg = RouteConfig(touch_reach_m=0.45)
    b = plan_route(cf(m), BOUNDS, (0.8, 2.0), (5.2, 2.0), cfg)
    assert a.found and b.found
    assert np.median(_clear_along(m, b.points)) \
        <= np.median(_clear_along(m, a.points))
    assert b.min_clearance_m >= cfg.robot_clearance_m - 1e-9


def test_the_shortcut_may_not_undo_what_the_cost_just_bought():
    """A collision-only line-of-sight shortcut takes whatever chord fits,
    which on a touch-band cost throws the route away -- measured on room2,
    where A* returned the wall-following L and the shortcut replaced it with
    the open-space diagonal.  The invariant that forbids it: shortcutting
    never RAISES the path's cost in A*'s own currency."""
    from planning.geometric import _penalized_length, shortcut

    def cost(grid, pts):
        return sum(_penalized_length(grid, a, b)
                   for a, b in zip(pts[:-1], pts[1:]))

    m = two_room(1.4)
    for cfg in (RouteConfig(), RouteConfig(touch_reach_m=0.45)):
        grid = OccupancyGrid(cf(m), BOUNDS, cfg)
        raw = plan_route(cf(m), BOUNDS, (0.8, 2.0), (5.2, 2.0), cfg).points
        assert cost(grid, shortcut(grid, raw)) <= cost(grid, raw) + 1e-9


def test_a_stance_beside_a_feature_is_on_the_side_the_arm_can_see():
    """Both sides of a wall are `free` to a clearance function, and the far
    side is often the NEARER one.  A stance is defined by what the arm can
    reach without going through material, not by proximity."""
    from planning.geometric import feature_stances
    m = two_room(1.4)
    cfg = RouteConfig(touch_reach_m=0.45)
    grid = OccupancyGrid(cf(m), BOUNDS, cfg)
    jamb = np.array([[3.0, 2.0 - 0.7]])          # the doorway's lower jamb
    inside = plan_route(cf(m), BOUNDS, (0.8, 2.0), (5.2, 2.0), cfg).points
    (st,) = feature_stances(grid, jamb, cfg, near=inside)
    assert st is not None
    assert grid.arm_can_see(st, st + 0.85 * (jamb[0] - st))
    assert np.linalg.norm(st - jamb[0]) <= cfg.robot_clearance_m + cfg.touch_reach_m


def test_visiting_a_feature_is_a_budget_not_a_price():
    """A per-metre discount cannot pull a route to a sparse feature; a
    bounded detour can, and the bound is what stops it touring the room."""
    from planning.geometric import plan_route_via_features
    m = two_room(1.4)
    cfg = RouteConfig(touch_reach_m=0.45)
    grid = OccupancyGrid(cf(m), BOUNDS, cfg)
    far = np.array([[3.0, 3.9]])                 # a feature off the direct line
    start, goal = (0.8, 2.0), (5.2, 2.0)
    direct = plan_route(cf(m), BOUNDS, start, goal, cfg)

    generous = plan_route_via_features(grid, start, goal, far, cfg,
                                       detour_budget=3.0)
    stingy = plan_route_via_features(grid, start, goal, far, cfg,
                                     detour_budget=1.0)
    d_gen = np.linalg.norm(np.asarray(generous.points) - far[0], axis=1).min()
    d_dir = np.linalg.norm(np.asarray(direct.points) - far[0], axis=1).min()
    assert d_gen < d_dir                          # the budget bought the visit
    assert generous.length_m > direct.length_m    # and it was paid for
    assert stingy.reason == ""                    # too dear: the direct route
    assert stingy.length_m == pytest.approx(direct.length_m)
