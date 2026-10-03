"""Grid formulation of the constrained planner (`lane_graph.grid_constrained`).

Pins the contracts that make the grid a drop-in for the roadmap: cells inside
the collision margin are never traversable, touch regions sit exactly in the
standoff band, moves respect the range limit, planning under loose bounds
reaches the goal, and the replay/schedule interface matches the roadmap's.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from lane_graph.constrained import PlannerBelief
from lane_graph.grid_constrained import GridBackend, build_grid, plan_grid, replay_grid
from lane_graph.scene3 import build_v3_scene
from lane_graph.tradeoff import evaluate_route, feasible, Timing
from lane_graph.uncertainty import DEFAULT


@pytest.fixture(scope="module")
def grid():
    return build_grid(build_v3_scene("r2_lbend"), DEFAULT)


def test_moves_respect_the_range_and_are_one_per_heading(grid):
    lengths = [L for _dx, _dy, L, _h in grid.moves]
    assert max(lengths) <= DEFAULT.max_move_m + 1e-9
    headings = [round(h, 6) for _dx, _dy, _L, h in grid.moves]
    assert len(set(headings)) == len(headings) == 16


def test_free_cells_keep_the_collision_margin(grid):
    min_clear = DEFAULT.contact_standoff_m - DEFAULT.contact_band_m
    assert grid.free.any()
    assert np.all(grid.clear[grid.free] >= min_clear - 1e-9)
    for (cx, cy, mi) in grid.move_ok:
        assert grid.free[cx, cy]
        dx, dy = grid.moves[mi][0], grid.moves[mi][1]
        assert grid.free[cx + dx, cy + dy]


def test_touch_regions_sit_in_the_standoff_band(grid):
    lo = DEFAULT.contact_standoff_m - DEFAULT.contact_band_m
    hi = DEFAULT.contact_standoff_m + DEFAULT.contact_band_m + 0.5 * DEFAULT.grid_resolution_m
    assert grid.contacts, "the L-room has walls, so some cells must be touch cells"
    for cell, cs in grid.contacts.items():
        assert grid.free[cell]
        for c in cs:
            assert lo - 1e-9 <= c["distance"] <= hi + 1e-9
            assert abs(np.linalg.norm(c["normal"]) - 1) < 1e-9
            assert c["kind"] in ("wall", "landmark")


def test_loose_bounds_reach_the_goal_and_replay_matches(grid):
    belief = PlannerBelief(minor=0, cov=np.eye(2) * 0.07**2, yaw_sigma=math.radians(3))
    cfg = DEFAULT.merged(position_threshold_m=0.5, yaw_threshold_rad=math.radians(30),
                         max_relaxations=0)
    plan = plan_grid(grid, belief, cfg, math.pi / 2)
    assert plan.feasible and plan.path[0] == grid.start and grid.is_goal(plan.path[-1])
    rows = replay_grid(grid, plan.path, belief, cfg, math.pi / 2)
    assert len(rows) == len(plan.path) - 1
    assert rows[-1]["distance_m"] == pytest.approx(plan.distance_m)
    assert all(r["max_position_sigma_m"] <= 0.5 + 1e-9 for r in rows)
    # Every consecutive pair is a legal move of the grid.
    for a, b in zip(plan.path[:-1], plan.path[1:]):
        assert math.hypot(b[0] - a[0], b[1] - a[1]) * grid.res <= cfg.max_move_m + 1e-9


def test_backend_evaluates_and_schedules_like_the_roadmap(grid):
    belief = PlannerBelief(minor=0, cov=np.eye(2) * 0.07**2, yaw_sigma=math.radians(3))
    cfg = DEFAULT.merged(position_threshold_m=0.5, yaw_threshold_rad=math.radians(30),
                         max_relaxations=0)
    backend = GridBackend(grid)
    plan = backend.plan(belief, cfg, math.pi / 2)
    dense, trace = evaluate_route(backend, plan.path, belief, cfg, math.pi / 2, Timing())
    none, _ = evaluate_route(backend, plan.path, belief, cfg, math.pi / 2, Timing(), frozenset())
    assert dense["distance_m"] == pytest.approx(backend.length(plan.path))
    assert none["probe_count"] == 0 and dense["probe_count"] >= none["probe_count"]
    assert none["peak_position_sigma_m"] >= dense["peak_position_sigma_m"]
    assert feasible(dense, 0.5, 30)
    # Wall-following paths can be longer; removing all contacts need not
    # remain feasible under the old distance-only route's loose bound.
    assert none["peak_position_sigma_m"] > dense["peak_position_sigma_m"]


def _wall_route_grid(with_detour=True):
    from types import SimpleNamespace
    nx, ny = 3, 2 if with_detour else 1
    moves = [(1, 0, 1., 0.), (-1, 0, 1., math.pi),
             (0, 1, 1., math.pi / 2), (0, -1, 1., -math.pi / 2)]
    edges = {(x, y, i): True for x in range(nx) for y in range(ny)
             for i, (dx, dy, *_rest) in enumerate(moves)
             if 0 <= x + dx < nx and 0 <= y + dy < ny}
    contacts = {(x, y): [dict(kind='wall', normal=(0., 1.), distance=.35)]
                for x in range(nx) for y in range(ny) if (x, y) != (1, 0)}
    cfg = DEFAULT.merged(position_threshold_m=10., yaw_threshold_rad=math.pi,
                         face_surface_on_touch=False, max_explored_states=10000)
    return SimpleNamespace(res=1., moves=moves, move_ok=edges, contacts=contacts,
                           free=np.ones((nx, ny), dtype=bool), cfg=cfg, start=(0, 0),
                           goal_xy=np.array([2., 0.]), goal_radius=.01,
                           is_goal=lambda c: c == (2, 0),
                           cell_xy=lambda c: np.asarray(c, float),
                           cell_of=lambda p: tuple(np.rint(p).astype(int)))


def test_loose_uncertainty_does_not_choose_open_shortcut():
    grid = _wall_route_grid()
    belief = PlannerBelief(minor=0, cov=np.eye(2)*.03**2, yaw_sigma=.02)
    plan = plan_grid(grid, belief, yaw0=0.)
    assert plan.feasible
    assert plan.distance_m == pytest.approx(4.)  # the open shortcut is only 2 m
    assert plan.free_distance_m == 0.
    assert (1, 0) not in plan.path


def test_unavoidable_gap_is_still_traversable():
    grid = _wall_route_grid(with_detour=False)
    belief = PlannerBelief(minor=0, cov=np.eye(2)*.03**2, yaw_sigma=.02)
    plan = plan_grid(grid, belief, yaw0=0.)
    assert plan.feasible and plan.path == [(0, 0), (1, 0), (2, 0)]
    assert plan.free_distance_m > 0.


@pytest.mark.parametrize("bound, distance", [(.14, 2.), (.055, 4.)])
def test_distance_objective_takes_shortcut_only_when_bound_allows(bound, distance):
    grid = _wall_route_grid()
    cfg = grid.cfg.merged(route_objective="distance_and_rotation",
                          position_threshold_m=bound, position_growth_per_step_m=.02,
                          yaw_growth_per_step_rad=0., drift_yaw_per_rad=0.,
                          landmark_position_obs_sigma_m=.005, min_axis_sigma_m=.001)
    # Contacts constrain both axes so the detour can reset uncertainty at
    # every step; the shortcut has a full step without an observation.
    for contacts in grid.contacts.values():
        for c in contacts:
            c["kind"] = "landmark"
    belief = PlannerBelief(minor=0, cov=np.eye(2)*.03**2, yaw_sigma=0.)
    plan = plan_grid(grid, belief, cfg, yaw0=0.)
    assert plan.feasible and plan.distance_m == pytest.approx(distance)
    rows = replay_grid(grid, plan.path, belief, cfg, yaw0=0.)
    assert all(r["max_position_sigma_m"] <= bound + 1e-9 for r in rows)
    assert ((1, 0) in plan.path) == (distance == 2.)
    assert (plan.free_distance_m > 0.) == (distance == 2.)
