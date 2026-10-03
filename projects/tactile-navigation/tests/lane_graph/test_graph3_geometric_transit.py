"""The geometric free-space layer, wired into the V3 route graph.

`PF_AUDIT.md` 6 audited `build_graph3` and found that lanes were never the gap
-- they are derived from the wall list already -- but free-space travel was:
waypoints on a 1.15 m grid inside hand-declared room rectangles, joined by
straight hops capped at 2.6 m, with no smoothing.  `planning.geometric` was
written and tested standalone but not wired in.

Two properties have to hold together for the wiring to be worth landing:

* it must be ADDITIVE -- every straight connector the shipped maps already rely
  on survives untouched, and the route those maps plan does not move.  The R2
  and R3 physical demos were produced by this planner; silently re-routing them
  would invalidate them.
* it must EARN ITS KEEP -- a passage the hop grid cannot represent is routed.
"""

from __future__ import annotations

import numpy as np
import pytest

from lane_graph.graph3 import (GEOM_DETOUR_MAX, Belief3, astar3, build_graph3,
                               transit_grid)
from lane_graph.maps_v3 import LANE_CLEAR, MAPS3, MapV3, Wall, clearance, in_rooms
from lane_graph.scene3 import build_v3_scene

SHIPPED = list(MAPS3)


# ── additive: the shipped maps do not move ──────────────────────────────────────

@pytest.mark.parametrize("name", SHIPPED)
def test_every_straight_connector_survives(name):
    sc = build_v3_scene(name)
    old = build_graph3(sc, geometric_transit=False)
    new = build_graph3(sc)

    def straight(g):
        return sorted((e.a, e.b) for e in g.edges
                      if e.kind == "FREE_TRANSIT" and e.poly is None)

    assert straight(new) == straight(old)
    assert [(e.a, e.b) for e in new.edges if e.kind == "TOUCH_FOLLOW"] == \
           [(e.a, e.b) for e in old.edges if e.kind == "TOUCH_FOLLOW"]


@pytest.mark.parametrize("name", SHIPPED)
def test_the_planned_route_is_unchanged_on_shipped_maps(name):
    """The demo-preserving property, stated as the planner's own answer."""
    sc = build_v3_scene(name)
    b = Belief3(sigma=0.10, minor=0.0)
    p_old, c_old = astar3(build_graph3(sc, geometric_transit=False), 0, 1, b)
    g_new = build_graph3(sc)
    p_new, c_new = astar3(g_new, 0, 1, b)
    assert p_old is not None and p_new is not None
    assert c_new == pytest.approx(c_old)
    assert [g_new.edges[i].kind for i in p_new] == \
           [build_graph3(sc, geometric_transit=False).edges[i].kind for i in p_old]


# ── the new edges are legal ─────────────────────────────────────────────────────

@pytest.mark.parametrize("name", SHIPPED)
def test_bent_connectors_are_clear_and_bounded(name):
    sc = build_v3_scene(name)
    spec = sc.spec
    g = build_graph3(sc)
    bent = [e for e in g.edges if e.kind == "FREE_TRANSIT" and e.poly is not None]
    for e in bent:
        pts = np.asarray(e.poly, float)
        # endpoints are the nodes, in some order
        ends = {tuple(np.round(pts[0], 6)), tuple(np.round(pts[-1], 6))}
        nodes = {tuple(np.round(np.asarray(g.nodes[e.a].xy), 6)),
                 tuple(np.round(np.asarray(g.nodes[e.b].xy), 6))}
        assert ends == nodes
        # every sample clears the same band the straight-hop test demanded
        assert float(np.min(clearance(pts, spec))) >= LANE_CLEAR - 1e-9
        assert bool(np.all(in_rooms(pts, spec)))
        # a connector, not a travel mode
        straight = float(np.linalg.norm(pts[0] - pts[-1]))
        assert e.length >= straight - 1e-9
        assert e.length <= GEOM_DETOUR_MAX * straight + 1e-9


@pytest.mark.parametrize("name", SHIPPED)
def test_the_transit_grid_never_calls_outside_a_room_free(name):
    spec = build_v3_scene(name).spec
    grid = transit_grid(spec)
    pts = np.array([grid.to_world(i, j)
                    for i in range(grid.nx) for j in range(grid.ny)])
    free = np.array([grid.free[i, j]
                     for i in range(grid.nx) for j in range(grid.ny)])
    assert not bool(np.any(free & ~in_rooms(pts, spec)))


# ── earns its keep: a passage the 1.15 m hop grid cannot represent ──────────────

def _pinch(gap_m: float, wall_x: float = 3.0) -> MapV3:
    """One room, one dividing wall, one centred doorway of a chosen width.

    ``gap_m`` is between wall CENTRELINES, so the free width the base sees is
    ``gap_m - THICK`` (0.10 m).  The robot needs ``2 * LANE_CLEAR`` = 0.56 m of
    it, i.e. a 0.66 m gap.  Measured on this map, the two layers disagree over
    a wide band:

    | free width | hop chain | geometric |
    |---|---|---|
    | 0.50 m | refused | refused  (correctly: the base does not fit) |
    | 0.56 m | refused | ROUTED   (exactly the clearance contract) |
    | 0.70 m | refused | ROUTED |
    | 0.80 m | routed  | routed |

    So the hop chain needs 0.80 m of free width to admit a door the robot can
    walk through at 0.56 m -- it is 43% more conservative than the contract,
    not because the geometry is tight but because a 1.15 m waypoint pitch has
    to get lucky and land a straight segment inside the gap.
    """
    y0, y1 = 0.0, 4.0
    h = gap_m / 2.0
    walls = [Wall((0, y0), (6, y0)), Wall((0, y1), (6, y1)),
             Wall((0, y0), (0, y1)), Wall((6, y0), (6, y1)),
             Wall((wall_x, y0), (wall_x, 2.0 - h)),
             Wall((wall_x, 2.0 + h), (wall_x, y1))]
    return MapV3(name="pinch", title="pinch", walls=walls, bulges=[], prots=[],
                 rooms=[(0, y0, 6, y1)], start=(0.8, 2.0), goal=(5.2, 2.0),
                 goal_r=0.25, doors=[], landmarks=[],
                 topology="one door", story="")


class _Scene:
    """The two fields build_graph3 reads off a scene, without a V3Scene build."""

    def __init__(self, spec: MapV3):
        self.spec = spec
        self.lanes = []
        self.start_map_xy = np.asarray(spec.start, float)
        self.goal_map_xy = np.asarray(spec.goal, float)


def _connected(g) -> bool:
    seen, stack = {0}, [0]
    while stack:
        n = stack.pop()
        if n == 1:
            return True
        for eid in g.adj[n]:
            e = g.edges[eid]
            m = e.b if e.a == n else e.a
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return 1 in seen


def test_a_doorway_the_base_fits_is_unroutable_without_the_layer():
    """The defect, reproduced: 0.70 m of free width, the base needs 0.56 m."""
    g = build_graph3(_Scene(_pinch(0.80)), geometric_transit=False)
    assert not _connected(g)


def test_the_geometric_layer_routes_that_doorway():
    g = build_graph3(_Scene(_pinch(0.80)))
    assert _connected(g)
    bent = [e for e in g.edges if e.kind == "FREE_TRANSIT" and e.poly is not None]
    assert bent, "the connection must come from a bent connector"


def test_a_doorway_too_narrow_is_still_refused():
    """0.60 m between centrelines is 0.50 m of free width after the 0.10 m wall
    thickness -- under 2 x LANE_CLEAR.  Refusing is the correct answer, and the
    new layer must not be the thing that starts accepting it."""
    g = build_graph3(_Scene(_pinch(0.60)))
    assert not _connected(g)


def test_the_layer_admits_exactly_the_clearance_contract():
    """The accept/refuse boundary sits at 2 x LANE_CLEAR of free width, not
    wherever a waypoint pitch happens to land a straight segment."""
    need = 2 * LANE_CLEAR + 0.10          # + wall thickness, centreline to centreline
    assert not _connected(build_graph3(_Scene(_pinch(need - 0.06))))
    assert _connected(build_graph3(_Scene(_pinch(need))))
