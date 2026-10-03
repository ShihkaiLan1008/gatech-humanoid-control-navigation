"""Regression: the reconstructed room still matches the drawing it came from.

The point of these tests is that they measure the BUILT artefacts -- the wall
slabs, the MuJoCo geoms, the route -- and check the labelled dimensions come
back out.  Restating the input constants would prove nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from demo_map import build_demo_map, build_reference_route
from demo_map.export import SPEC_PATH, spec_dict
from demo_map.geometry import (D_BOTTOM_TO_UPPER, D_C1_CHORD, D_C1_PROUD,
                               D_C2_CHORD, D_C2_PROUD, D_CORNER_TO_DOOR,
                               D_CORRIDOR_W, D_DOOR_W, D_LOWER_W,
                               D_RUN_C1_C2, D_RUN_C2_CORNER, HEIGHT, LANE_OFF,
                               THICKNESS, clearance, in_rooms, remeasure)

TOL = 1e-9

m = build_demo_map()
route = build_reference_route(m)


# ── the labelled dimensions ───────────────────────────────────────────────────


@pytest.mark.parametrize("row", remeasure(m), ids=lambda r: r["feature"])
def test_labelled_dimension_recovered(row):
    """Every blue number on measured_map.jpg, measured back off the geometry."""
    assert abs(row["labelled"] - row["recon"]) < TOL, row


def test_dimension_chain_closes_on_the_corner():
    chain = D_C1_CHORD + D_RUN_C1_C2 + D_C2_CHORD + D_RUN_C2_CORNER
    assert m.key_points["x_corner"] == pytest.approx(chain, abs=TOL)
    assert m.key_points["x_corner"] == pytest.approx(10.3505, abs=TOL)


def test_key_wall_coordinates():
    """Literal coordinates, so an accidental edit to the chain is caught."""
    kp = m.key_points
    for name, want in (
        ("x_c1_w", 0.0), ("x_c1_e", 0.5969), ("x_c2_w", 7.3533),
        ("x_c2_e", 8.0899), ("x_corner", 10.3505), ("x_corr_e", 12.7635),
        ("y_bottom", -1.6256), ("y_upper_s", 8.9154),
        ("x_door_w", 14.2265), ("x_door_e", 15.8521),
    ):
        assert kp[name] == pytest.approx(want, abs=TOL), name


def test_corridor_and_doorway_are_clear_spans_between_faces():
    kp = m.key_points
    assert kp["x_corr_e"] - kp["x_corner"] == pytest.approx(D_CORRIDOR_W, abs=TOL)
    assert 0.0 - kp["y_bottom"] == pytest.approx(D_LOWER_W, abs=TOL)
    assert kp["y_upper_s"] - kp["y_bottom"] == pytest.approx(D_BOTTOM_TO_UPPER, abs=TOL)
    assert m.wall("W8").length == pytest.approx(D_CORNER_TO_DOOR, abs=TOL)
    assert m.opening("DOOR").width == pytest.approx(D_DOOR_W, abs=TOL)


# ── the two curved features ───────────────────────────────────────────────────


@pytest.mark.parametrize("cid,chord,proud", [("C1", D_C1_CHORD, D_C1_PROUD),
                                             ("C2", D_C2_CHORD, D_C2_PROUD)])
def test_curve_circle_cuts_the_wall_at_its_chord(cid, chord, proud):
    """The reconstructed circle must meet the wall face exactly at the labelled
    opening AND reach exactly the labelled depth -- both numbers, one circle."""
    c = m.curve(cid)
    d = abs(c.center[1] - 0.0)
    assert 2.0 * math.sqrt(c.radius ** 2 - d ** 2) == pytest.approx(chord, abs=TOL)
    assert -(c.center[1] - c.radius) == pytest.approx(proud, abs=TOL)
    assert c.chord0[1] == 0.0 and c.chord1[1] == 0.0


def test_c2_is_a_major_arc_and_c1_is_not():
    """C2's depth exceeds half its opening, so its centre sits INSIDE the
    corridor: the feature is wider below the wall face than at it."""
    assert m.curve("C1").center[1] > 0.0
    assert m.curve("C2").center[1] < 0.0
    assert D_C2_PROUD > D_C2_CHORD / 2.0
    assert D_C1_PROUD < D_C1_CHORD / 2.0


def test_curves_sit_in_the_gaps_left_in_the_long_wall():
    """The wall run stops at each chord end, so the built boundary is
    continuous: wall, arc, wall."""
    assert m.wall("W1").p1[0] == pytest.approx(m.curve("C1").chord0[0], abs=TOL)
    assert m.wall("W2").p0[0] == pytest.approx(m.curve("C1").chord1[0], abs=TOL)
    assert m.wall("W2").p1[0] == pytest.approx(m.curve("C2").chord0[0], abs=TOL)
    assert m.wall("W3").p0[0] == pytest.approx(m.curve("C2").chord1[0], abs=TOL)


# ── slabs, thickness, height ──────────────────────────────────────────────────


def test_every_primitive_carries_the_declared_thickness_and_height():
    for w in m.walls:
        assert w.thickness == THICKNESS
        assert w.height == HEIGHT == 1.8
    for c in m.curves:
        assert c.height == HEIGHT


def test_centreline_is_half_a_thickness_behind_the_face():
    """Walls are extruded AWAY from the free space, so the measured face stays
    where the tape put it."""
    for w in m.walls:
        a, b = (np.asarray(p, float) for p in w.centreline)
        n = np.asarray(w.inward, float)
        assert np.allclose(a, np.asarray(w.p0) - 0.5 * THICKNESS * n, atol=TOL)
        assert np.allclose(b, np.asarray(w.p1) - 0.5 * THICKNESS * n, atol=TOL)
        # step off the middle of the face (not a corner, where an abutting
        # wall's own face plane would meet the probe) and land in free space
        mid = 0.5 * (np.asarray(w.p0, float) + np.asarray(w.p1, float))
        assert clearance([mid + 0.30 * n], m)[0] > 0.20, w.wid


# ── the MuJoCo build ──────────────────────────────────────────────────────────


def test_mujoco_model_compiles_and_measures_back_to_the_spec():
    mujoco = pytest.importorskip("mujoco")
    from demo_map.mujoco_scene import demo_map_spec, measure_model

    model = demo_map_spec(m).compile()
    mujoco.mj_forward(model, mujoco.MjData(model))
    built = measure_model(model)

    assert set(built["walls"]) == {w.wid for w in m.walls}
    assert set(built["curves"]) == {c.cid for c in m.curves}

    for w in m.walls:
        g = built["walls"][w.wid]
        assert 2.0 * g["half_height"] == pytest.approx(HEIGHT, abs=TOL)
        assert 2.0 * g["half_thick"] == pytest.approx(THICKNESS, abs=TOL)
        assert 2.0 * g["half_len"] == pytest.approx(w.length, abs=1e-7)
        # push the slab's centreline back out to the room-side face
        sgn = np.sign(np.dot(g["n"], np.asarray(w.inward, float)))
        face = [g["p0"] + sgn * g["half_thick"] * g["n"],
                g["p1"] + sgn * g["half_thick"] * g["n"]]
        got = sorted(face, key=lambda p: (round(p[0], 6), round(p[1], 6)))
        want = sorted([np.asarray(w.p0, float), np.asarray(w.p1, float)],
                      key=lambda p: (round(p[0], 6), round(p[1], 6)))
        assert np.allclose(got, want, atol=1e-7), w.wid

    for c in m.curves:
        g = built["curves"][c.cid]
        assert g["radius"] == pytest.approx(c.radius, abs=1e-9)
        assert np.allclose(g["center"], np.asarray(c.center), atol=1e-9)
        assert 2.0 * g["half_height"] == pytest.approx(HEIGHT, abs=TOL)


def test_mapv3_adapter_matches_the_metric_spec():
    """The project's MapV3/mj_assets path must build the same room."""
    from demo_map.mapv3 import demo_map_obstacles, demo_map_v3
    from lane_graph.maps_v3 import THICK

    v3 = demo_map_v3(m)
    assert THICK == THICKNESS
    assert len(v3.walls) == len(m.walls)
    for w, vw in zip(m.walls, v3.walls):
        assert np.allclose(vw.p0, w.centreline[0], atol=TOL)
        assert np.allclose(vw.p1, w.centreline[1], atol=TOL)
    for c, b in zip(m.curves, v3.bulges):
        assert b.radius == pytest.approx(c.radius, abs=TOL)
        assert np.allclose(b.center, c.center, atol=TOL)

    obs = demo_map_obstacles(m)
    assert len(obs) == len(m.walls) + len(m.curves)
    assert {o.half_height for o in obs} == {0.5 * HEIGHT}
    for o, w in zip([o for o in obs if o.kind == "wall"], m.walls):
        # +0.10 total: each wall is extended half the 0.10 thickness at
        # both ends so corners weld instead of showing a wedge of daylight
        assert 2.0 * o.size[0] == pytest.approx(w.length + 0.10, abs=1e-9)


def test_v3_validate_passes():
    from demo_map.mapv3 import demo_map_v3
    from lane_graph.maps_v3 import validate

    assert validate(demo_map_v3(m)) == []


# ── the reference route ───────────────────────────────────────────────────────


def test_route_is_continuous_and_inside_the_free_space():
    for a, b in zip(route.segs[:-1], route.segs[1:]):
        assert np.allclose(a.p1, b.p0, atol=TOL), (a.sid, b.sid)
    pts = route.polyline(0.02)
    assert in_rooms(pts, m).all()


def test_route_clears_every_surface():
    """G1 body clearance: the tightest point on the route, against the same
    slab-distance convention lane_graph.maps_v3 uses."""
    assert float(clearance(route.polyline(0.01), m).min()) > 0.45


def test_followed_segments_hold_the_lane_offset():
    """A followed lane is the offset curve of the wall envelope, so the gap to
    the followed surface is constant -- including around the curved features,
    which is the whole reason the lane bows."""
    followed = [s for s in route.segs
                if s.kind in ("wall_follow", "curve_follow", "corner_turn")]
    assert [s.sid for s in followed] == ["R1", "R2", "R3", "R4", "R6", "R7", "R8"]
    for s in followed:
        d = clearance(s.sample(0.01), m)
        assert d.max() <= LANE_OFF + 1e-6, s.sid
        # the shortfall is the slab-end rounding where a perpendicular wall
        # abuts, never the followed face itself
        assert d.min() > LANE_OFF - 0.06, s.sid


def test_route_bows_around_the_deep_curved_feature():
    """A straight 0.56 m lane would pass 0.09 m from C2's apex -- a collision.
    The bow has to be there."""
    c2 = m.curve("C2")
    assert LANE_OFF - c2.proud < 0.10
    apex_lane = min(p[1] for p in route.segs[3].sample(0.002))
    # abs is the sampling chord error, not a geometry tolerance
    assert apex_lane == pytest.approx(c2.center[1] - c2.radius - LANE_OFF, abs=1e-5)


def test_route_topology_matches_the_desired_sketch():
    kinds = [s.kind for s in route.segs]
    assert kinds == ["approach", "curve_follow", "wall_follow", "curve_follow",
                     "wall_follow", "transit", "wall_follow", "corner_turn",
                     "wall_follow", "goal_approach"]
    assert [s.follows for s in route.segs[:5]] == ["W1", "C1", "W2", "C2", "W3"]
    # left hand on the long wall, right hand after the corridor crossing
    assert {s.hand for s in route.segs[:5]} == {"left"}
    assert {s.hand for s in route.segs[6:9]} == {"right"}
    assert route.segs[5].kind == "transit" and route.segs[5].hand == ""


def test_route_starts_in_start_region_and_ends_in_goal_region():
    assert np.linalg.norm(np.subtract(route.segs[0].p0, m.start.center)) <= m.start.radius
    assert np.linalg.norm(np.subtract(route.segs[-1].p1, m.goal.center)) <= m.goal.radius
    assert m.goal.center[0] == pytest.approx(m.opening("DOOR").mid[0], abs=TOL)


def test_start_and_goal_have_spawn_clearance():
    assert float(clearance([m.start.center], m)[0]) >= 0.60
    assert float(clearance([m.goal.center], m)[0]) >= 0.60


# ── the spec file on disk ─────────────────────────────────────────────────────


def test_yaml_spec_is_in_sync_with_the_derivation():
    """docs/demo_maps/demo_map_geometry.yaml is generated; regenerate it with
    `PYTHONPATH=src python3 scripts/build_demo_map.py` if this fails."""
    yaml = pytest.importorskip("yaml")
    from demo_map.export import dump_yaml

    assert SPEC_PATH.exists(), f"{SPEC_PATH} missing"
    assert yaml.safe_load(SPEC_PATH.read_text()) == yaml.safe_load(dump_yaml(m, route))
