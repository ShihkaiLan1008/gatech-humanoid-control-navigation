"""Regression: a scan of somewhere the tape never measured.

The demo room has an external datum -- a drawing with eleven tape dimensions --
so its scan is registered onto that frame.  A scan of anywhere else has no such
anchor, and the pipeline has to say so rather than force a fit.  These tests pin
the differences: no inherited curved features, no demo anchors, a frame derived
from the scan's own walls, and a generic traverse instead of the demo route.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from demo_map.geometry import HEIGHT, THICKNESS, clearance, in_rooms
from demo_map.scan_map import (build_scan_map, build_scan_traverse,
                               load_scan_walls)

WALLS = Path("docs/demo_maps/room2_scan_walls.json")

pytestmark = pytest.mark.skipif(not WALLS.exists(),
                                reason="room2 scan walls JSON not extracted")

room = build_scan_map(WALLS)
doc = load_scan_walls(WALLS)


def test_the_frame_is_declared_self_derived():
    assert doc["frame"] == "self"
    assert doc["registration"]["residual"].get("note"), \
        "a self-derived frame has nothing to residual against and must say so"


def test_the_scan_is_squared_to_its_own_walls():
    """The frame's rotation comes from the area-weighted dominant direction of
    the vertical surfaces, so the built walls must come out axis-aligned."""
    assert abs(doc["registration"]["theta_deg"]) < 45.0
    for w in room.walls:
        d = np.subtract(w.p1, w.p0)
        assert min(abs(d[0]), abs(d[1])) < 1e-9, w.wid


def test_the_origin_sits_on_the_corner_of_the_two_longest_walls():
    xs = [w for w in room.walls if abs(w.p0[0] - w.p1[0]) < 1e-9]
    ys = [w for w in room.walls if abs(w.p0[1] - w.p1[1]) < 1e-9]
    assert xs and ys
    assert abs(max(xs, key=lambda w: w.length).p0[0]) < 0.05
    assert abs(max(ys, key=lambda w: w.length).p0[1]) < 0.05


def test_no_tape_geometry_leaks_into_a_room_the_tape_never_saw():
    assert room.curves == [], "C1/C2 belong to the measured corridor only"
    assert room.openings == []
    assert room.key_points == {}, "no demo anchors exist in this room"
    # and the start is NOT the demo map's spawn
    assert np.linalg.norm(np.subtract(room.start.center, (-1.30, -0.65))) > 1.0


def test_the_room_is_built_and_physical():
    # the 8_11 scan captured an L of two walls (west + north) plus the west
    # wall's recess planes.  The default build flattens the recess planes
    # into their wall (the coherent-walls ruling, 2026-08-27), leaving the
    # L itself; the recess structure is still there for a caller that asks
    # for it, and the deploy scene does (flatten_steps=False)
    assert len(room.walls) >= 2
    assert len(build_scan_map(WALLS, flatten_steps=False).walls) >= 4
    assert len(room.boxes) >= 1
    for w in room.walls:
        assert w.thickness == THICKNESS and w.height == HEIGHT
        assert w.length >= 0.30
    for b in room.boxes:
        assert min(b.half) > 0.0 and 0.0 <= b.z0 < b.z1 <= 2.4


def test_mujoco_build_measures_back_to_the_spec():
    mujoco = pytest.importorskip("mujoco")
    from demo_map.mujoco_scene import demo_map_spec, measure_model

    model = demo_map_spec(room).compile()
    mujoco.mj_forward(model, mujoco.MjData(model))
    built = measure_model(model)
    assert set(built["walls"]) == {w.wid for w in room.walls}
    assert set(built["boxes"]) == {b.bid for b in room.boxes}
    for w in room.walls:
        g = built["walls"][w.wid]
        assert 2.0 * g["half_len"] == pytest.approx(w.length, abs=1e-7)
        assert 2.0 * g["half_height"] == pytest.approx(HEIGHT, abs=1e-9)


def test_it_reaches_the_injection_path_as_walls_and_cabinets():
    from demo_map.mapv3 import demo_map_obstacles

    from demo_map import mapv3

    obs = demo_map_obstacles(room)
    assert sum(o.kind == "wall" for o in obs) == len(room.walls)
    assert sum(o.kind == "pillar" for o in obs) == 0
    # the room's boxes are found by the scan but not built: a landmark is a
    # corner or a cylinder, and clutter is removed down to the wall behind it
    assert not mapv3.BOX_LANDMARKS
    assert room.boxes, "the scan should still be finding the clutter"
    assert sum(o.kind == "cabinet" for o in obs) == 0
    assert {o.half_height for o in obs} == {0.5 * HEIGHT}


def test_the_generic_traverse_crosses_the_room_and_is_clear():
    r = build_scan_traverse(room)
    assert r is not None, "no collision-free path across the scanned free space"
    assert r.length > 2.0, "a traverse that goes nowhere is not a demo"
    for a, b in zip(r.segs[:-1], r.segs[1:]):
        assert np.allclose(a.p1, b.p0, atol=1e-9)
    pts = r.polyline(0.02)
    assert float(clearance(pts, room).min()) > 0.40
    assert in_rooms(pts, room).all()
    assert not any(s.hand for s in r.segs), "nothing is wall-followed here"


def test_start_and_goal_are_spawnable_and_far_apart():
    assert float(clearance([room.start.center], room)[0]) >= 0.60
    assert float(clearance([room.goal.center], room)[0]) >= 0.60
    assert in_rooms([room.start.center], room)[0]
    assert in_rooms([room.goal.center], room)[0]
    assert np.linalg.norm(np.subtract(room.goal.center, room.start.center)) > 2.0


def test_the_demo_map_is_unaffected_by_any_of_this():
    from demo_map import build_demo_map
    from demo_map.scan_map import SCAN_WALLS

    tape = build_demo_map()
    assert len(tape.walls) == 12 and len(tape.curves) == 2
    if SCAN_WALLS.exists():
        demo_scan = build_scan_map()
        assert len(demo_scan.curves) == 2, "the demo scan keeps its tape circles"
        assert demo_scan.key_points, "the demo scan keeps its anchors"
