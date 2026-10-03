"""Regression: the scan-derived map, and what it says about the tape one.

The interesting assertions here are the AGREEMENTS -- an iPhone mesh and a tape
measure are independent instruments, so where they agree the reconstruction is
corroborated, and where they disagree one of them is wrong in a way worth
pinning down.  The 1.6 m disagreement on the drawing's 1054.1 cm datum is
asserted deliberately: if it ever vanishes, the data changed and the report
needs rewriting.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from demo_map import build_demo_map
from demo_map.geometry import (D_BOTTOM_TO_UPPER, D_CORRIDOR_W, D_LOWER_W,
                               HEIGHT, LANE_OFF, THICKNESS, clearance, in_rooms)
from demo_map.scan_map import SCAN_WALLS, build_scan_map, build_scan_route

pytestmark = pytest.mark.skipif(not SCAN_WALLS.exists(),
                                reason="scan walls JSON not extracted")

scan = build_scan_map()
tape = build_demo_map()
route = build_scan_route(scan)
A = scan.key_points


# ── registration ──────────────────────────────────────────────────────────────


def test_registration_is_a_quarter_turn_with_a_small_residual():
    from demo_map.scan_map import load_scan_walls

    reg = load_scan_walls()["registration"]
    assert abs(reg["theta_deg"] - 90.0) < 1.0, "the scan should sit square to the map"
    assert reg["residual"]["median_mm"] < 60.0
    assert reg["residual"]["p90_mm"] < 200.0
    assert reg["residual"]["n"] > 1000


# ── the scan corroborates the tape where the tape actually measured ──────────


@pytest.mark.parametrize("name,tape_val,scan_val,tol_mm", [
    # 90, not 80: the coherent-walls flatten (2026-08-27) puts the merged
    # south wall at the length-weighted depth of its boards, 2.5 cm behind
    # the front plane the anchor used to quote -- the tape's single number
    # lands between the planes, same story as the branch width below
    ("lower corridor clear width", D_LOWER_W,
     A["long_wall_y"] - A["bottom_y"], 90.0),
    # 110, not 80: the scan resolves TWO planes on the corridor's east wall
    # (a ~6 cm step the extractor now keeps instead of averaging away), and
    # the anchor deliberately takes the front -- touchable -- plane.  The
    # tape's single number lands between the planes; the old 2.395 m
    # "agreement" was the average of both, i.e. a surface that exists nowhere.
    ("branch corridor clear width", D_CORRIDOR_W,
     A["branch_e_x"] - A["branch_w_x"], 110.0),
    ("long wall origin -> branch mouth", 10.3505, A["branch_w_x"], 80.0),
])
def test_scan_agrees_with_the_tape(name, tape_val, scan_val, tol_mm):
    assert abs(scan_val - tape_val) * 1000.0 < tol_mm, name


def test_the_two_curved_features_sit_where_the_tape_put_them():
    """They are kept from the tape, so this asserts the scan's own wall runs
    break around them -- the independent confirmation."""
    for c in scan.curves:
        t = tape.curve(c.cid)
        assert c.radius == pytest.approx(t.radius, abs=1e-12)
        assert np.allclose(c.center, t.center, atol=1e-12)
    # ...and the wall RUNS THROUGH the chord, with the feature standing proud
    # of it.  The scan resolves no surface across a chord -- the phone saw the
    # bulge, not the wall behind it -- but `_join_runs` bridges the gap
    # (2026-08-31), because a 0.29 m sagitta on a 0.2985 m radius is a
    # half-cylinder on a wall, not a hole in one.
    long_runs = [(w.p0[0], w.p1[0]) for w in scan.walls
                 if abs(w.p0[1] - A["long_wall_y"]) < 0.12
                 and abs(w.p0[1] - w.p1[1]) < 1e-9]
    for c in scan.curves:
        mid = 0.5 * (c.chord0[0] + c.chord1[0])
        assert any(a < mid < b for a, b in long_runs), \
            f"{c.cid}: the wall must run continuously behind the feature"
        assert c.proud >= c.radius - 0.02, \
            f"{c.cid}: at least half of it stands out of the wall behind it"


def test_the_1054_datum_is_the_one_that_disagrees():
    """The drawing's bottom-wall -> upper-room datum is 1.6 m short of where the
    scan finds the branch corridor's end.  Everything else agrees to <= 80 mm,
    so this is a real inconsistency in the drawing, not scan error."""
    scan_span = A["branch_end_y"] - A["bottom_y"]
    assert scan_span - D_BOTTOM_TO_UPPER > 1.0
    assert scan_span - D_BOTTOM_TO_UPPER < 2.0


# ── the built map ─────────────────────────────────────────────────────────────


def test_scan_map_primitives_are_well_formed():
    from demo_map.geometry import DemoMap, Region

    # twelve boards, not the scan's fifty-one faces: one plane per wall
    # after the flatten, the run join and the contained-board drop
    assert 10 <= len(scan.walls) <= 16
    for w in scan.walls:
        assert w.thickness == THICKNESS
        assert w.height == HEIGHT == 1.8
        # 0.30, not 0.45: doorway jamb stubs are real walls and the robot
        # touches them, so the builder keeps faces down to 0.30 m now
        assert w.length >= 0.30
        assert abs(float(np.linalg.norm(w.inward)) - 1.0) < 1e-9


def test_every_buried_wall_face_is_buried_by_its_own_wall_system():
    """A wall face the robot cannot reach must be unreachable for a REASON:
    a nearer parallel plane of the same wall system in front of it (the
    extractor keeps steps, recesses and bowed walls as separate boards on
    purpose -- the deeper board documents where the surface really is).

    What stays forbidden is ACCIDENTAL burial: a face shadowed by a
    perpendicular slab, or by a parallel board on the far side of a face
    that claims to front free space -- those meant the extraction was wrong.
    Same-plane duplication is the stacked-boards test's job."""
    from demo_map.geometry import DemoMap, Region

    def covered_by_front(w, pts):
        horiz = abs(w.p0[1] - w.p1[1]) < 1e-9
        perp = 1 if horiz else 0
        along = 1 - perp
        pw = w.p0[perp]
        inw = w.inward[perp]
        ok = np.zeros(len(pts), bool)
        for v in scan.walls:
            if v.wid == w.wid:
                continue
            vh = abs(v.p0[1] - v.p1[1]) < 1e-9
            if vh != horiz:
                continue
            pv = v.p0[perp]
            if not (0.0 < (pv - pw) * inw <= 0.15):
                continue                     # not a nearer parallel plane
            v0 = min(v.p0[along], v.p1[along])
            v1 = max(v.p0[along], v.p1[along])
            ok |= (pts[:, along] >= v0 - 0.06) & (pts[:, along] <= v1 + 0.06)
        return ok

    for w in scan.walls:
        others = DemoMap(walls=[v for v in scan.walls if v.wid != w.wid],
                         curves=scan.curves, openings=[], rooms=list(scan.rooms),
                         start=Region("s", (0, 0), 0.1, 0.0),
                         goal=Region("g", (0, 0), 0.1, 0.0))
        t = np.linspace(0.0, 1.0, 41)[:, None]
        pts = (1 - t) * np.asarray(w.p0, float) + t * np.asarray(w.p1, float)
        buried = clearance(pts, others) <= -1e-9
        # corner joins bury up to a slab thickness at each END by
        # construction (a perpendicular wall's slab crosses the endpoint,
        # and ``_flatten_boards`` may shift a neighbour's join line a few
        # cm along the stub) -- exclude the end zones explicitly instead
        # of leaning on the percentage slack
        d_end = np.minimum(t, 1.0 - t)[:, 0] * w.length
        unexplained = (buried & ~covered_by_front(w, pts)
                       & (d_end > THICKNESS + 0.01))
        frac = float(unexplained.mean())
        assert frac <= 0.30, \
            f"{w.wid}: {frac:.0%} of its face is buried by something that is " \
            f"not a nearer plane of its own wall"


def test_scan_map_is_axis_aligned_in_the_map_frame():
    for w in scan.walls:
        d = np.subtract(w.p1, w.p0)
        assert min(abs(d[0]), abs(d[1])) < 1e-9, w.wid


def test_scan_mujoco_build_measures_back():
    mujoco = pytest.importorskip("mujoco")
    from demo_map.mujoco_scene import demo_map_spec, measure_model

    model = demo_map_spec(scan).compile()
    mujoco.mj_forward(model, mujoco.MjData(model))
    built = measure_model(model)
    assert set(built["walls"]) == {w.wid for w in scan.walls}
    for w in scan.walls:
        g = built["walls"][w.wid]
        assert 2.0 * g["half_height"] == pytest.approx(HEIGHT, abs=1e-9)
        assert 2.0 * g["half_len"] == pytest.approx(w.length, abs=1e-7)
        sgn = np.sign(np.dot(g["n"], np.asarray(w.inward, float)))
        face = sorted([g["p0"] + sgn * g["half_thick"] * g["n"],
                       g["p1"] + sgn * g["half_thick"] * g["n"]],
                      key=lambda p: (round(p[0], 6), round(p[1], 6)))
        want = sorted([np.asarray(w.p0, float), np.asarray(w.p1, float)],
                      key=lambda p: (round(p[0], 6), round(p[1], 6)))
        assert np.allclose(face, want, atol=1e-7), w.wid


def test_scan_map_passes_the_v3_validator():
    from demo_map.mapv3 import demo_map_v3
    from lane_graph.maps_v3 import validate

    assert validate(demo_map_v3(scan)) == []


def test_scan_obstacles_carry_the_demo_height():
    from demo_map.mapv3 import demo_map_obstacles

    obs = demo_map_obstacles(scan)
    kinds = {k: sum(o.kind == k for o in obs) for k in ("wall", "pillar", "cabinet")}
    assert kinds["wall"] == len(scan.walls)
    # scanned round pillars ride the same channel as the curved features, and
    # shaped landmarks fall back to their min-area rectangle on this path
    assert kinds["pillar"] == len(scan.curves) + len(scan.pillars)
    # CLUTTER NO LONGER REACHES THE SCENE.  A landmark is a corner or a
    # cylinder by owner directive; boxes and hulls are removed so the hand
    # meets the flat wall behind them.  The scan still FINDS them -- that is
    # what `scan.boxes`/`scan.hulls` are -- and `BOX_LANDMARKS` decides
    # whether they get built, so both sides are asserted here rather than
    # deleting the one that changed.
    from demo_map import mapv3

    assert not mapv3.BOX_LANDMARKS
    assert kinds["cabinet"] == 0, "clutter must not reach the injector"
    assert scan.boxes or scan.hulls, "the scan should still be finding it"

    was = mapv3.BOX_LANDMARKS
    mapv3.BOX_LANDMARKS = True
    try:
        on = demo_map_obstacles(scan)
    finally:
        mapv3.BOX_LANDMARKS = was
    assert sum(o.kind == "cabinet" for o in on) == \
        len(scan.boxes) + len(scan.hulls)

    assert len(obs) == sum(kinds.values())
    assert {o.half_height for o in obs} == {0.5 * HEIGHT}


# ── the two defects the first build had ──────────────────────────────────────


def test_no_two_boards_are_stacked_on_the_same_wall_line():
    """The visible defect: one physical SURFACE extracted as several collinear
    lines a couple of cm apart, each built as its own board, overlapping.

    Same PLANE means within scan surface thickness (~2.5 cm).  Boards
    2.5-10 cm apart that overlap are NOT the defect any more -- they are a
    step, a recess or a proud strip, exactly the structure a tactile robot
    navigates by, and the builder now keeps them apart on purpose.  Also
    excluded: perpendicular slabs meeting at a corner (that is what a corner
    IS), and a feature standing proud of a wall, which must intersect the
    wall it grows from."""
    SAME_PLANE = 0.025

    def line(w):
        vertical = abs(w.p0[0] - w.p1[0]) < 1e-9
        pos = w.p0[0] if vertical else w.p0[1]
        s = (w.p0[1], w.p1[1]) if vertical else (w.p0[0], w.p1[0])
        return vertical, pos, min(s), max(s)

    bad = []
    for i, w in enumerate(scan.walls):
        for v in scan.walls[i + 1:]:
            if float(np.dot(w.inward, v.inward)) <= 0.5:
                continue
            aw, pw, w0, w1 = line(w)
            av, pv, v0, v1 = line(v)
            if aw != av or abs(pw - pv) > SAME_PLANE:
                continue                      # a step, not the same surface
            if min(w1, v1) - max(w0, v0) > 1e-6:
                bad.append((w.wid, v.wid, round(abs(pw - pv), 4)))
    assert not bad, f"boards stacked on one wall line: {bad}"


def test_every_long_extracted_face_survives_into_the_map():
    """The bug this catches cost the lower corridor its entire south wall: a
    0.9 m jut 0.17 m off a 12.7 m wall overlapped more than half of the STUB,
    and the duplicate rule let the stub veto the wall.  Nothing checked that
    the map still contained what the extractor found, because every safety
    check gets *easier* when a wall disappears."""
    from demo_map.scan_map import load_scan_walls

    def line(w):
        vertical = abs(w.p0[0] - w.p1[0]) < 1e-9
        pos = w.p0[0] if vertical else w.p0[1]
        s = (w.p0[1], w.p1[1]) if vertical else (w.p0[0], w.p1[0])
        return ("x" if vertical else "y"), pos, min(s), max(s)

    built = [line(w) + (tuple(w.inward),) for w in scan.walls]
    missing = []
    for f in load_scan_walls()["faces"]:
        if f["length"] < 1.0:
            continue
        if not any(ax == f["axis"] and inw == tuple(f["inward"])
                   and abs(pos - f["pos"]) <= 0.30
                   and (min(s1, f["s1"]) - max(s0, f["s0"])) > 0.5 * f["length"]
                   for ax, pos, s0, s1, inw in built):
            missing.append((f["axis"], round(f["pos"], 3), round(f["s0"], 2),
                            round(f["s1"], 2), round(f["length"], 2)))
    assert not missing, f"extracted faces >= 1 m absent from the map: {missing}"


def test_the_long_wall_is_one_run_with_the_features_proud_of_it():
    """The hallway wall is ONE surface (owner, 2026-08-31).

    The scan extracted it as five boards spread over 0.128 m in depth, broken
    by the two curve chords (0.74 m and 0.84 m of missing surface) and by the
    panel joints at its west end.  None of those is an opening: the openings
    this room has are the branch mouth and the corner into the end room, both
    2.39 m.  What must NOT be bridged is exactly those -- so the run stops at
    the branch mouth and starts again beyond it."""
    runs = sorted((min(w.p0[0], w.p1[0]), max(w.p0[0], w.p1[0]))
                  for w in scan.walls
                  if abs(w.p0[1] - w.p1[1]) < 1e-9
                  and abs(w.p0[1] - A["long_wall_y"]) < 0.20 and w.inward[1] < 0)
    west = [r for r in runs if r[0] < A["branch_w_x"]]
    assert len(west) == 1, f"the wall west of the branch is one run: {runs}"
    a, b = west[0]
    for c in scan.curves:
        mid = 0.5 * (c.chord0[0] + c.chord1[0])
        assert a < mid < b, f"{c.cid}: the run must carry past the chord"
    assert b == pytest.approx(A["branch_w_x"], abs=0.02), \
        "and it must STOP at the branch mouth, which is a real opening"


def test_clutter_boxes_are_physical():
    assert len(scan.boxes) > 5
    for b in scan.boxes:
        assert min(b.half) > 0.0 and max(b.half) < 4.0, b.bid
        assert 0.0 <= b.z0 < b.z1 <= 2.4, b.bid
        assert b.height > 0.05, b.bid


def test_clutter_reaches_the_mujoco_build():
    mujoco = pytest.importorskip("mujoco")
    from demo_map.mujoco_scene import demo_map_spec, measure_model

    built = measure_model(demo_map_spec(scan).compile())
    assert set(built["boxes"]) == {b.bid for b in scan.boxes}
    for b in scan.boxes:
        g = built["boxes"][b.bid]
        assert np.allclose(g["center"], b.center, atol=1e-9), b.bid
        assert np.allclose(g["half"], b.half, atol=1e-9), b.bid
        assert 2.0 * g["half_height"] == pytest.approx(b.height, abs=1e-9)


def test_most_of_the_scanned_wall_surface_is_represented():
    from demo_map.scan_map import load_scan_walls

    cov = load_scan_walls().get("coverage", {})
    assert cov.get("claimed_by_walls", 0.0) > 0.80, cov


# ── the route on the scanned geometry ────────────────────────────────────────


def test_scan_route_is_continuous_collision_free_and_inside_free_space():
    for a, b in zip(route.segs[:-1], route.segs[1:]):
        assert np.allclose(a.p1, b.p0, atol=1e-9), (a.sid, b.sid)
    pts = route.polyline(0.01)
    assert float(clearance(pts, scan).min()) > 0.40
    assert in_rooms(pts, scan).all()


def test_scan_route_keeps_the_drawing_s_shape():
    kinds = [s.kind for s in route.segs]
    assert kinds[:6] == ["approach", "curve_follow", "wall_follow",
                         "curve_follow", "wall_follow", "transit"]
    assert "corner_turn" in kinds
    assert {s.hand for s in route.segs[:5]} == {"left"}
    assert {s.hand for s in route.segs if s.follows.startswith(("branch", "end"))} \
        == {"right"}


def test_scan_route_reaches_further_than_the_tape_route():
    """The branch corridor is really 1.4 m longer, so the run up it grows."""
    up = next(s for s in route.segs if s.follows == "branch east wall")
    assert up.length > 10.0


def test_scan_start_and_goal_are_spawnable():
    assert float(clearance([scan.start.center], scan)[0]) >= 0.60
    assert float(clearance([scan.goal.center], scan)[0]) >= 0.60
    assert in_rooms([scan.start.center], scan)[0]
    assert in_rooms([scan.goal.center], scan)[0]


def test_tape_map_is_untouched_by_any_of_this():
    """The scan work must not have edited the drawing's reconstruction."""
    assert len(tape.walls) == 12
    assert tape.key_points["x_corner"] == pytest.approx(10.3505, abs=1e-9)
    assert tape.key_points["y_upper_s"] == pytest.approx(8.9154, abs=1e-9)
    assert len(tape.openings) == 1
