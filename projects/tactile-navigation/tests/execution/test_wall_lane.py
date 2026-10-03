"""Wall-parallel lane geometry: no simulator, no scene-specific rules."""

from __future__ import annotations

import math

import numpy as np
import pytest

from execution.wall_lane import (WallSegment, face_present, tap_stations,
                                 wall_segments)
from global_nav.geometry import raycast, rect
from lane_graph.maps_v3 import MAPS3
from lane_graph.scene3 import build_v3_scene

STANDOFF = 0.30


def _box_room(w=4.0, h=3.0, t=0.10):
    """A closed rectangular room: four slabs, interior (0,0)-(w,h)."""
    return (rect(-t, w + t, -t, 0.0, "south"), rect(-t, w + t, h, h + t, "north"),
            rect(-t, 0.0, 0.0, h, "west"), rect(w, w + t, 0.0, h, "east"))


def test_lane_is_offset_from_the_surface_along_the_face_normal():
    seg = WallSegment(p0=np.array([1.0, 0.0]), tangent=np.array([1.0, 0.0]),
                      normal=np.array([0.0, 1.0]), length=3.0)
    assert seg.lane_point(0.0, 0.3) == pytest.approx([1.0, 0.3])
    assert seg.lane_point(2.0, 0.3) == pytest.approx([3.0, 0.3])
    assert seg.yaw == pytest.approx(0.0)
    # projections invert the construction
    assert seg.arc_length_of(seg.lane_point(1.7, 0.3)) == pytest.approx(1.7)
    assert seg.standoff_of(seg.lane_point(1.7, 0.3)) == pytest.approx(0.3)


def test_hand_side_follows_the_wall_and_flips_with_travel_direction():
    seg = WallSegment(p0=np.array([0.0, 0.0]), tangent=np.array([1.0, 0.0]),
                      normal=np.array([0.0, -1.0]), length=3.0)
    assert seg.hand_side == 1                     # wall at +y -> left hand
    back = seg.reversed()
    assert back.hand_side == -1                   # same face, other way -> right
    assert back.normal == pytest.approx(seg.normal)
    assert back.length == pytest.approx(seg.length)
    assert back.surface_point(back.length) == pytest.approx(seg.surface_point(0.0))


def test_outward_faces_are_rejected_because_the_lane_is_outside():
    segs = wall_segments(_box_room(), standoff_m=STANDOFF, min_length_m=1.0)
    assert segs, "a closed room must offer walkable lanes"
    for s in segs:
        lane_mid = s.lane_point(s.length / 2.0, STANDOFF)
        assert 0.0 < lane_mid[0] < 4.0 and 0.0 < lane_mid[1] < 3.0


def test_a_lane_blocked_by_an_obstacle_is_split_not_returned_whole():
    room = _box_room(w=6.0, h=3.0)
    blocked = room + (rect(2.7, 3.3, 0.0, 0.5, "pillar"),)
    long_clear = max(s.length for s in wall_segments(
        room, standoff_m=STANDOFF, min_length_m=0.5)
        if s.normal[1] > 0.5)
    pieces = [s for s in wall_segments(blocked, standoff_m=STANDOFF,
                                       min_length_m=0.5)
              if s.normal[1] > 0.5 and "south" in s.bodies]
    assert len(pieces) == 2
    assert max(p.length for p in pieces) < long_clear
    for p in pieces:                              # neither piece straddles the pillar
        xs = [p.lane_point(s, STANDOFF)[0] for s in (0.0, p.length)]
        assert not (min(xs) < 3.0 < max(xs))


def test_a_recess_is_walkable_but_not_tappable():
    """The base walks past a bay; only the touch needs a face."""
    room = _box_room(w=6.0, h=3.0, t=0.10)
    # cut a 0.9 m alcove into the south wall and re-close it 0.5 m further out
    walls = (rect(-0.1, 2.5, -0.1, 0.0, "s0"), rect(3.4, 6.1, -0.1, 0.0, "s1"),
             rect(2.5, 3.4, -0.6, -0.5, "bay_back"),
             rect(2.5, 2.6, -0.5, 0.0, "bay_l"), rect(3.3, 3.4, -0.5, 0.0, "bay_r"),
             rect(-0.1, 6.1, 3.0, 3.1, "north"), rect(-0.1, 0.0, 0.0, 3.0, "west"),
             rect(6.0, 6.1, 0.0, 3.0, "east"))
    south = [s for s in wall_segments(walls, standoff_m=STANDOFF, min_length_m=1.0)
             if s.normal[1] > 0.5]
    assert len(south) == 1 and south[0].length > 4.0      # one continuous lane
    seg = south[0]
    taps = tap_stations(seg, 0.30, walls, standoff_m=STANDOFF,
                        first_offset_m=0.0, end_margin_m=0.0)
    assert taps, "the wall either side of the bay is tappable"
    gap = [s for s in np.arange(0.0, seg.length, 0.05)
           if not face_present(seg, s, walls, standoff_m=STANDOFF)]
    assert gap, "the bay mouth has no face at the standoff"
    for s in taps:                                        # no station in the mouth
        assert face_present(seg, s, walls, standoff_m=STANDOFF)


def test_tap_stations_are_evenly_spaced_by_arc_length():
    room = _box_room(w=6.0, h=3.0)
    seg = max((s for s in wall_segments(room, standoff_m=STANDOFF, min_length_m=1.0)
               if s.normal[1] > 0.5), key=lambda s: s.length)
    taps = tap_stations(seg, 0.8, room, standoff_m=STANDOFF,
                        first_offset_m=0.4, end_margin_m=0.3)
    assert len(taps) >= 4
    assert np.allclose(np.diff(taps), 0.8)
    assert taps[0] == pytest.approx(0.4)
    assert taps[-1] <= seg.length - 0.3 + 1e-9


@pytest.mark.parametrize("name", sorted(MAPS3))
def test_every_v3_map_offers_a_walkable_tappable_segment(name):
    scene = build_v3_scene(name)
    bodies = scene.plan.bodies
    segs = wall_segments(bodies, standoff_m=STANDOFF, min_length_m=1.5,
                         both_directions=True)
    assert segs, f"{name}: no walkable wall lane at {STANDOFF} m"
    good = [s for s in segs
            if len(tap_stations(s, 0.8, bodies, standoff_m=STANDOFF)) >= 2]
    assert good, f"{name}: no segment carries two taps"
    best = good[0]
    assert best.length >= 1.5
    # heading comes from the surface: the lane is parallel to the face it offsets
    for s in np.linspace(0.0, best.length, 9):
        assert best.standoff_of(best.lane_point(s, STANDOFF)) == pytest.approx(STANDOFF)
    # and both hands are reachable across the returned set
    assert {s.hand_side for s in segs} == {1, -1}


@pytest.mark.parametrize("name", sorted(MAPS3))
def test_tap_stations_see_the_face_the_press_ray_would_hit(name):
    scene = build_v3_scene(name)
    bodies = scene.plan.bodies
    for seg in wall_segments(bodies, standoff_m=STANDOFF, min_length_m=1.5)[:4]:
        for s in tap_stations(seg, 0.8, bodies, standoff_m=STANDOFF):
            lane = seg.lane_point(s, STANDOFF)
            d, n, _ = raycast(lane, -np.asarray(seg.normal, float), 2.0, bodies)
            assert d == pytest.approx(STANDOFF, abs=0.02)
            # the tangent of the hit surface IS the heading the walk holds
            tang = np.array([-n[1], n[0]])
            assert abs(abs(float(tang @ seg.tangent)) - 1.0) < 1e-6
            assert abs(math.atan2(n[1], n[0])
                       - math.atan2(seg.normal[1], seg.normal[0])) < 1e-6
