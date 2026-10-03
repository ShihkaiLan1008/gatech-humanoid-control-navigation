"""The signed-distance function and the ray-cast, against hand-computed answers.

Everything downstream is a likelihood built on these two, so an error here would be
invisible: it would come out as a filter that is merely a bit worse, which is
exactly the kind of bug that survives.
"""

import math

import numpy as np
import pytest

from global_nav.geometry import ConvexPolygon, raycast, rect, se2_apply, wrap_angle


def test_rectangle_signed_distance_matches_hand_computation():
    box = rect(0.0, 2.0, 0.0, 1.0)
    pts = np.array([
        [1.0, 2.0],     # 1.0 above the top face
        [-0.5, 0.5],    # 0.5 left of the left face
        [1.0, 0.4],     # inside: nearest face is the bottom, 0.4 away
        [3.0, 2.0],     # off the top-right corner: sqrt(1 + 1)
    ])
    sd, _ = box.signed_distance(pts)
    assert sd == pytest.approx([1.0, 0.5, -0.4, math.sqrt(2.0)], abs=1e-12)


def test_outward_normals_point_away_from_the_body():
    box = rect(0.0, 2.0, 0.0, 1.0)
    _, n = box.signed_distance(np.array([[1.0, 2.0], [-0.5, 0.5], [1.0, -3.0]]))
    np.testing.assert_allclose(n, [[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], atol=1e-12)


def test_winding_is_normalized_so_a_clockwise_polygon_still_works():
    ccw = ConvexPolygon(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    cw = ConvexPolygon(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]))
    # points with an UNAMBIGUOUS nearest face; the exact centre is equidistant from
    # all four and its normal is a tie-break, not a fact about the geometry
    p = np.array([[0.5, 0.3], [0.5, 2.0], [-1.0, 0.5]])
    np.testing.assert_allclose(ccw.signed_distance(p)[0], cw.signed_distance(p)[0], atol=1e-12)
    np.testing.assert_allclose(ccw.signed_distance(p)[1], cw.signed_distance(p)[1], atol=1e-12)


def test_penetration_agrees_with_signed_distance_but_skips_the_outside():
    """``penetration`` is the optimized swept-free path -- it must not be a
    different function, only a cheaper one."""
    box = rect(0.0, 2.0, 0.0, 1.0)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.5, 3.0, size=(400, 2))
    sd, _ = box.signed_distance(pts)
    assert box.penetration(pts) == pytest.approx(np.maximum(-sd, 0.0), abs=1e-12)


def test_per_point_offsets_translate_the_polygon_not_the_points():
    box = rect(0.0, 2.0, 0.0, 1.0)
    pts = np.array([[1.0, 2.0], [1.0, 2.0]])
    offsets = np.array([[0.0, 0.0], [0.0, 0.5]])   # second: box moves up 0.5
    sd, _ = box.signed_distance(pts, offsets)
    assert sd == pytest.approx([1.0, 0.5])


def test_distance_to_aabb_lower_bounds_the_true_distance():
    """The nearest-surface pruning is only correct if this bound never overshoots."""
    poly = ConvexPolygon(np.array([[0.0, 0.0], [2.0, 0.3], [1.6, 1.4], [0.2, 1.1]]))
    rng = np.random.default_rng(1)
    pts = rng.uniform(-2.0, 4.0, size=(500, 2))
    sd, _ = poly.signed_distance(pts)
    assert np.all(poly.distance_to_aabb(pts) <= np.abs(sd) + 1e-12)


def test_raycast_returns_first_entry_and_its_normal():
    near = rect(1.0, 2.0, -1.0, 1.0, "near")
    far = rect(3.0, 4.0, -1.0, 1.0, "far")
    d, n, i = raycast(np.zeros(2), np.array([1.0, 0.0]), 10.0, (far, near))
    assert d == pytest.approx(1.0)
    assert n == pytest.approx([-1.0, 0.0])
    assert i == 1                                    # the NEAR body, not the first listed


def test_raycast_misses_return_none_which_is_the_swept_free_observation():
    box = rect(1.0, 2.0, -1.0, 1.0)
    assert raycast(np.zeros(2), np.array([0.0, 1.0]), 10.0, (box,))[0] is None
    assert raycast(np.zeros(2), np.array([1.0, 0.0]), 0.5, (box,))[0] is None  # out of reach


def test_raycast_is_exact_on_an_oblique_ray():
    box = rect(-5.0, 5.0, 1.0, 2.0)
    d, n, _ = raycast(np.zeros(2), np.array([math.cos(math.radians(60.0)),
                                             math.sin(math.radians(60.0))]), 5.0, (box,))
    assert d == pytest.approx(1.0 / math.sin(math.radians(60.0)))
    assert n == pytest.approx([0.0, -1.0])


def test_se2_apply_and_wrap_angle():
    xy = np.array([[1.0, 2.0], [1.0, 2.0]])
    out = se2_apply(xy, np.array([0.0, math.pi / 2]), np.array([1.0, 0.0]))
    np.testing.assert_allclose(out, [[2.0, 2.0], [1.0, 3.0]], atol=1e-12)
    assert wrap_angle(2.5 * math.pi) == pytest.approx(0.5 * math.pi)
    assert wrap_angle(-2.5 * math.pi) == pytest.approx(-0.5 * math.pi)
    assert abs(wrap_angle(3.0 * math.pi)) == pytest.approx(math.pi)   # +-pi is a tie
