"""The route reference: arc length, the perpendicular, and the three regions."""

import math

import numpy as np
import pytest

from data_collect.rooms import build_scene, planned_centreline
from data_collect.route import RouteRef


@pytest.fixture(scope="module")
def ref():
    scene = build_scene("room2")
    return RouteRef.from_polyline(planned_centreline(scene), scene.plan.bodies)


def test_the_centreline_is_uniformly_sampled(ref):
    """`s ~ U(0, L)` has to be uniform on the ROUTE, not on its node list.

    The planner emits 0.05 m lane polylines and 1.0 m transit hops in one
    route; sampling nodes would put 20x the weight on a metre of lane."""
    d = np.linalg.norm(np.diff(ref.xy, axis=0), axis=1)
    # Every step is the same length but the ones that straddle a route VERTEX,
    # whose chord is shorter than the two arcs it replaces.  room2 has one
    # such vertex (the 148 deg jog off the west lane).
    short = d < 0.98 * np.median(d)
    assert short.sum() <= 6, d[short]      # heading changes + the corner fillet
    # 0.05 m steps agreeing to a tenth of a millimetre; the corner
    # reconstruction resamples twice, so this is not bit-exact
    assert d[~short].max() - d[~short].min() < 1e-3
    assert ref.length_m > 4.0

    # ...and that is what makes `s ~ U(0, L)` uniform on the ROUTE rather than
    # on a node list where lane polylines outnumber transit hops 20 to 1
    rng = np.random.default_rng(0)
    draws = np.array([ref.sample_s(rng) for _ in range(4000)])
    counts, _ = np.histogram(draws, bins=8, range=(0.0, ref.length_m))
    assert counts.min() > 0.75 * counts.mean()


def test_the_walk_onto_the_lane_is_not_part_of_the_reference(ref):
    """room2's planned route opens with 0.48 m of FREE_TRANSIT that darts
    from the start pose to the lane -- (0,0) -> (0.01,-0.238) ->
    (-0.054,-0.004).  A base pose sampled there faces open floor."""
    from data_collect.rooms import _astar_polyline, _drop_leading_transit
    scene = build_scene("room2")
    raw, kinds = _astar_polyline(scene)
    full = float(np.sum(np.linalg.norm(np.diff(raw, axis=0), axis=1)))
    kept = _drop_leading_transit(raw, kinds)
    trimmed = float(np.sum(np.linalg.norm(np.diff(kept, axis=0), axis=1)))
    assert full - trimmed > 0.4, "the transit stub was not trimmed"
    # what survives starts ON the lane: a wall to follow at every node, and a
    # heading that runs along it rather than at it
    assert np.isfinite(ref.wall_dist).all()
    assert ref.wall_dist[0] == pytest.approx(0.496, abs=0.02)
    assert abs(float(ref.tangent[0] @ ref.normal[0])) < 0.05


def test_every_node_knows_which_wall_it_follows_and_faces_it(ref):
    """`face_wall` yaw must point INTO the followed surface, on every leg."""
    for i in range(0, len(ref.xy), 7):
        s = float(ref.s[i])
        p = ref.pose_at(s, 0.0, yaw_mode="face_wall")
        head = np.array([math.cos(p.yaw_map), math.sin(p.yaw_map)])
        assert float(head @ p.route_normal) < -0.9, s
        assert p.wall_body, s


def test_the_normal_points_away_from_the_wall_on_both_legs(ref):
    """A positive lateral offset means "further into the room" everywhere, or
    the +-0.20 m histogram is two different histograms glued together."""
    for s in (0.5, 1.5, 4.5, 5.2):
        near = ref.pose_at(s, -0.10)
        far = ref.pose_at(s, +0.10)
        # `stand_dist_m` is what a REACH has to cover: a ray cast along the
        # pose's own facing direction.  On a straight leg that is the
        # perpendicular standoff, so moving away from the wall lengthens it.
        assert far.stand_dist_m > near.stand_dist_m
        assert near.stand_dist_m == pytest.approx(near.wall_dist_m - 0.10,
                                                  abs=2e-3)


def test_room2_is_a_vertical_leg_a_corner_and_a_horizontal_leg(ref):
    spans = ref.spans()
    assert set(spans) == {"vertical", "corner", "horizontal"}
    assert len(ref.corner_centres_m) == 1, (
        "the 90 deg turn is taken in three pieces -- a 148 deg jog off the "
        "west lane, a 44 deg diagonal, then the north lane -- and reading "
        "them as three turns makes the corner region half the route")
    for name in ("vertical", "corner", "horizontal"):
        assert ref.region_length_m(name) > 0.9, name


def test_the_whole_corner_region_is_within_a_jitter_of_a_wall(ref):
    """The planner's diagonal cut left both faces 0.7-1.1 m away, so no touch
    taken in the corner region could reach anything -- an empty quadrant in
    the coverage plot.  With the elbow, every point of the corner region is
    close enough that the +-0.20 m lateral jitter can bring a face into the
    arm's 0.48 m reach."""
    lo, hi = ref.corner_spans_m[0]
    near = np.array([float(ref.wall_dist[i]) for i, s in enumerate(ref.s)
                     if lo <= s <= hi])
    assert len(near) > 10
    assert near.max() < 0.62, "part of the corner region is out of reach"
    assert near.min() < 0.50
    # and it reaches onto both straight legs, so the corner's contacts are not
    # all taken from one face
    reg = [ref.region_at(s) for s in (lo - 0.15, hi + 0.15)]
    assert set(reg) == {"vertical", "horizontal"}


def test_regions_are_sampled_inside_their_own_spans(ref):
    rng = np.random.default_rng(0)
    for name in ("vertical", "corner", "horizontal"):
        for _ in range(50):
            s = ref.sample_s(rng, name)
            assert ref.region_at(s) == name


def test_a_pose_is_the_centreline_pushed_along_the_perpendicular(ref):
    p = ref.pose_at(1.0, 0.13)
    off = np.asarray(p.xy_map) - np.asarray(p.route_xy_map)
    assert float(np.linalg.norm(off)) == pytest.approx(0.13, abs=1e-9)
    assert float(off @ p.route_normal) == pytest.approx(0.13, abs=1e-9)
    assert abs(float(off @ p.route_tangent)) < 1e-9


def test_along_route_yaw_is_available_and_is_the_travel_direction(ref):
    p = ref.pose_at(1.0, 0.0, yaw_mode="along_route")
    t = np.array([math.cos(p.yaw_map), math.sin(p.yaw_map)])
    assert float(t @ p.route_tangent) > 0.999
    with pytest.raises(ValueError):
        ref.pose_at(1.0, 0.0, yaw_mode="sideways")


@pytest.mark.slow
def test_the_room_flag_generalises_to_the_other_scan():
    """`--room demo_room` has to be a working path, not a placeholder: it is
    a 25.7 m route with TWO turns and curved wall features, so it exercises
    the multi-turn branch of the region classifier and the "followed body is
    a cylinder" branch of the wall resolver."""
    scene = build_scene("demo_room")
    ref = RouteRef.from_polyline(planned_centreline(scene), scene.plan.bodies)
    assert ref.length_m > 20.0
    assert len(ref.corner_centres_m) >= 2
    spans = ref.spans()
    assert set(spans) == {"horizontal", "corner", "vertical"}
    for name in spans:
        assert ref.region_length_m(name) > 1.0, name
    bodies = {ref.pose_at(s, 0.0).wall_body
              for s in np.linspace(0.2, ref.length_m - 0.2, 40)}
    assert len(bodies) > 2, "every stretch followed the same body"
    assert any(b.startswith("bulge") for b in bodies), (
        "the curved features are followable and must resolve as the wall")


def test_the_corner_shortcut_is_replaced_by_the_wall_following_elbow():
    """The planner rounds room2's corner on a diagonal across open floor.

    Measured on its raw route: the cut passes 1.13 m from the corner VERTEX
    and 0.79 m from any face, against an arm that reaches 0.48 m -- so with
    base poses drawn +-0.20 m about the centreline, the one landmark in the
    room collected no contact data at all.  It was visible as an empty
    quadrant in the coverage plot.

    The elbow is the point each leg's own standoff from each of the two
    faces, so the straight legs are untouched and only the crossing changes."""
    from data_collect.rooms import build_scene, planned_centreline

    scene = build_scene("room2")
    bodies = tuple(scene.plan.bodies)
    vertex = np.array([-0.55, 3.75])       # where the two inner faces meet

    def profile(hug):
        ref = RouteRef.from_polyline(
            planned_centreline(scene, hug_corner=hug), bodies)
        near = np.array([min(float(b.signed_distance(p.reshape(1, 2))[0][0])
                             for b in bodies) for p in ref.xy])
        return ref, near, np.linalg.norm(ref.xy - vertex, axis=1).min()

    raw, near_raw, vtx_raw = profile(False)
    hug, near_hug, vtx_hug = profile(True)

    assert near_raw.max() > 0.75 and vtx_raw > 1.1, "the raw cut moved"
    # the elbow brings the whole route back within a jitter of the wall
    assert near_hug.max() < 0.62
    assert vtx_hug < 0.85
    # ...and leaves the straight legs where the planner put them
    for s in (0.5, 1.5):
        assert hug.pose_at(s, 0.0).wall_dist_m == pytest.approx(
            raw.pose_at(s, 0.0).wall_dist_m, abs=0.01)
    # both faces of the corner are within reach of a pose on the elbow
    i = int(np.argmin(np.linalg.norm(hug.xy - vertex, axis=1)))
    p = hug.xy[i] + 0.15 * (vertex - hug.xy[i]) / np.linalg.norm(
        vertex - hug.xy[i])
    assert abs(p[0] - (-0.55)) < 0.48 and abs(3.75 - p[1]) < 0.48
