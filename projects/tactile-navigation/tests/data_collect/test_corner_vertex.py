"""Standing beside the corner VERTEX, which the centreline cannot do.

The reference centreline TURNS at the corner, so no pose on it ever stands
alongside the vertex on either face: the elbow is 0.656 m from it and the
hand, leading +-0.35 m along the face, reached y = 3.487 on the west face and
x = -0.415 on the north.  The last ~0.14 m of each face went untouched, and
the exact corner -- the only place a touch can feel two faces -- had no data.

The vertex is therefore its own region, sampled from the two faces directly.
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from data_collect.action_space import corner_standoff_m
from data_collect.coverage import REGION_SHARE, region_share_with_vertex
from data_collect.records import BeliefStore, TransitionWriter, load_particles
from data_collect.rooms import build_scene, planned_centreline
from data_collect.route import RouteRef


@pytest.fixture(scope="module")
def ref():
    scene = build_scene("room2")
    return RouteRef.from_polyline(planned_centreline(scene),
                                  scene.plan.bodies)


def test_the_corner_vertex_is_recovered_from_the_route(ref):
    """The faces either side of the turn are lines `{x : (x-p)·n = -d}`; two
    of them meet at the vertex.  Room 2's is the origin of its scan frame."""
    vs = ref.corner_vertices()
    assert len(vs) == 1
    v = vs[0]
    assert np.allclose(v.xy, [-0.55, 3.75], atol=0.01)
    assert set(v.bodies) == {"wall_0", "wall_1"}
    # the two faces are perpendicular, and each tangent runs back down its leg
    n0, n1 = (np.asarray(x, float) for x in v.normals)
    assert abs(float(n0 @ n1)) < 1e-6
    for n, t in zip(v.normals, v.tangents):
        assert abs(float(np.asarray(n, float) @ np.asarray(t, float))) < 1e-6


def test_a_vertex_pose_stands_square_to_one_face_beside_the_corner(ref):
    v = ref.corner_vertices()[0]
    s0 = corner_standoff_m()
    for face in (0, 1):
        n = np.asarray(v.normals[face], float)
        t = np.asarray(v.tangents[face], float)
        for a in (0.30, 0.50, 0.70):
            p = ref.vertex_pose(v, face, a, 0.0, s0)
            assert p.region == "vertex"
            assert p.wall_body == v.bodies[face]
            # square to the face it follows
            head = np.array([math.cos(p.yaw_map), math.sin(p.yaw_map)])
            assert float(head @ n) == pytest.approx(-1.0, abs=1e-6)
            # `a` from the vertex along the face, `s0` off it
            off = np.asarray(p.xy_map, float) - np.asarray(v.xy, float)
            assert float(off @ t) == pytest.approx(a, abs=1e-9)
            assert float(off @ n) == pytest.approx(s0, abs=1e-9)
            # and the reach it has to cover is the standoff, not sqrt(2) of it
            assert p.stand_dist_m == pytest.approx(s0, abs=1e-3)


def test_a_vertex_pose_can_put_the_hand_on_the_vertex(ref):
    """`a` down to the base's own footprint clearance, plus a 0.35 m hand
    lead along the face, reaches the corner itself -- which is the point."""
    from execution.vforce_backend import TRAINED_TIP_Y_M
    from execution.wall_lane import LANE_CLEAR_M

    v = ref.corner_vertices()[0]
    lead = float(TRAINED_TIP_Y_M[1])
    assert lead >= LANE_CLEAR_M, (
        "the hand cannot lead as far along the face as the base must stand "
        "back from the other wall, so the vertex is out of reach")
    p = ref.vertex_pose(v, 0, LANE_CLEAR_M, 0.0, corner_standoff_m())
    t = np.asarray(v.tangents[0], float)
    # the hand's furthest reach along the face, measured from the vertex
    reach_from_vertex = LANE_CLEAR_M - lead
    assert reach_from_vertex <= 0.0
    assert p.stand_dist_m < 0.48                      # and within the reach


def test_the_vertex_share_rescales_the_route_regions():
    assert region_share_with_vertex(0.0) == REGION_SHARE
    r = region_share_with_vertex(0.10)
    assert r["vertex"] == pytest.approx(0.10)
    assert sum(r.values()) == pytest.approx(1.0)
    for k, v in REGION_SHARE.items():
        assert r[k] == pytest.approx(v * 0.90)


def test_the_vertex_region_collects_contacts_on_the_corner_itself():
    """End to end against the fake backend: with a vertex share, contacts
    land within centimetres of the vertex instead of 0.63 m from it."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_backend import FakeBackend

    from data_collect import diagnostics
    from data_collect.action_space import TouchWorkspace
    from data_collect.belief_prior import CovariancePool
    from data_collect.collector import CollectCfg, Collector
    from data_collect.coverage import lateral_bin_edges
    from data_collect.probe import FreeTouchProbe
    from data_collect.rooms import map_identity
    from test_collector_offline import PRIOR

    scene = build_scene("room2")
    route = RouteRef.from_polyline(planned_centreline(scene),
                                   scene.plan.bodies)
    ws = TouchWorkspace.from_controller()
    probe = FreeTouchProbe(FakeBackend(scene), ws, press_rate_m_s=0.3)
    d = Path(tempfile.mkdtemp())
    w = TransitionWriter(d, "t", store_qpos=False)
    cfg = CollectCfg(target_n=300, touches_per_pose=12, seed=5,
                     n_particles=120, progress_every=10 ** 9,
                     vertex_share=0.25)
    col = Collector(probe=probe, route=route, workspace=ws,
                    bodies=scene.plan.bodies, writer=w, cfg=cfg, run_id="t",
                    map_id="m", controller={}, scene=scene,
                    cov_pool=CovariancePool.from_npz(PRIOR),
                    beliefs=BeliefStore(d))
    w.write_manifest({"room": "room2", "target_n": 300,
                      "cfg": {"n_particles": 120, "seed": 5, "n_envs": 1},
                      "map": map_identity("room2", scene),
                      "route": route.as_dict(),
                      "workspace": {**ws.as_dict(),
                                    "all_bin_keys": ws.all_bin_keys()},
                      "feasible_action_bins": sorted(ws.feasible_bins()),
                      "lateral_bin_edges_m": lateral_bin_edges(0.20),
                      "controller": {}})
    col.run()
    w.close()
    col.beliefs.close()
    s = diagnostics.report(d, plots=False)

    assert s["region"]["vertex"] > 0.15 * s["totals"]["valid"]
    vp = s["vertex_proximity"]["per_vertex"][0]
    assert vp["closest_contact_m"] < 0.15
    assert vp["within_0.30m"] >= 5
    assert s["checks"]["pf_chain_is_recursive"] is True


def test_append_extends_a_dataset_instead_of_replacing_it():
    """Rows appended, belief shards continued, old particle rows still
    resolving to the beliefs they always did."""
    d = Path(tempfile.mkdtemp())
    st = BeliefStore(d, shard_segments=2)
    for i in range(4):
        st.add({"particles": np.full((5, 4), i, "float32")})
        st.end_segment()
    st.close()
    w = TransitionWriter(d, "r")
    w.write({"uid": "a", "segment": 0})
    w.write_manifest({"target_n": 4})
    w.close()

    st2 = BeliefStore(d, shard_segments=2, append=True)
    assert st2._rows == 4 and st2._shard == 2
    row = st2.add({"particles": np.full((5, 4), 99, "float32")})
    st2.end_segment()
    st2.close()
    w2 = TransitionWriter(d, "r", append=True)
    w2.write({"uid": "b", "segment": 1})
    w2.write_manifest({"target_n": 12})
    w2.close()

    assert row == 4
    assert float(load_particles(d, 4)[0, 0]) == 99.0
    assert float(load_particles(d, 0)[0, 0]) == 0.0
    uids = [json.loads(l)["uid"] for l in
            (d / "transitions.jsonl").read_text().splitlines()]
    assert uids == ["a", "b"]
    man = json.loads((d / "manifest.json").read_text())
    assert man["target_n"] == 12
    assert man["previous_runs"][0]["target_n"] == 4
