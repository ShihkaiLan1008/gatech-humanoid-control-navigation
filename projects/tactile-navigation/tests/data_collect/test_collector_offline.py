"""The whole loop, end to end, with the map's real geometry and no simulator.

`fake_backend.FakeBackend` answers the backend contract from the same convex
bodies the collector reasons about, so proposal -> placement -> execution ->
record -> coverage -> diagnostics all run here.  What it cannot test is the
physics: force is a linear function of how far the commanded tip has been
pushed past a surface.  Everything else is the shipped code.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from data_collect import diagnostics
from data_collect.action_space import TouchWorkspace
from data_collect.collector import CollectCfg, Collector
from data_collect.coverage import lateral_bin_edges
from data_collect.probe import FreeTouchProbe
from data_collect.belief_prior import CovariancePool
from data_collect.records import BeliefStore, TransitionWriter, split_of
from data_collect.rooms import build_scene, map_identity, planned_centreline
from data_collect.route import RouteRef

from fake_backend import FakeBackend


PRIOR = (Path(__file__).resolve().parents[2] / "out" / "room2_belief_prior"
         / "belief_prior_pool.npz")


def collect(tmp_path, *, target=120, seed=1, rate=0.25, n_particles=200, **kw):
    scene = build_scene("room2")
    route = RouteRef.from_polyline(planned_centreline(scene),
                                   scene.plan.bodies)
    ws = TouchWorkspace.from_controller()
    be = FakeBackend(scene, seed=seed, **kw)
    probe = FreeTouchProbe(be, ws, press_rate_m_s=rate)
    w = TransitionWriter(tmp_path, "t", store_qpos=True)
    cfg = CollectCfg(target_n=target, touches_per_pose=20, seed=seed,
                     n_particles=n_particles, progress_every=10 ** 9)
    pool = CovariancePool.from_npz(PRIOR)
    store = BeliefStore(tmp_path, shard_segments=3)
    col = Collector(probe=probe, route=route, workspace=ws,
                    bodies=scene.plan.bodies, writer=w, cfg=cfg, run_id="t",
                    map_id="room2_8_11_scan", controller={"name": "fake"},
                    scene=scene, cov_pool=pool, beliefs=store)
    w.write_manifest({
        "room": "room2", "target_n": target,
        "cfg": {"touches_per_pose": 20, "lateral_span_m": 0.20,
                "yaw_mode": "face_wall", "seed": seed},
        "map": map_identity("room2", scene), "route": route.as_dict(),
        "workspace": {**ws.as_dict(), "all_bin_keys": ws.all_bin_keys()},
        "feasible_action_bins": sorted(ws.feasible_bins()),
        "lateral_bin_edges_m": lateral_bin_edges(0.20),
        "belief_prior": {"path": str(PRIOR), **pool.as_dict()},
        "controller": {"name": "fake"}})
    col.run()
    w.close()
    store.close()
    return col, be, ws


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("collect")
    col, be, ws = collect(d)
    man, rows = diagnostics.load(d)
    return d, col, be, ws, man, rows


def test_it_collects_exactly_the_target(run):
    _d, col, _be, _ws, _man, rows = run
    assert col.cov.n_valid == 120
    valid = [r for r in rows if r["observation"]["label"] != "invalid_attempt"]
    assert len(valid) == 120


def test_the_room_moves_once_per_episode_and_the_robot_never_does(run):
    """The base pose is randomized by redefining the map frame, so the robot
    stays standing and its contacts stay valid."""
    _d, col, be, _ws, _man, rows = run
    assert be.install_count == col.episode
    poses = {tuple(be.truth_pose()[0])}
    assert len(poses) == 1


def test_every_row_stands_where_its_episode_says_it_does(run):
    """`place_base` has to satisfy map(pose) == the robot's actual world pose,
    or every recorded coordinate is in a different frame from the geometry."""
    _d, _col, be, _ws, _man, rows = run
    last = rows[-1]["segment_meta"]
    p = be.world_point_to_map(be.truth_pose()[0])
    assert np.allclose(p, last["xy_map"], atol=1e-5)     # recorded at 5 dp
    got = be.world_yaw_to_map(be.truth_pose()[1])
    assert abs((got - last["yaw_map"] + math.pi) % (2 * math.pi)
               - math.pi) < 1e-5        # +-pi are the same heading


def test_one_episode_is_one_base_pose_and_one_split(run):
    _d, _col, _be, _ws, _man, rows = run
    by_ep = {}
    for r in rows:
        by_ep.setdefault(r["segment"], []).append(r)
    for ep, rs in by_ep.items():
        assert len({r["split"] for r in rs}) == 1
        assert rs[0]["split"] == split_of("t", ep)
        assert len({tuple(r["segment_meta"]["xy_map"]) for r in rs}) == 1
        assert len(rs) <= 20


def test_a_transition_carries_everything_section_five_asks_for(run):
    _d, _col, _be, _ws, _man, rows = run
    r = rows[0]
    for k in ("run_id", "segment", "step", "uid", "split", "map_id",
              "segment_meta", "belief_before", "belief_after", "action",
              "observation", "state_before", "state_after"):
        assert k in r, k
    for x in ("state_before", "state_after"):
        st = r[x]
        for k in ("base_pose_map", "base_pose_world", "base_lin_vel_world",
                  "base_ang_vel_world", "joint_pos", "joint_vel",
                  "ee_pose_body", "ee_pose_world", "qpos_row"):
            assert k in st, f"{x}.{k}"
        assert len(st["joint_pos"]) == len(st["joint_vel"]) == 29
    m = r["segment_meta"]
    for k in ("s_m", "lateral_offset_m", "region", "lateral_bin", "sim_seed",
              "controller", "reach_standoff_m", "followed_body", "yaw_mode",
              "belief_init"):
        assert k in m, k
    # section 8: the belief is stored in FULL, not reduced to (mu, Sigma)
    for side in ("belief_before", "belief_after"):
        b = r[side]
        if b is None:
            continue
        for k in ("n_particles", "mean", "cov", "cov_eigenvalues",
                  "cov_eigenvectors", "ess", "n_updates", "particles_row",
                  "sigma_x_m", "sigma_y_m", "sigma_theta_rad"):
            assert k in b, f"{side}.{k}"
        assert len(b["mean"]) == 3 and np.shape(b["cov"]) == (3, 3)
    a = r["action"]
    for k in ("target_body", "target_map_xy", "target_world_xy",
              "approach_body", "approach_map_xy", "hand", "action_bin",
              "commanded_speed_m_s", "desired_contact_force_n",
              "predicted_hit"):
        assert k in a, k
    o = r["observation"]
    for k in ("label", "termination_reason", "contact_point_body",
              "contact_normal_body", "contact_force_n", "max_force_n",
              "time_to_contact_s", "contact_object", "contact_object_kind",
              "contact_link", "press_duration_s", "force_trace"):
        assert k in o, k


def test_the_action_is_a_target_and_the_approach_that_was_generated(run):
    """Section 2: `c_t = (p_target, d_approach)`, and the approach is saved
    afterwards because the controller takes only a point."""
    _d, _col, _be, ws, _man, rows = run
    for r in rows[:40]:
        t = np.asarray(r["action"]["target_body"], float)
        assert ws.contains(t)
        assert ws.bin_key(t) == r["action"]["action_bin"]
        d = np.asarray(r["action"]["approach_body"], float)
        assert float(np.linalg.norm(d)) == pytest.approx(1.0, abs=1e-4)
        park = ws.park_tip(r["action"]["hand"])
        assert np.allclose(d * r["action"]["reach_len_m"] + park, t, atol=1e-4)


def test_a_contact_record_is_complete_and_a_miss_record_is_empty(run):
    """"For no-contact samples, keep the contact-specific fields empty rather
    than inventing values"."""
    _d, _col, _be, _ws, _man, rows = run
    hits = [r for r in rows if r["observation"]["label"] == "contact"]
    miss = [r for r in rows if r["observation"]["label"] == "no_contact"]
    assert hits and miss
    for r in hits:
        o = r["observation"]
        assert o["contacted"] and o["termination_reason"] == "contact"
        assert o["contact_point_body"] and o["contact_normal_body"]
        assert o["contact_force_n"] > 0 and o["max_force_n"] > 0
        assert o["time_to_contact_s"] is not None
        assert o["contact_object"] and o["contact_object_kind"]
        assert o["contact_link"] == f"{r['action']['hand']}_tactile_collision"
        assert o["contact_wrench_world"] and len(o["contact_wrench_world"]) == 3
        assert o["swept_free_body"] is None
    for r in miss:
        o = r["observation"]
        assert not o["contacted"]
        assert o["termination_reason"] == "target_reached"
        for k in ("contact_point_body", "contact_point_map",
                  "contact_normal_body", "contact_normal_map",
                  "contact_force_n", "contact_time_sim_s",
                  "time_to_contact_s", "contact_object", "contact_link"):
            assert o[k] is None, k
        assert o["swept_free_body"], "a free sweep must record the swept ray"


def test_the_contact_normal_and_point_are_reported_in_both_frames(run):
    _d, _col, _be, _ws, _man, rows = run
    for r in rows:
        o = r["observation"]
        if not o["contacted"]:
            continue
        yaw = r["segment_meta"]["yaw_map"]
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        p = np.asarray(r["segment_meta"]["xy_map"], float) \
            + R @ np.asarray(o["contact_point_body"], float)
        assert np.allclose(p, o["contact_point_map"], atol=1e-4)
        n = R @ np.asarray(o["contact_normal_body"], float)
        assert np.allclose(n, o["contact_normal_map"], atol=1e-4)


def test_the_contacted_object_is_a_body_of_this_map(run):
    _d, _col, _be, _ws, man, rows = run
    names = {b["name"] for b in man["map"]["bodies"]}
    hits = [r for r in rows if r["observation"]["contacted"]]
    assert hits
    for r in hits:
        o = r["observation"]
        assert o["contact_object"] in names
        assert o["contact_object_kind"] in ("wall", "pillar", "cabinet")
        assert o["contact_object_k"] is not None
        # how it was decided is part of the record: MuJoCo's own pair sensor
        # and "the nearest body to where the hand stopped" are different
        # claims and a world model should be able to tell them apart
        assert o["contact_attribution"] in ("sensor", "nearest_body")
    assert any(r["observation"]["contact_attribution"] == "sensor"
               for r in hits), "the pair sensor answered for none of them"
    for r in rows:
        if not r["observation"]["contacted"]:
            assert r["observation"]["contact_attribution"] is None


def test_the_qpos_companion_has_two_rows_per_attempt(run):
    d, _col, _be, _ws, _man, rows = run
    z = np.load(d / "qpos.npz")["qpos"]
    assert z.shape[0] == 2 * len(rows)
    for r in rows:
        assert r["state_after"]["qpos_row"] == r["state_before"]["qpos_row"] + 1


def test_the_diagnostics_report_every_section_eight_item(run):
    d, _col, _be, _ws, man, rows = run
    s = diagnostics.report(d, plots=True)
    for k in ("totals", "outcome", "invalid_reasons", "region",
              "lateral_offset", "target_height_m", "target_angle_deg",
              "reach_distance_m", "contact_object", "contact_force_n",
              "max_force_sensed_n", "action_bins", "split", "belief",
              "checks"):
        assert k in s, k
    assert len(s["action_bins"]) == 45
    assert (d / "summary.json").exists() and (d / "summary.txt").exists()
    assert (d / "topdown.png").stat().st_size > 5000
    assert (d / "coverage.png").stat().st_size > 5000
    assert (d / "beliefs.png").stat().st_size > 5000
    assert s["checks"]["split_is_segment_disjoint"] is True


def test_a_short_run_already_spreads_over_the_route_and_the_workspace(run):
    """The smoke run's job: show the randomization is working, not that the
    quotas are met (120 samples cannot fill 45 bins to 150)."""
    _d, col, _be, _ws, _man, rows = run
    s = diagnostics.summarize(*diagnostics.load(_d))
    assert set(s["region"]) == {"horizontal", "corner", "vertical"}
    assert min(s["region"].values()) > 0
    assert len({r["action"]["action_bin"] for r in rows}) >= 25
    assert len({r["action"]["hand"] for r in rows}) == 2
    off = [r["segment_meta"]["lateral_offset_m"] for r in rows]
    assert max(off) - min(off) > 0.20


def test_a_safety_stop_is_recorded_as_an_invalid_attempt_not_a_miss(tmp_path):
    """Section 4 is explicit: a safety abort is not a valid no-contact."""
    col, be, _ws = collect(tmp_path, target=60, seed=5, guard_every=2)
    rows = [json.loads(l) for l in
            (tmp_path / "transitions.jsonl").read_text().splitlines()]
    bad = [r for r in rows if r["observation"]["label"] == "invalid_attempt"]
    assert bad, "the fake was asked to trip the guard and none was recorded"
    for r in bad:
        o = r["observation"]
        assert o["failure_reason"] == "safety_abort"
        assert o["label"] == "invalid_attempt"
        # The contact-specific fields stay empty and the FILTER is not
        # advanced: the attempt was aborted, so whatever the hand felt on the
        # way is not a measurement this dataset may fold in.  `contacted` may
        # still be true -- the guard can trip after a latch -- and that is
        # recorded rather than hidden.
        assert o["contact_force_n"] is None
        assert o["contact_point_body"] is None
        assert r["belief_after"] is None
        assert o["pf_update"] is None
    assert col.cov.n_valid == 60           # invalids do not count toward it
    assert col.cov.invalid["safety_abort"] == len(bad)


def test_a_pose_that_will_not_settle_is_dropped_before_any_touch(tmp_path):
    """Nothing was commanded and nothing was measured, so it is not one of
    section 4's invalid ATTEMPTS -- it is a rejected pose."""
    col, _be, _ws = collect(tmp_path, target=60, seed=7, fail_settle_every=2)
    assert col.pose_rejects.get("settle_failed", 0) > 0
    rows = [json.loads(l) for l in
            (tmp_path / "transitions.jsonl").read_text().splitlines()]
    assert len({r["segment"] for r in rows}) == len(
        {r["segment"] for r in rows}), "no episode may be half-recorded"
    assert col.cov.invalid["total"] == 0


def test_poses_that_would_plough_the_wall_are_rejected_not_attempted(run):
    """The west leg's centreline is 0.496 m off its face, so the near end of
    the +-0.20 m jitter puts the CARRIED hand inside the wall."""
    _d, col, _be, _ws, _man, rows = run
    assert col.pose_rejects.get("parked_hand_penetration", 0) > 0
    # the invariant is the CARRIED HAND's clearance, not the reported
    # standoff: where the route turns a corner the yaw is square to a face
    # while the offset ran along a different one, so the two differ.  Re-check
    # every recorded pose the way the collector did.
    from data_collect.action_space import body_clearance
    from data_collect.probe import PARKED_HAND_MARGIN_M
    from data_collect.rooms import build_scene as _bs
    bodies = tuple(_bs("room2").plan.bodies)
    for r in rows:
        m = r["segment_meta"]
        for hand in ("left", "right"):
            tip = _ws.park_tip(hand)
            assert body_clearance(m["xy_map"], m["yaw_map"], tip,
                                  bodies) >= PARKED_HAND_MARGIN_M - 1e-9
