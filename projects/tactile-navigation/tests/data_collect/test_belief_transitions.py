"""Section 11's belief checks, run offline against the fake backend.

The dataset's claim is `(b_t, c_t, M) -> (o_{t+1}, b_{t+1})` where `b` is a
live particle filter.  Everything here checks that claim on the written file,
not on the collector's own bookkeeping.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from data_collect import diagnostics
from data_collect.belief_prior import CovariancePool, weighted_moments
from data_collect.belief_segment import BeliefSegment, observations_from
from data_collect.records import load_particles
from data_collect.rooms import build_scene

from test_collector_offline import PRIOR, collect


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("belief")
    col, be, ws = collect(d, target=100, seed=4, n_particles=300)
    rows = [json.loads(l) for l in
            (d / "transitions.jsonl").read_text().splitlines()]
    return d, col, rows


def test_the_covariance_pool_is_measured_not_invented():
    """Section 3: do not choose a covariance before checking the data.  The
    pool holds matrices the Room 2 filter really carried, bucketed by region,
    and at least some of them came from actual particle sets."""
    pool = CovariancePool.from_npz(PRIOR)
    assert set(pool.regions()) == {"horizontal", "corner", "vertical"}
    for r in pool.regions():
        assert pool.count(r) >= 20, r
    srcs = pool.as_dict()["sources"]
    assert "particles" in srcs, (
        "no particle-derived covariance in the pool -- run "
        "scripts/room2_run.py --pf-log and rebuild the prior")
    for s in pool.pool_for("corner"):
        w = np.linalg.eigvalsh(s.cov)
        assert (w > 0).all(), "a pooled covariance is not positive definite"


def test_sampling_the_pool_returns_a_covariance_that_was_observed():
    """A bootstrap, not a draw from a fitted prior: every sampled matrix is an
    observed one times a single scalar, so its SHAPE is one the filter had."""
    pool = CovariancePool.from_npz(PRIOR)
    rng = np.random.default_rng(0)
    observed = {tuple(np.round(s.cov.ravel() / max(s.cov[0, 0], 1e-12), 6))
                for s in pool.pool_for("vertical")}
    for _ in range(30):
        cov, prov = pool.sample(rng, "vertical")
        shape = tuple(np.round(cov.ravel() / max(cov[0, 0], 1e-12), 6))
        assert shape in observed, "the sampled shape was never observed"
        assert 0.3 < prov["jitter_scale"] < 3.0
        assert prov["source"] in ("particles", "nav_trace")


def test_a_pure_bootstrap_reproduces_the_pool_exactly():
    pool = CovariancePool.from_npz(PRIOR)
    pool.jitter_log_sigma = 0.0
    rng = np.random.default_rng(1)
    seen = {tuple(np.round(pool.sample(rng, "corner")[0].ravel(), 10))
            for _ in range(200)}
    have = {tuple(np.round(s.cov.ravel(), 10)) for s in pool.pool_for("corner")}
    assert seen <= have


def test_b0_is_placed_near_the_truth_but_never_on_it():
    """Section 4: the filter gets a PLAUSIBLE ESTIMATE around the true pose,
    never the true pose.  And the offset is drawn from the same covariance the
    particles are, so a belief is never 0.3 m wrong with a 0.05 m sigma."""
    scene = build_scene("room2")
    pool = CovariancePool.from_npz(PRIOR)
    rng = np.random.default_rng(7)
    gt = np.array([-0.05, 1.2, math.pi])
    offs, ratios = [], []
    for _ in range(40):
        seg = BeliefSegment(scene, pool, rng, region="vertical",
                            gt_pose_map=gt, n_particles=300)
        mu = seg.b.pose_estimate()
        d = float(np.linalg.norm(mu[:2] - gt[:2]))
        assert d > 1e-6, "the filter was handed the ground-truth pose"
        offs.append(d)
        sig = math.sqrt(max(seg.init.cov0[0, 0] + seg.init.cov0[1, 1], 1e-12))
        ratios.append(d / sig)
    assert 0.005 < np.median(offs) < 0.5
    # the offset is O(sigma): a Mahalanobis-scale check, not a magic number
    assert np.median(ratios) < 2.5


def test_the_filter_is_alive_across_a_segment(run):
    """Section 5: the same PF, folded forward once per touch.  `n_updates`
    counts, and the particle row `b_{t+1}` was filed to is the row the next
    step reads as `b_t`."""
    _d, _col, rows = run
    per = {}
    for r in rows:
        per.setdefault(r["segment"], []).append(r)
    assert len(per) >= 2
    for seg, rs in per.items():
        rs.sort(key=lambda r: r["step"])
        assert [r["step"] for r in rs] == list(range(len(rs)))
        prev = None
        for i, r in enumerate(rs):
            b0, b1 = r["belief_before"], r["belief_after"]
            assert b0["n_updates"] == i, (seg, r["step"])
            assert b1["n_updates"] == i + 1
            if prev is not None:
                assert b0["particles_row"] == prev
            prev = b1["particles_row"]


def test_the_stored_particles_are_the_stored_gaussian(run):
    """`(mu, Sigma)` in the row must be the moments of the particle set beside
    it, or the two representations the dataset offers disagree."""
    _d, _col, rows = run
    d = _d
    for r in rows[:12]:
        for side in ("belief_before", "belief_after"):
            b = r[side]
            p = load_particles(d, b["particles_row"])
            assert p.shape[0] == b["n_particles"]
            mu, cov = weighted_moments(p[:, :2], p[:, 2], p[:, 3])
            assert np.allclose(mu, b["mean"], atol=1e-4), side
            assert np.allclose(cov, b["cov"], atol=1e-6), side


def test_an_accepted_contact_actually_moves_the_belief(run):
    """The pipeline property this can check offline: an accepted contact
    changes the posterior -- both its spread and its mean.

    Whether it TIGHTENS is a question about physics, and the fake backend's
    contact geometry is a linear penetration model, so it cannot answer it:
    only 35 % of its accepted contacts shrink the trace.  Measured on the real
    12 000-transition dataset, 65.5 % of accepted contacts and 68.4 % of free
    sweeps tighten the position covariance (median -8.8e-5 m2, p10
    -2.3e-3 m2).  That number belongs in the dataset's own summary, not in a
    test that runs without a simulator."""
    _d, _col, rows = run
    moved_cov, moved_mean, n = 0, 0, 0
    for r in rows:
        pf = r["observation"].get("pf_update") or {}
        if (r["observation"]["label"] != "contact"
                or not pf.get("n_contacts") or pf.get("n_outliers")):
            continue                       # not folded in: nothing to move
        b0, b1 = r["belief_before"], r["belief_after"]
        n += 1
        t0 = b0["cov"][0][0] + b0["cov"][1][1]
        t1 = b1["cov"][0][0] + b1["cov"][1][1]
        moved_cov += abs(t1 - t0) > 1e-9
        moved_mean += float(np.linalg.norm(
            np.asarray(b1["mean"][:2]) - np.asarray(b0["mean"][:2]))) > 1e-9
    assert n >= 10
    assert moved_cov == n, f"{n - moved_cov}/{n} accepted contacts left the "\
                          "covariance untouched"
    assert moved_mean == n


def test_a_no_contact_is_folded_in_as_a_swept_free_observation(run):
    """A miss is evidence too -- the bounded free sweep is exactly the span
    the tip crossed without meeting anything."""
    _d, _col, rows = run
    miss = [r for r in rows if r["observation"]["label"] == "no_contact"]
    assert miss
    assert all(r["observation"]["pf_update"]["n_free"] == 1 for r in miss)
    assert all(r["observation"]["pf_update"]["n_contacts"] == 0 for r in miss)


def test_an_invalid_attempt_advances_nothing(run):
    """Nothing was measured, so nothing may be folded in: `b_{t+1}` is absent
    and the next step starts from the same belief."""
    _d, col, rows = run
    bad = [r for r in rows if r["observation"]["label"] == "invalid_attempt"]
    for r in bad:
        assert r["belief_after"] is None
    assert col.n_pf_updates == sum(
        r["observation"]["label"] != "invalid_attempt" for r in rows)


def test_the_filter_sees_a_noisy_measurement_not_the_simulators_answer(run):
    """The registered sensor model is applied before the PF, once."""
    _d, _col, rows = run
    hits = [r for r in rows if r["observation"]["label"] == "contact"]
    assert hits
    moved = 0
    for r in hits:
        o = r["observation"]
        assert "p_noisy_body" in o
        if not np.allclose(o["p_noisy_body"], o["contact_point_body"],
                           atol=1e-9):
            moved += 1
    assert moved > 0.8 * len(hits), "the measurement reached the PF unnoised"


def test_no_ground_truth_reaches_the_filter(run):
    """Section 11 item 9.  The belief mean is never the true pose -- a filter
    handed truth would sit on it."""
    _d, _col, rows = run
    s = diagnostics.belief_report(rows)
    assert s["gt_leak"]["ok"]
    assert s["gt_leak"]["belief_to_gt_m"]["min"] > 1e-3
    assert s["chain"]["ok"] and s["one_step"]["ok"]


def test_the_belief_update_path_cannot_read_the_simulator():
    """A source guard on the module that owns `b`: it may import the filter
    and the observation types, and nothing that knows where the robot is."""
    import ast
    src = (Path(__file__).resolve().parents[2] / "src" / "data_collect"
           / "belief_segment.py")
    read = src.read_text
    tree = ast.parse(read())
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, ast.Name):
            names.add(n.id)
    for leak in ("truth_pose", "root_link_pos_w", "heading_w",
                 "map_point_to_world"):
        assert leak not in names, leak
    # Ground truth may be named ONLY by the functions that place `mu_0`.
    # Pinning literal lines instead was measured breaking on a refactor that
    # changed nothing about the leak, which makes the guard noise rather than
    # protection.
    placers = {"__init__", "_single", "_mixture"}
    src = read()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        body = ast.get_source_segment(src, fn) or ""
        names = {n.id for n in ast.walk(ast.parse(body.lstrip()))
                 if isinstance(n, ast.Name)}
        touches_gt = bool({"gt", "gt_pose_map"} & names)
        if fn.name in placers:
            continue
        assert not touches_gt, f"{fn.name} names the ground-truth pose"
    for name in ("update", "predict", "snapshot", "_build"):
        fn = next((f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
                   and f.name == name), None)
        if fn is None:
            continue
        body = ast.get_source_segment(read(), fn) or ""
        assert "gt" not in body, f"{name} mentions the ground truth"


def test_the_leak_test_survives_a_lucky_draw_but_not_a_real_leak():
    """A filter handed truth sits on it SYSTEMATICALLY.

    Testing that no single row is within a millimetre is a coincidence
    detector: over 12 000 real transitions, 5 rows land under 1 mm against a
    median of 97 mm, because a belief that drew a lucky `mu_0` and then kept
    being told it was right stays there.  The test has to look at the
    distribution."""
    from data_collect import diagnostics as D

    def rows_at(dists, n_updates=1):
        out = []
        for i, d in enumerate(dists):
            b = {"mean": [d, 0.0, 0.0], "cov": [[1e-3, 0, 0], [0, 1e-3, 0],
                                                [0, 0, 1e-3]],
                 "ess": 100.0, "n_particles": 100, "n_updates": 0,
                 "particles_row": 2 * i}
            a = {**b, "n_updates": 1, "particles_row": 2 * i + 1}
            out.append({"uid": f"u{i}", "segment": i, "step": 0,
                        "belief_before": b, "belief_after": a,
                        "segment_meta": {"belief_init": {}},
                        "observation": {"label": "contact", "pf_update": {}},
                        "state_before": {"base_pose_map": [0.0, 0.0, 0.0]}})
        return out

    # a real run: a smooth tail with a few coincidences
    good = rows_at([0.0002, 0.0009] + [0.02 * (i + 1) for i in range(300)])
    assert D.belief_report(good)["gt_leak"]["ok"]

    # a leak: the belief is ON the truth, every time
    bad = rows_at([1e-9] * 300)
    r = D.belief_report(bad)["gt_leak"]
    assert not r["ok"]
    assert r["within_1mm"] == 300 and r["median_m"] < D.GT_LEAK_MEDIAN_M


def test_the_leak_test_survives_a_lucky_draw_but_not_a_real_leak():
    """A filter handed truth sits on it SYSTEMATICALLY.

    Testing that no single row is within a millimetre is a coincidence
    detector, not a leak detector: over 12 000 real transitions, 5 rows land
    under 1 mm and 146 under 10 mm against a median of 97 mm.  That is a
    smooth tail, and it is the filter converging -- the closest of them began
    its segment 2.6 mm out on a lucky `mu_0` draw and the touches held it
    there.  A leak puts a spike at exactly zero."""
    from data_collect import diagnostics as D

    def rows_at(dists):
        out = []
        for i, dist in enumerate(dists):
            b = {"mean": [dist, 0.0, 0.0],
                 "cov": [[1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1e-3]],
                 "ess": 100.0, "n_particles": 100, "n_updates": 0,
                 "particles_row": 2 * i}
            out.append({"uid": f"u{i}", "segment": i, "step": 0,
                        "belief_before": b,
                        "belief_after": {**b, "n_updates": 1,
                                         "particles_row": 2 * i + 1},
                        "segment_meta": {"belief_init": {}},
                        "observation": {"label": "contact", "pf_update": {}},
                        "state_before": {"base_pose_map": [0.0, 0.0, 0.0]}})
        return out

    good = rows_at([0.0002, 0.0009] + [0.02 * (i + 1) for i in range(300)])
    assert D.belief_report(good)["gt_leak"]["ok"]

    bad = rows_at([1e-9] * 300)
    r = D.belief_report(bad)["gt_leak"]
    assert not r["ok"]
    assert r["within_1mm"] == 300 and r["median_m"] < D.GT_LEAK_MEDIAN_M


def test_a_mixture_component_outside_the_mapped_world_is_rejected():
    """Clearance alone called a hypothesis 4.8 m past the far wall plausible.

    The 20k run's segment 130 opened with a component at y = 8.66 m in a room
    whose north wall is at y = 3.85 m.  It was clear of every body by metres,
    so the obstacle test passed it, and the belief the model would have been
    trained on put a third of its mass outside the world.
    """
    from data_collect.belief_segment import (BeliefSegment,
                                             MIXTURE_ENVELOPE_M)
    from data_collect.rooms import build_scene

    scene = build_scene("room2", False)
    seg = BeliefSegment.__new__(BeliefSegment)
    seg.scene = scene
    lo, hi = seg._envelope()

    assert not seg._plausible((1.776588, 8.657884))     # the real one
    assert seg._plausible((1.0, 3.0))                   # mid-room
    assert not seg._plausible((-0.6, 2.0))              # inside wall_0
    # the envelope is the mapped extent, not a guess
    verts = np.concatenate([np.asarray(b.verts, float)
                            for b in scene.plan.bodies])
    assert np.allclose(lo, verts.min(0) - MIXTURE_ENVELOPE_M)
    assert np.allclose(hi, verts.max(0) + MIXTURE_ENVELOPE_M)
