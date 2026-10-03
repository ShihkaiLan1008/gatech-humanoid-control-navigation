"""Active landmark INSPECT: the second half of a rank-2 pair.

`PF_AUDIT.md` 1 measured why the belief drifts between touches on these maps:
a press on flat wall is rank 1 by the observation model's own construction, so
it re-measures the one component that was never in doubt.  The median
likelihood margin between the true pose and the filter's own answer was 0.00
nats on R2 and 0.03 on R3 -- the evidence did not distinguish them at all --
and the structural channel contributed 0.000 nats on every update of both
episodes.

INSPECT spends one extra press at a stance whose fan straddles a mapped
discontinuity, so the pair (contact, free) carries the along-wall coordinate
that neither half carries alone.  These tests pin the trigger, not the
aggregate: a behaviour that fires everywhere would just be "press twice".

Measured A/B, kinematic, 5 maps x 20 seeds -- see `scripts/bench_v3.py`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from execution.backends import BackendConfig, KinematicBackend
from lane_graph.executive3 import ExecutiveV3, ExecV3Config
from lane_graph.graph3 import build_graph3
from lane_graph.run_v3 import PRESS_TARGET_N
from lane_graph.scene3 import build_v3_scene


def _exec(name="r3_tpinch", seed=0, **kw):
    sc = build_v3_scene(name)
    be = KinematicBackend(sc, BackendConfig(probe_force_n=PRESS_TARGET_N),
                          np.random.default_rng(seed))
    return ExecutiveV3(sc, be, build_graph3(sc), ExecV3Config(seed=seed, **kw)), be


# ── the trigger ─────────────────────────────────────────────────────────────────

def test_flat_wall_offers_no_partner():
    """Where every fan offset predicts the same outcome there is nothing to
    learn, and INSPECT must not spend a probe."""
    ex, be = _exec()
    try:
        # a stance square onto a long flat wall: pick the lane point of the
        # first TOUCH_FOLLOW edge, well away from either end
        e = next(x for x in ex.g.edges if x.kind == "TOUCH_FOLLOW")
        i = len(e.poly) // 2
        lane_pt, n_wall = e.poly[i], e.normal[i]
        u = e.poly[min(i + 1, len(e.poly) - 1)] - e.poly[max(i - 1, 0)]
        u = u / max(float(np.linalg.norm(u)), 1e-9)
        stance = ex._stance(lane_pt, n_wall)
        hits = [ex._ray_hits(stance, n_wall, u, o) for o in ex.cfg.fan_offsets_m]
        if len(set(hits)) == 1:                       # genuinely flat here
            assert ex._inspect_partner(stance, n_wall, u, 0.0) is None
    finally:
        be.close()


def _bimodal(ex, separation_m=0.2):
    """Two live hypotheses either side of the tracked pose, and a map whose
    surface starts at the tracked frame's origin -- so the two disagree about
    where the edge is, which is what INSPECT is for."""
    ex._believed_xy = lambda: np.zeros(2)
    ex._hyps = lambda: [{"mass": 0.5, "xy": np.array([separation_m, 0.0])},
                        {"mass": 0.5, "xy": np.array([-separation_m, 0.0])}]
    ex._ray = lambda origin, direction, delta=(0.0, 0.0): (
        (0.2, np.array([0.0, -1.0]))
        if float(origin[0]) - float(np.asarray(delta, float)[0]) <= 0.0
        else (None, None))


def test_a_mapped_edge_inside_the_fan_offers_the_nearest_opposite():
    """Synthetic geometry, so the assertion is about the rule and not the map."""
    ex, be = _exec()
    try:
        n_wall = np.array([0.0, 1.0])
        u = np.array([1.0, 0.0])
        stance = np.array([0.0, 0.0])
        _bimodal(ex)
        # hit at offsets <= 0, miss at offsets > 0 (the TRACKED hypothesis)
        ex._ray_hits = lambda s, n, uu, off: float(off) <= 0.0
        ex._grazes = lambda s, n, uu, off: False
        assert ex._inspect_partner(stance, n_wall, u, 0.0) == pytest.approx(0.12)
        assert ex._inspect_partner(stance, n_wall, u, 0.24) == pytest.approx(0.0)
        assert ex._inspect_partner(stance, n_wall, u, -0.24) == pytest.approx(0.12)
    finally:
        be.close()


def test_a_single_hypothesis_never_inspects():
    """The measured trigger: reading an edge the belief already agrees about is
    a second press, not information.  See `_inspect_partner`."""
    ex, be = _exec()
    try:
        n_wall, u, stance = np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.zeros(2)
        _bimodal(ex)
        ex._ray_hits = lambda s, n, uu, off: float(off) <= 0.0
        ex._grazes = lambda s, n, uu, off: False
        assert ex._inspect_partner(stance, n_wall, u, 0.0) is not None
        ex._hyps = lambda: [{"mass": 1.0, "xy": np.zeros(2)}]
        assert ex._inspect_partner(stance, n_wall, u, 0.0) is None
    finally:
        be.close()


def test_a_grazing_partner_is_refused():
    """A grazing HIT wrist-grinds; the whole point of the fan chooser is to
    refuse those, and INSPECT does not get to bypass it."""
    ex, be = _exec()
    try:
        n_wall, u, stance = np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.zeros(2)
        _bimodal(ex)
        ex._ray_hits = lambda s, n, uu, off: float(off) <= 0.0
        ex._grazes = lambda s, n, uu, off: float(off) <= 0.0     # every hit grazes
        # pressing a miss, so the partner would have to be a (grazing) hit
        assert ex._inspect_partner(stance, n_wall, u, 0.12) is None
        # pressing a hit, so the partner is a miss -- a miss cannot graze
        assert ex._inspect_partner(stance, n_wall, u, 0.0) == pytest.approx(0.12)
    finally:
        be.close()


def test_a_confident_belief_does_not_inspect():
    ex, be = _exec()
    try:
        from lane_graph.graph3 import Belief3
        ex._bel = lambda: Belief3(sigma=0.01, minor=0.0)
        assert not ex._needs_along_wall_info()
        ex._bel = lambda: Belief3(sigma=0.40, minor=0.0)
        assert ex._needs_along_wall_info()
        ex._bel = lambda: Belief3(sigma=0.01, minor=0.50)
        assert ex._needs_along_wall_info()          # bimodal is also unresolved
    finally:
        be.close()


# ── the budget, and the flag ────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["r2_lbend", "r3_tpinch", "r4_winding"])
def test_inspect_respects_its_ceiling(name):
    ex, be = _exec(name, seed=0, inspect=True, inspect_max=1)
    try:
        r = ex.run()
        assert r["n_inspects"] <= 1
    finally:
        be.close()


@pytest.mark.parametrize("name", ["r2_lbend", "r3_tpinch"])
def test_the_flag_defaults_to_the_shipped_cadence(name):
    ex, be = _exec(name, seed=0)                     # inspect defaults OFF
    try:
        r = ex.run()
        assert r["n_inspects"] == 0
        assert all(p["station"] != "INSPECT" for p in r["probes"])
    finally:
        be.close()


def test_inspect_fires_and_is_logged_as_its_own_station():
    # divert OFF so this isolates INSPECT: a diversion changes the trajectory
    # and therefore which stances the fan ever straddles.
    ex, be = _exec("r5_twinbay", seed=1, inspect=True, divert=False,
                   skip_uninformative_probes=False)
    try:
        r = ex.run()
        assert r["n_inspects"] > 0
        assert any(p["station"] == "INSPECT" for p in r["probes"])
        # the diagnostic PF_AUDIT.md 5 needed and the probe log never carried
        assert all("reason" in p and "duration_s" in p for p in r["probes"])
    finally:
        be.close()
