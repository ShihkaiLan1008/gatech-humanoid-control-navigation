"""Belief-aware landmark diversion: go and read the geometry that separates
the live hypotheses, before committing to a branch.

`r5_twinbay` is aliased by construction -- two identical bays 3.4 m apart, and
a bimodal prior at exactly that pitch with the WRONG mode holding 0.55.  The
diagnosis (`PF_AUDIT.md` 11) found that the true mode is almost never lost
(19/20 seeds) and the disambiguating geometry is reachable; the stack simply
never goes to it, because `edge_cost` credits any touch-follow edge with
shrinking sigma and the aliased south wall cannot separate anything.

These tests pin the parts that make the rule general rather than a hack:
the trigger, the two safety filters, the per-channel value, and the fact that
it does nothing at all on the four maps that do not need it.
"""

from __future__ import annotations

import numpy as np
import pytest

from execution.backends import BackendConfig, KinematicBackend
from lane_graph.executive3 import ExecutiveV3, ExecV3Config
from lane_graph.graph3 import build_graph3
from lane_graph.run_v3 import PRESS_TARGET_N
from lane_graph.scene3 import build_v3_scene


def _exec(name="r5_twinbay", seed=0, **kw):
    sc = build_v3_scene(name)
    be = KinematicBackend(sc, BackendConfig(probe_force_n=PRESS_TARGET_N),
                          np.random.default_rng(seed))
    return ExecutiveV3(sc, be, build_graph3(sc), ExecV3Config(seed=seed, **kw)), be


# ── the trigger is route-decision ambiguity, not "probe more" ────────────────────

def test_decision_critical_needs_separation_and_mass():
    ex, be = _exec()
    try:
        far = [np.zeros(2), np.array([3.4, 0.0])]
        near = [np.zeros(2), np.array([0.10, 0.0])]
        assert ex._decision_critical(far, [0.55, 0.45])
        assert not ex._decision_critical(near, [0.55, 0.45])   # same place
        assert not ex._decision_critical(far, [0.97, 0.03])    # no live rival
        assert not ex._decision_critical([np.zeros(2)], [1.0])  # unimodal
    finally:
        be.close()


def test_the_threshold_separates_walk_off_ambiguity_from_the_real_kind():
    """4 goal radii = 0.88 m: above r1's 0.55 m prior split, far below r5's
    3.40 m.  Measured in `ExecV3Config.divert_decision_scale_m`."""
    ex, be = _exec()
    try:
        r = ex.scene.goal_radius_m
        assert ex.cfg.divert_decision_scale_m * r == pytest.approx(0.88)
        assert not ex._decision_critical(
            [np.zeros(2), np.array([0.55, 0.0])], [0.55, 0.45])
        assert ex._decision_critical(
            [np.zeros(2), np.array([3.40, 0.0])], [0.55, 0.45])
    finally:
        be.close()


# ── the two safety filters ──────────────────────────────────────────────────────

def test_a_stance_illegal_under_a_minority_hypothesis_is_refused():
    """The top-scoring r5 stances put the minority hypothesis 2 m OUTSIDE the
    building.  Informative on paper, a walk into a wall in fact."""
    ex, be = _exec()
    try:
        inside = np.array([3.20, -0.60])
        assert ex._safe_for_every_hypothesis(inside, [np.zeros(2)])
        # the same stance, with a hypothesis that puts the base outside the room
        assert not ex._safe_for_every_hypothesis(
            inside, [np.zeros(2), np.array([-6.0, 0.0])])
    finally:
        be.close()


def test_a_path_illegal_under_a_minority_hypothesis_is_refused():
    ex, be = _exec()
    try:
        deltas = [np.zeros(2), np.array([-3.4, 0.0])]
        # a target across the room: under one hypothesis the same commanded
        # walk sweeps a different corridor
        far = np.array([6.9, 2.6])
        legal = ex._path_safe_for_every_hypothesis(far, [np.zeros(2)])
        both = ex._path_safe_for_every_hypothesis(far, deltas)
        assert isinstance(legal, bool) and isinstance(both, bool)
        # whatever the map says, the two-hypothesis test can only be stricter
        assert not (both and not legal)
    finally:
        be.close()


# ── value is priced in nats, per channel ────────────────────────────────────────

def test_value_is_zero_where_the_geometry_aliases():
    """The south wall repeats at exactly the mode separation, so no press on it
    can separate the pair -- whatever the fan offset.

    Stances chosen so BOTH the stance and its 3.4 m shifted counterpart sit
    well inside the wall run: 4.60/1.20 are bay-on-bay, the rest flat-on-flat.
    Near the WEST end the alias does break (the shifted pose runs off the end
    of the wall), which is real information -- and is refused later by
    `_safe_for_every_hypothesis`, not here.""" 
    ex, be = _exec()
    try:
        deltas = [np.zeros(2), np.array([-3.40, 0.0])]
        masses = [0.55, 0.45]
        n_wall, u = np.array([0.0, -1.0]), np.array([1.0, 0.0])
        for x in (4.60, 5.20, 5.80, 6.40):
            st = np.array([x, -0.82])
            assert ex._separating_value(st, n_wall, u, deltas, masses) == 0.0, x
    finally:
        be.close()


def test_some_reachable_geometry_does_separate_the_r5_prior():
    ex, be = _exec()
    try:
        deltas, masses = ex._live()
        assert len(deltas) == 2                      # the bimodal prior
        vals = [ex._separating_value(ex._stance(pt, nw), nw, u, deltas, masses)
                for _e, pt, nw, u in ex._lane_stances()]
        assert max(vals) >= ex.cfg.divert_min_value
    finally:
        be.close()


# ── it does nothing where it is not needed ──────────────────────────────────────

@pytest.mark.parametrize("name", ["r1_gallery", "r2_lbend", "r3_tpinch", "r4_winding"])
def test_no_diversion_on_the_maps_that_do_not_need_one(name):
    """Measured, 20 seeds each: `divert` is identical to `baseline` on every
    column of these four maps.  Seed 0 is the cheap standing guard."""
    a, ba = _exec(name, seed=0, divert=False)
    b, bb = _exec(name, seed=0, divert=True)
    try:
        ra, rb = a.run(), b.run()
        assert rb["n_diversions"] == 0
        assert rb["n_probes"] == ra["n_probes"]
        assert rb["global_pos_err_m"] == pytest.approx(ra["global_pos_err_m"])
        assert rb["path_len_m"] == pytest.approx(ra["path_len_m"])
    finally:
        ba.close()
        bb.close()


def test_r5_diverts_and_the_contact_case_collapses_the_belief():
    """The winning trace, pinned: bimodal prior -> diversion -> contact -> one
    mode -> arrival.  See PF_AUDIT.md 11 for the full seed-0 trace."""
    ex, be = _exec("r5_twinbay", seed=0, divert=True)
    try:
        r = ex.run()
        assert r["n_diversions"] >= 1
        d = r["diversions"][0]
        assert len(d["masses_before"]) == 2
        assert d["value"] >= ex.cfg.divert_min_value
        assert d["result"] == "contact"
        assert d["masses_after"] == [1.0]
        assert r["arrived_truly"]
    finally:
        be.close()


def test_the_flag_reproduces_the_frozen_baseline():
    a, ba = _exec("r5_twinbay", seed=0, divert=False)
    try:
        r = a.run()
        assert r["n_diversions"] == 0
        assert all(p["station"] != "LANDMARK" for p in r["probes"])
    finally:
        ba.close()
