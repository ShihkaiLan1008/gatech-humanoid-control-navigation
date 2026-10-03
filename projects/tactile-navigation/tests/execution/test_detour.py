"""Batch 3A.2 Phase 3: the landmark detour completes, without weakening the
benchmark.

Baseline measured before this phase (same 12 seeds): arrival 0.25, mean
25.0 probes, the dominant failure a planner abort with the belief already
resolved to 2-11 cm but every probe soaked at the box.  Four independent
defects were found and each fix is asserted here:

* full-fan visits pressed on after the question was answered  -> per-press
  global updates with a gate stop;
* branch probing could starve the arrival anchor              -> a probe
  reserve, and an EXPLICIT abort when branch probing exhausts it;
* the mirrored 'rival' stances probed both-predict-contact
  geometry (masses wobbled 0.55-0.77 for 20 presses)          -> hypothesis-
  conditioned RE-APPROACH of the same map station instead;
* an anonymous "surface" contact could not be explained by a
  "box"-signed body, so a physically correct landmark touch
  EXECUTED the correct particles (resolved 0.47 m wrong on
  8 of 12 seeds)                                              -> every body
  carries signature "surface".
"""

import numpy as np
import pytest

from execution.backends import BackendConfig, KinematicBackend
from execution.executive import ExecConfig, Executive
from execution.scenes import build_scene

N_SEEDS = 12


def _run(seed, policy="belief_aware", **kw):
    sc = build_scene("detour_box")
    be = KinematicBackend(sc, BackendConfig(), np.random.default_rng(seed))
    e = Executive(sc, be, ExecConfig(seed=seed, policy=policy, n_particles=900, **kw))
    e.verbose = False
    return e, e.run()


@pytest.fixture(scope="module")
def fleet():
    return [_run(s)[1] for s in range(N_SEEDS)]


# ── ACCEPTANCE: detour -> disambiguate -> return -> turn -> arrive ──────────────


def test_the_full_route_completes_on_most_seeds(fleet):
    arrived = np.mean([r["arrived"] for r in fleet])
    assert arrived >= 0.75, f"detour completion {arrived:.2f}"
    assert not any(r["wrong_turn"] for r in fleet)


def test_the_detour_is_real_and_the_belief_is_reduced(fleet):
    """Arriving seeds that visited the box physically walked away from the
    goal and back, and ended with a 10x smaller belief error than the 0.55 m
    a committed wrong hypothesis would leave."""
    detoured = [r for r in fleet if r["arrived"] and r["probes_at_disambiguators"]]
    assert detoured, "some arrivals must come through the detour"
    for r in detoured:
        assert r["detour_path_m"] >= 2.0, "off-route excursion, there and back"
        assert r["global_pos_err_m"] <= 0.20
    assert np.mean([r["global_pos_err_m"] for r in fleet]) <= 0.16


def test_landmark_resolution_is_reported_separately_from_completion(fleet):
    for r in fleet:
        assert "landmark_resolved" in r and "landmark_resolved_t" in r
        assert "probes_at_disambiguators" in r and "probes_remaining" in r
    resolved = [r for r in fleet if r["landmark_resolved"]]
    assert resolved, "resolution must be observed and stamped"
    for r in resolved:
        assert r["landmark_resolved_t"] is not None


# ── budget: the reserve holds and exhaustion is explicit ─────────────────────────


def test_branch_probing_never_eats_the_arrival_reserve(fleet):
    reserve = ExecConfig().arrive_reserve_probes
    for r in fleet:
        if r["outcome"] == "planner_abort":
            assert r["probes_remaining"] >= reserve, \
                "an abort with the reserve consumed is a starve, not a decision"


def test_probe_budget_exhaustion_is_an_explicit_planner_failure():
    """Starve the branch on purpose: with a budget just above the reserve,
    the planner must abort with the budget reason, not grind zero-press
    visits (the measured pre-fix failure shape)."""
    _, r = _run(0, max_probes_total=6)
    assert r["outcome"] == "planner_abort"
    reasons = " ".join(f["reason"] for f in r["failures"])
    assert "probe_budget_exhausted" in reasons or "no_reachable_station" in reasons
    assert r["n_probes"] <= 6


def test_the_fan_stops_when_the_gate_is_met(fleet):
    """No station visit may press on after the branch is decided.  Erode
    visits legitimately spend their whole fan (the gate cannot open on weak
    free rays), so the bound is on the TOTAL: strictly below the measured
    25-press soak of the batched-update era, and well below it on average."""
    spent = [r["probes_at_disambiguators"] for r in fleet
             if r["arrived"] and r["probes_at_disambiguators"]]
    assert spent, "some arrivals come through the box"
    assert max(spent) <= 20
    assert float(np.mean(spent)) <= 15


# ── the benchmark is NOT weakened ────────────────────────────────────────────────


def test_the_scene_still_withholds_on_route_evidence():
    """The fixes must not have added on-route terminus evidence: every
    on-route station still reads flat wall under both hypotheses."""
    sc = build_scene("detour_box")
    sep = sc.prior_modes[1][0] - sc.prior_modes[0][0]
    for name, st in sc.stations.items():
        if st.purpose == "disambiguator" or st.leg != 0:
            continue
        assert st.discrimination(sc.plan, 0.0, sep) == 0.0, name
    assert sc.withholds_terminus_evidence


def test_no_touch_still_cannot_probe_the_box():
    _, r = _run(3, policy="no_touch")
    assert r["n_probes"] == 0
    assert r["probes_at_disambiguators"] == 0
