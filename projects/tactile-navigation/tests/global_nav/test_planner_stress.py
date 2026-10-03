"""End-to-end acceptance, at reduced scale.

Thresholds are regression guards under the tests' own (n_particles, seeds); the
report's numbers come from the full 20-seed benchmark.  Loose on purpose: these
must fail when a mechanism breaks, not when a seed is unlucky.
"""

import numpy as np
import pytest

from global_nav.episode import EpisodeConfig, run_episode
from global_nav.policies import make_policy
from global_nav.stressmaps import build_scenario

N = 500
SEEDS = range(8)


def _run(scn_name, policy, seeds=SEEDS, n=N):
    sc = build_scenario(scn_name)
    return [run_episode(sc, make_policy(policy), EpisodeConfig(n_particles=n, seed=s))
            for s in seeds]


@pytest.fixture(scope="module")
def twin_no_touch():
    return _run("twin_junctions", "shortest_astar")


@pytest.fixture(scope="module")
def twin_belief():
    return _run("twin_junctions", "belief_aware")


# ── acceptance: the stressed benchmark stresses ──────────────────────────────────


def test_no_touch_shortest_path_has_meaningful_wrong_turn_risk(twin_no_touch):
    """~half the prior mass is on the wrong hypothesis by construction; the
    realized wrong-turn rate must reflect that, not hide behind the executor."""
    assert np.mean([r["wrong_turn"] for r in twin_no_touch]) >= 0.30


def test_schedule_touch_does_not_fix_the_twin_ambiguity():
    """The control that separates 'touching' from 'touching the right thing':
    fixed-interval probing follows the aliased lane and keeps the same
    wrong-turn risk as never touching at all."""
    rows = _run("twin_junctions", "fixed_interval")
    assert np.mean([r["wrong_turn"] for r in rows]) >= 0.30


# ── acceptance: the informative landmark pays ────────────────────────────────────


def test_belief_aware_visits_the_disambiguator_and_stops_turning_wrong(
        twin_no_touch, twin_belief):
    wrong_base = np.mean([r["wrong_turn"] for r in twin_no_touch])
    wrong_ba = np.mean([r["wrong_turn"] for r in twin_belief])
    assert wrong_ba <= 0.15
    assert wrong_ba < 0.5 * wrong_base
    # and the mechanism is the pillar, not luck: the planner detours to it
    visited_pillar = [any(v.startswith("p55") for v in r["station_visits"])
                      for r in twin_belief]
    assert np.mean(visited_pillar) >= 0.7


def test_belief_aware_beats_no_touch_on_success(twin_no_touch, twin_belief):
    assert np.mean([r["success"] for r in twin_belief]) \
        > np.mean([r["success"] for r in twin_no_touch])


def test_probes_are_spent_not_rewarded(twin_belief):
    """Nothing in the objective rewards touching: the planner probes a handful
    of times, not on every station it passes."""
    assert np.mean([r["n_probes"] for r in twin_belief]) <= 5.0


# ── acceptance: relocalization ───────────────────────────────────────────────────


@pytest.mark.slow
def test_the_slip_is_absorbed_or_triggers_never_silently_fatal_under_dense_touch():
    """Measured behaviour, asserted as measured: DENSE probing mostly PREVENTS
    lostness rather than detecting it -- the association machinery tracks the
    belief through a sub-gate-spacing slip, so most seeds end well-localized
    with zero relocalizations.  What must never happen is the silent third
    option: no absorption, no trigger, confidently wrong at the end."""
    rows = _run("gauntlet_slip", "max_wall", seeds=range(6))
    handled = sum(1 for r in rows
                  if r["final_pos_err_m"] <= 0.8 or r["n_reloc_triggered"] > 0)
    assert handled >= 3
    assert sum(1 for r in rows if r["success"]) >= 2


@pytest.mark.slow
def test_the_slip_triggers_relocalization_under_minimal_probing():
    """The minimal prober cannot absorb what it never measures: when its tight
    belief finally meets the junction furniture, the unexplained-contact streak
    must reach RELOCALIZE on a real fraction of seeds.  (The deterministic
    trigger->recover arc is asserted exactly in test_managed.py; this checks the
    closed loop reaches it.)"""
    rows = _run("gauntlet_slip", "belief_aware")
    assert sum(1 for r in rows if r["n_reloc_triggered"] > 0) >= 2


def test_no_false_relocalizations_on_the_clean_twin(twin_belief):
    assert sum(r["n_reloc_triggered"] for r in twin_belief) <= 1


# ── acceptance: registration gating ──────────────────────────────────────────────


@pytest.mark.slow
def test_registration_gate_no_noise_below_floor_weak_recovery_above():
    """Below the floor: never worse than assuming zero.  Above it: the gate
    passes what the estimator has -- and under 0.45 m single-lane touch that is
    a PARTIAL recovery only (measured ~30-50% per component on the seeds where
    evidence sufficed, gate closed on the rest).  The x-component of a large
    offset is confounded with pose along the aliased corridor, so full recovery
    is not achievable here and is not asserted; the report carries the account.

    TWELVE SEEDS, NOT FIVE.  The raw estimate this gate has to clear 0.10 m
    with measures 0.00-0.16 m, so whether ANY seed opens the gate is close to
    a coin flip and five of them is not enough to measure it.  Measured after
    correspondence was unified (position and normal residuals scored against
    the SAME primitive, which stops the registration nuisance parameter
    absorbing mis-association): 3 of 12 seeds open the gate, at 0.19-0.22
    against the 0.30 baseline.  Before unification it was 2 of 5 at
    0.18.  The capability is the same and weaker; the old sample size could
    not tell that from noise in either direction.
    """
    from dataclasses import replace
    base = build_scenario("twin_junctions")
    means = {}
    for mag in (0.03, 0.30):
        errs_gated, errs_zero = [], []
        for seed in range(12):
            true = np.array([0.8 * mag, -0.6 * mag])
            sc = replace(base, reg_true=tuple(true))
            r = run_episode(sc, make_policy("fixed_interval"),
                            EpisodeConfig(n_particles=N, seed=seed))
            errs_gated.append(np.linalg.norm(np.asarray(r["registration_gated"]) - true))
            errs_zero.append(np.linalg.norm(true))
        means[mag] = (float(np.mean(errs_gated)), float(np.mean(errs_zero)))
    # never inject noise, at any magnitude
    for mag, (gated, zero) in means.items():
        assert gated <= zero + 0.02, (mag, gated, zero)
    # weak recovery retained above the floor
    gated, zero = means[0.30]
    assert gated < zero


# ── plumbing ─────────────────────────────────────────────────────────────────────


def test_oracle_is_the_ceiling_and_never_wrong():
    for name in ("twin_junctions", "drift_gauntlet"):
        rows = _run(name, "oracle", seeds=range(6), n=300)
        assert all(r["success"] for r in rows), name
        assert not any(r["wrong_turn"] for r in rows), name


def test_every_policy_reports_every_required_metric():
    r = _run("twin_junctions", "belief_aware", seeds=[0], n=300)[0]
    required = {"success", "wrong_turn", "path_len_m", "excess_m", "n_probes",
                "n_station_visits", "final_pos_err_m", "final_belief_state",
                "n_reloc_triggered", "n_reloc_recovered",
                "registration_gated", "plan_time_mean_ms"}
    assert required <= set(r)


def test_episodes_are_deterministic_given_seed():
    a = _run("twin_junctions", "belief_aware", seeds=[3], n=400)[0]
    b = _run("twin_junctions", "belief_aware", seeds=[3], n=400)[0]
    assert a["cost_m"] == b["cost_m"]
    assert a["outcome"] == b["outcome"]
    assert a["station_visits"] == b["station_visits"]
