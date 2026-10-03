"""The six validations, end to end.

These run the benchmark, so they are slower than ``test_belief.py`` and they assert
looser things -- what must be true of the whole pipeline rather than of one
mechanism.  Thresholds are set from the measured 5-seed spread with room to spare;
they are regression guards, not the results themselves.  The results are in
``GLOBAL_TACNAV_BELIEF_V0_REPORT.md`` and are reproduced by ``run_benchmark``.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from global_nav.benchmark import (
    build_scenarios,
    command_sequence,
    run_bad_association,
    run_scenario,
)
from global_nav.floorplan import benchmark_floorplan

N_PARTICLES = 600


@pytest.fixture(scope="module")
def scns():
    return build_scenarios(n_particles=N_PARTICLES, seed=0)


def _run(scns, name, **over):
    scn = scns[name]
    return run_scenario(replace(scn, **over) if over else scn)


# ── 1. odometry-only drift ───────────────────────────────────────────────────────


def test_odometry_only_drifts_and_says_so(scns):
    """Two claims, and the second is the one that matters.

    The belief must drift -- there is nothing correcting it.  But it must also
    report uncertainty commensurate with the error it is actually making.  A filter
    that drifts and knows it is usable; one that drifts and reports a tight
    covariance is worse than no filter, because a planner will act on it.
    """
    r = _run(scns, "odometry_only")
    tr = r["trace"]
    err = np.array([row["position_error_m"] for row in tr])
    std = np.array([np.linalg.norm(row["pos_std_m"]) for row in tr])

    assert err[-1] > 3.0 * err[0], "with no evidence the error must grow"
    assert std[-1] > 2.5 * std[0], "and the reported uncertainty must grow with it"
    # monotone in expectation only: the spread is a statistic of a finite sample, so
    # single steps jitter.  What must not happen is a real contraction.
    assert float(np.diff(std).min()) > -0.02
    assert r["summary"]["n_resamples"] == 0, "nothing to resample on: no weights change"


def test_the_drifting_belief_is_calibrated_not_merely_wide(scns):
    """Calibration is a statement about a DISTRIBUTION of runs, not one run.

    ``pos_std_m`` is a per-axis spread, so its norm is the RMS radius: a single
    realisation exceeds it about 37% of the time by construction, and asserting
    otherwise on one seed tests nothing.  The real claim is that the 3-sigma
    envelope almost always holds and the belief is not systematically over-wide.
    """
    ratios = []
    for seed in (0, 1, 2, 3):
        tr = run_scenario(build_scenarios(N_PARTICLES, seed)["odometry_only"])["trace"]
        for row in tr:
            std = float(np.linalg.norm(row["pos_std_m"]))
            if std > 1e-6:
                ratios.append(row["position_error_m"] / std)
    ratios = np.array(ratios)
    assert float(np.mean(ratios <= 3.0)) > 0.95, "3-sigma envelope must hold"
    assert float(np.median(ratios)) < 1.5, "and it must not be doing so by being vacuous"


# ── 2. contact-only update ───────────────────────────────────────────────────────


def test_contacts_re_anchor_the_pose(scns):
    odom = _run(scns, "odometry_only")["summary"]
    contact = _run(scns, "contact_only")["summary"]
    assert contact["final_position_error_m"] < 0.4 * odom["final_position_error_m"]
    assert contact["final_pos_std_m"] < 0.4 * odom["final_pos_std_m"]


def test_the_map_prior_alone_buys_essentially_nothing(scns):
    """The control that makes the other ablations attributable.

    Knowing you are inside a corridor, without touching it, does not tell you where
    along it you are -- so whatever contact_only beats THIS by is bought by touch,
    not by the map.
    """
    odom = _run(scns, "odometry_only")["summary"]
    structural = _run(scns, "structural_only")["summary"]
    assert structural["final_position_error_m"] > 0.5 * odom["final_position_error_m"]


# ── 3. swept-free-only update ────────────────────────────────────────────────────


def test_negative_information_alone_localizes(scns):
    """Touching nothing is a measurement.  A run that never makes contact still has
    to beat dead reckoning, or the swept-free channel is decoration."""
    odom = _run(scns, "odometry_only")["summary"]
    free = _run(scns, "swept_free_only")["summary"]
    assert free["mean_position_error_m"] < 0.75 * odom["mean_position_error_m"]
    assert free["final_pos_std_m"] < 0.6 * odom["final_pos_std_m"]


# ── 4. combined ──────────────────────────────────────────────────────────────────


def test_every_tactile_channel_beats_dead_reckoning(scns):
    """The effect this test can see, at the size it can see it.

    Each tactile channel cuts the mean position error to roughly a third of dead
    reckoning -- four standard errors, visible at four seeds and 600 particles.
    The gaps BETWEEN the tactile channels are an order of magnitude smaller
    (~0.011 m) and take a per-seed PAIRED comparison at 12 seeds to resolve at all;
    that measurement lives in the report, not here.  So this asserts only that
    combining does not hurt, which is what a cheap regression guard can honestly
    check.
    """
    names = ("odometry_only", "contact_only", "swept_free_only", "combined")
    mean_err = {n: [] for n in names}
    for seed in (0, 1, 2, 3):
        built = build_scenarios(n_particles=N_PARTICLES, seed=seed)
        for n in names:
            mean_err[n].append(run_scenario(built[n])["summary"]["mean_position_error_m"])
    avg = {n: float(np.mean(v)) for n, v in mean_err.items()}

    for tactile in ("contact_only", "swept_free_only", "combined"):
        assert avg[tactile] < 0.55 * avg["odometry_only"], tactile
    assert avg["combined"] < 1.5 * min(avg["contact_only"], avg["swept_free_only"])


def test_the_combined_belief_stays_calibrated(scns):
    """Sharp is only useful if it is also honest: the reported spread must still
    bracket the actual error most of the time."""
    tr = _run(scns, "combined")["trace"]
    err = np.array([row["position_error_m"] for row in tr])
    std = np.array([np.linalg.norm(row["pos_std_m"]) for row in tr])
    assert float(np.mean(err <= 3.0 * std)) > 0.85


# ── 5. ambiguous landmarks ───────────────────────────────────────────────────────


def test_a_fifty_fifty_prior_over_identical_corners_is_carried_not_guessed(scns):
    """Given two identical wall termini and no evidence to separate them, the belief
    must stay in both places.  Collapsing early is not decisiveness -- it is a
    coin-flip reported as a fact, and the mean it produces is 1.5 m from truth."""
    r = _run(scns, "ambiguous_side_only")
    tr = [row for row in r["trace"] if row["station"] is not None]
    bimodal = [row for row in tr if row["n_modes"] >= 2]
    assert len(bimodal) >= 2, "the ambiguity must survive more than one station"

    first = bimodal[0]
    weights = sorted((m["weight"] for m in first["modes"]), reverse=True)[:2]
    assert weights[1] > 0.20, "neither hypothesis may be discarded"
    assert weights[0] + weights[1] > 0.90, "and the mass must be in those two places"

    # while bimodal the mean sits between the modes, where no particle lives
    assert first["position_error_m"] > 1.0
    # and it does resolve once something distinguishes them
    assert tr[-1]["n_modes"] == 1
    assert tr[-1]["position_error_m"] < 0.25


def test_the_fifty_fifty_split_is_preserved_and_tightens_with_particle_count():
    """The 20% floor above is Monte Carlo slack, not the property.

    With 600 particles the split lands near 2:1 from sampling noise alone; the
    symmetric prior is only genuinely preserved once there are enough particles to
    represent it.  If this ever fails while the looser test passes, the filter has
    a real asymmetry rather than a small sample.
    """
    r = run_scenario(build_scenarios(n_particles=1600, seed=0)["ambiguous_side_only"])
    bimodal = [row for row in r["trace"]
               if row["station"] is not None and row["n_modes"] >= 2]
    w = sorted((m["weight"] for m in bimodal[0]["modes"]), reverse=True)[:2]
    assert 0.35 < w[0] < 0.65
    assert w[1] > 0.35


def test_the_forward_probe_is_what_resolves_the_ambiguity(scns):
    """Same ambiguity, one fan withheld.  If dropping the forward probe does not
    lengthen the ambiguity, then something else was resolving it and the causal
    story in the report is wrong."""
    with_fwd = _run(scns, "ambiguous_landmarks")["trace"]
    without = _run(scns, "ambiguous_side_only")["trace"]
    n_with = sum(1 for r in with_fwd if r["station"] is not None and r["n_modes"] >= 2)
    n_without = sum(1 for r in without if r["station"] is not None and r["n_modes"] >= 2)
    assert n_without > n_with


def test_a_bimodal_belief_is_flagged_as_bimodal(scns):
    """The covariance cannot say this, so something else has to."""
    for row in _run(scns, "ambiguous_side_only")["trace"]:
        spread = float(np.linalg.norm(row["pos_std_m"]))
        if row["position_error_m"] > 1.0 and spread > 0.5:
            assert row["n_modes"] >= 2
            return
    pytest.fail("expected at least one wide, badly-centred step to be flagged multi-modal")


# ── 6. incorrect association rejection ───────────────────────────────────────────


def test_phantom_contacts_are_rejected_every_time(scns):
    r = run_bad_association(scns["bad_association"])
    s = r["summary"]
    assert s["n_injected_phantoms"] >= 2
    assert s["phantom_rejection_rate"] == 1.0


def test_rejecting_phantoms_leaves_the_belief_as_good_as_if_they_never_came(scns):
    """The point of the gate: a run under attack must match the clean run.

    Measured over 6 seeds at 1500 particles the two differ by at most 0.017 m; the
    0.03 m here is that with slack for the lower particle count.
    """
    clean = _run(scns, "combined", route_end_m=6.0)["summary"]
    attacked = run_bad_association(scns["bad_association"])["summary"]
    assert attacked["final_position_error_m"] == pytest.approx(
        clean["final_position_error_m"], abs=0.03)


def test_an_ambiguous_association_is_not_treated_as_a_bad_one(scns):
    """Rejection must fire on impossible measurements, not merely ambiguous ones --
    otherwise the gate quietly deletes the informative corner touches."""
    for name in ("combined", "contact_only", "ambiguous_side_only"):
        assert _run(scns, name)["summary"]["n_outliers"] == 0, name


# ── the scripted walk itself ─────────────────────────────────────────────────────


def test_the_command_sequence_walks_the_route_and_turns_at_the_corners():
    plan = benchmark_floorplan()
    cmds = list(command_sequence(plan, 0.25))
    turns = [c for c in cmds if abs(c[1]) > 1e-9]
    assert len(turns) == 2, "the route has exactly two corners"
    assert [round(math.degrees(t[1])) for t in turns] == [90, -90], "left then right"
    walked = sum(float(np.linalg.norm(c[0])) for c in cmds)
    assert walked == pytest.approx(plan.route_length_m, abs=1e-6)


def test_the_benchmark_is_deterministic(scns):
    a = _run(scns, "combined")["summary"]
    b = _run(scns, "combined")["summary"]
    assert a == b


def test_every_scenario_runs_and_reports_the_required_outputs(scns):
    required = {"pose_mean", "pose_cov", "edge_probabilities", "landmark_probabilities",
                "registration_mean_m", "registration_error_m", "wrong_turn_probability",
                "ess", "n_modes", "modes"}
    for name in scns:
        r = run_bad_association(scns[name]) if name == "bad_association" else _run(scns, name)
        assert required <= set(r["trace"][-1]), f"{name} is missing outputs"
        assert 0.0 <= r["trace"][-1]["wrong_turn_probability"] <= 1.0
        assert np.all(np.isfinite(np.asarray(r["trace"][-1]["pose_cov"], float)))
