"""The planner-facing belief wrapper: gates, states, reseeding, route frame."""

import math

import numpy as np
import pytest
from conftest import make_belief

from global_nav.managed import BeliefState, ManagedBelief, ManagedConfig, route_arclength
from global_nav.observations import ContactObs
from global_nav.stressmaps import build_scenario


@pytest.fixture(scope="module")
def scn():
    return build_scenario("twin_junctions")


def managed(scn, **cfg_kw) -> ManagedBelief:
    b = make_belief(scn.plan, n=500, xy=(1.0, 0.0), sigma_xy_m=0.10, sigma_reg_m=0.05)
    return ManagedBelief(b, ManagedConfig(**cfg_kw))


# ── registration gate ────────────────────────────────────────────────────────────


def test_registration_gate_reports_zero_below_the_floor(scn):
    mb = managed(scn)
    mb.b.reg = 0.04 * np.ones((mb.b.n, 2)) / math.sqrt(2)   # |raw| = 0.04 < 0.10
    assert np.allclose(mb.registration_for_planning(), 0.0)


def test_registration_gate_opens_above_the_floor_and_holds_by_hysteresis(scn):
    mb = managed(scn)
    mb.b.reg = np.tile([[0.10, -0.08]], (mb.b.n, 1))        # |raw| = 0.128 > 0.10
    assert np.allclose(mb.registration_for_planning(), [0.10, -0.08])
    # estimate dips into the dead zone: an open gate does not chatter shut
    mb.b.reg = np.tile([[0.06, -0.05]], (mb.b.n, 1))        # |raw| = 0.078
    assert np.allclose(mb.registration_for_planning(), [0.06, -0.05])
    mb.b.reg = np.tile([[0.03, -0.02]], (mb.b.n, 1))        # below close threshold
    assert np.allclose(mb.registration_for_planning(), 0.0)


# ── belief states ────────────────────────────────────────────────────────────────


PHANTOM = ContactObs(p_body=(0.35, 0.02), signature="door_handle")
WALL = ContactObs(p_body=(0.0, 0.80), signature="surface", normal_body=(0.0, -1.0))


def test_consecutive_unexplained_contacts_walk_the_state_machine(scn):
    mb = managed(scn)
    assert mb.state is BeliefState.NORMAL
    mb.observe([PHANTOM])
    assert mb.state is BeliefState.NORMAL
    mb.observe([PHANTOM])
    assert mb.state is BeliefState.SUSPECT_LOST
    mb.observe([PHANTOM])
    assert mb.state is BeliefState.SUSPECT_LOST
    mb.observe([PHANTOM])
    assert mb.state is BeliefState.RELOCALIZE
    assert mb.n_reloc_triggered == 1


def test_one_well_explained_contact_clears_suspicion(scn):
    """SUSPECT is cheap and reversible: it was a phantom after all."""
    mb = managed(scn)
    mb.observe([PHANTOM])
    mb.observe([PHANTOM])
    assert mb.state is BeliefState.SUSPECT_LOST
    mb.observe([WALL])
    assert mb.state is BeliefState.NORMAL
    assert mb._low_streak == 0


def test_relocalize_reseeds_along_the_route_and_stops_rejecting(scn):
    mb = managed(scn)
    spread0 = mb.b.xy[:, 0].std()
    for _ in range(4):
        mb.observe([PHANTOM])
    assert mb.state is BeliefState.RELOCALIZE
    # the population now spans the route band, not the old tight cloud
    assert mb.b.xy[:, 0].std() > 3.0 * spread0
    # and the outlier gate is off: measurements are trusted over the belief
    assert mb.b.cfg.observation.outlier_min_support == 0.0


def test_recovery_exits_relocalize_and_restores_the_gate(scn):
    mb = managed(scn)
    for _ in range(4):
        mb.observe([PHANTOM])
    assert mb.state is BeliefState.RELOCALIZE
    # feed well-explained wall contacts; the reseeded cloud must converge and exit
    for _ in range(12):
        mb.observe([WALL])
        if mb.state is BeliefState.NORMAL:
            break
    assert mb.state is BeliefState.NORMAL
    assert mb.n_reloc_recovered == 1
    assert mb.b.cfg.observation.outlier_min_support > 0.0


# ── route frame ──────────────────────────────────────────────────────────────────


def test_route_arclength_extends_past_corners(scn):
    """A hypothesis PAST the route's turn must keep its arclength -- clamping at
    the corner collapsed a 3.0 m mode separation to 0.03 m and opened the branch
    gate on a phantom singleton (measured; the wrong turn followed)."""
    rf = route_arclength(scn.plan)
    s = lambda p: float(rf.project(np.array([p]))[0][0])
    assert s((2.6, 0.0)) == pytest.approx(2.6, abs=0.01)
    assert s((8.5, 0.0)) == pytest.approx(8.5, abs=0.01)     # past the turn at 7
    assert s((11.5, 0.0)) == pytest.approx(11.5, abs=0.01)   # far overshoot
    # separation preserved exactly where the twin ambiguity lives
    assert s((8.6, 0.0)) - s((5.6, 0.0)) == pytest.approx(3.0, abs=0.02)


def test_hypotheses_merge_at_route_scale_but_keep_real_modes(scn):
    mb = managed(scn)
    n = mb.b.n
    half = n // 2
    # two REAL hypotheses 3 m apart plus a sampling-jitter split inside one
    mb.b.xy = np.concatenate([
        np.array([2.6, 0.0]) + 0.05 * mb.b.rng.standard_normal((half - 60, 2)),
        np.array([2.95, 0.0]) + 0.05 * mb.b.rng.standard_normal((60, 2)),
        np.array([5.6, 0.0]) + 0.05 * mb.b.rng.standard_normal((n - half, 2))])
    mb.b.logw = np.zeros(n)
    hyps = mb.hypotheses()
    assert len(hyps) == 2, "the 0.35 m split is jitter, the 3.0 m split is real"
    assert abs(hyps[0]["s"] - hyps[1]["s"]) == pytest.approx(3.0, abs=0.35)


def test_route_s_within_measures_the_mode_not_the_separation(scn):
    mb = managed(scn)
    n = mb.b.n
    half = n // 2
    mb.b.xy = np.concatenate([
        np.array([2.6, 0.0]) + 0.10 * mb.b.rng.standard_normal((half, 2)),
        np.array([5.6, 0.0]) + 0.10 * mb.b.rng.standard_normal((n - half, 2))])
    mb.b.logw = np.zeros(n)
    total = mb.route_s()[1]
    within = mb.route_s_within()
    assert total > 1.0, "total spread is dominated by the 3 m separation"
    assert within < 0.2, "within-mode spread is the mode's own"


def test_reaching_for_a_mapped_wall_and_finding_nothing_counts_as_lost(scn):
    """The health monitor only stepped on CONTACTS, so a run whose every
    press MISSED never advanced the streak.  Measured on the reverse leg's
    second wall: the belief sat 0.39 m off the wall while truly 1.19 m out --
    six times the cloud's own sigma, so no reweighting could reach the truth
    (there were no particles there) -- and the state stayed NORMAL through
    six consecutive misses.  A miss where the map promised a wall is the
    strongest pose evidence there is: no pose the cloud holds explains it."""
    b = managed(scn)
    assert b.state is BeliefState.NORMAL
    seen = [b.note_unexplained_station() for _ in range(8)]
    assert BeliefState.SUSPECT_LOST.value in seen
    assert b.state is BeliefState.RELOCALIZE, seen
