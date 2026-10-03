"""The mechanisms, isolated.

These assert the properties the design claims -- the rank of a flat-wall touch, the
pose/registration trade-off, sample-density independence, ambiguity survival -- in
settings small enough that the right answer can be worked out by hand.  The
end-to-end validation lives in ``test_validation.py``; if both fail, this file says
which mechanism broke.
"""

import math
from dataclasses import replace

import numpy as np
import pytest
from conftest import make_belief

from global_nav.belief import _systematic_resample
from global_nav.metrics import cluster_modes, effective_sample_size, weighted_pose_mean
from global_nav.models import (
    ObservationConfig,
    contact_loglik,
    max_penetration,
    swept_free_loglik,
)
from global_nav.observations import ContactObs, SweptFreeObs

# A touch on the corridor's north wall from below.  The measured normal points back
# at the robot, and it is NOT decoration -- see
# ``test_position_alone_cannot_tell_which_face_of_a_thin_slab_was_touched``.
NORTH_WALL_TOUCH = ContactObs(p_body=(0.0, 0.80), signature="surface", normal_body=(0.0, -1.0))


def _std_xy(b):
    w = b.weights
    m = weighted_pose_mean(b.xy, b.theta, w)
    return (float(np.sqrt(w @ (b.xy[:, 0] - m[0]) ** 2)),
            float(np.sqrt(w @ (b.xy[:, 1] - m[1]) ** 2)))


# ── the rank of a touch ──────────────────────────────────────────────────────────


def test_flat_wall_touch_is_rank_one_it_collapses_normal_not_tangential(plan):
    """Touching a featureless wall says how far away it is and NOTHING about where
    you are along it.  Sliding is free, and that is where the drift lives."""
    b = make_belief(plan, xy=(1.0, 0.0), sigma_xy_m=0.20, sigma_reg_m=0.0,
                    use_swept_free=False, use_structural=False)
    sx0, sy0 = _std_xy(b)
    b.update(contacts=[NORTH_WALL_TOUCH])
    sx1, sy1 = _std_xy(b)
    # the normal direction collapses to the sensor noise floor and stops there
    assert sy1 < 0.20 * sy0
    assert sy1 < 1.5 * b.cfg.observation.sigma_contact_m
    assert sx1 > 0.90 * sx0            # tangential direction does not move at all


def test_position_alone_cannot_tell_which_face_of_a_thin_slab_was_touched(plan):
    """A wall is a slab with TWO faces, and a contact scored on position alone is
    equally happy on either.  With the normal withheld, the rank-1 collapse is
    only partial, because a hypothesis that has the robot standing inside the
    alcove touching the same wall from above scores just as well.

    So the surface normal is not an optional refinement of this model: without it
    every wall in the map is a mirror.
    """
    without = make_belief(plan, xy=(1.0, 0.0), sigma_xy_m=0.20, sigma_reg_m=0.0,
                          use_swept_free=False, use_structural=False)
    with_normal = make_belief(plan, xy=(1.0, 0.0), sigma_xy_m=0.20, sigma_reg_m=0.0,
                              use_swept_free=False, use_structural=False)
    without.update(contacts=[ContactObs(p_body=(0.0, 0.80), signature="surface")])
    with_normal.update(contacts=[NORTH_WALL_TOUCH])
    assert _std_xy(without)[1] > 2.0 * _std_xy(with_normal)[1]


def test_a_corner_is_rank_two_it_collapses_both(plan):
    """The step-change the association latent is for: a named point feature pins
    position in BOTH directions, where a wall pins one."""
    b = make_belief(plan, xy=(1.4, 0.0), sigma_xy_m=0.20, sigma_reg_m=0.0,
                    use_swept_free=False, use_structural=False)
    sx0, sy0 = _std_xy(b)
    b.update(contacts=[ContactObs(p_body=(0.80, 0.80), signature="north_terminus")])
    sx1, sy1 = _std_xy(b)
    assert sx1 < 0.35 * sx0
    assert sy1 < 0.35 * sy0


def test_flat_wall_leaves_pose_and_registration_ambiguous(plan):
    """"I am 5 cm nearer than I thought" and "the wall is drawn 5 cm nearer" are the
    SAME measurement.  A touch constrains their difference and cannot split them,
    so the posterior is a ridge -- and the pose std falls by only sqrt(2), not by
    the ratio of the sensor noise to the prior.

    A Gaussian filter has to pick a point on that ridge and will pick it wrongly.
    This test is the reason the belief is a particle set.
    """
    frozen = make_belief(plan, xy=(1.0, 0.0), sigma_xy_m=0.20, sigma_reg_m=0.0,
                         use_swept_free=False, use_structural=False)
    free = make_belief(plan, xy=(1.0, 0.0), sigma_xy_m=0.20, sigma_reg_m=0.20,
                       use_swept_free=False, use_structural=False)
    frozen.update(contacts=[NORTH_WALL_TOUCH])
    free.update(contacts=[NORTH_WALL_TOUCH])

    w = free.weights
    y, r = free.xy[:, 1], free.reg[:, 1]
    ybar, rbar = float(w @ y), float(w @ r)
    corr = float(w @ ((y - ybar) * (r - rbar))) / math.sqrt(
        float(w @ (y - ybar) ** 2) * float(w @ (r - rbar) ** 2))
    assert corr > 0.60, "pose and registration must trade off along the wall normal"

    # 0.20 -> ~0.141 with the offset free, versus ~0.03 with it frozen
    assert _std_xy(free)[1] == pytest.approx(0.20 / math.sqrt(2.0), rel=0.30)
    assert _std_xy(free)[1] > 3.0 * _std_xy(frozen)[1]


# ── swept-free ───────────────────────────────────────────────────────────────────


def test_swept_free_penalises_a_hypothesised_wall_in_the_swept_volume(plan):
    """The channel that rules a hypothesis out without touching anything."""
    b = make_belief(plan, xy=(2.6, 0.0), sigma_xy_m=0.0001, sigma_reg_m=0.0)
    ray = SweptFreeObs(np.stack([np.zeros(10), np.linspace(0.11, 1.10, 10)], axis=1))
    clear = swept_free_loglik(plan, b.xy, b.theta, b.reg, ray, ObservationConfig())
    # same ray, but from x=1.0 where the north wall really is 0.8 m to the left
    b2 = make_belief(plan, xy=(1.0, 0.0), sigma_xy_m=0.0001, sigma_reg_m=0.0)
    blocked = swept_free_loglik(plan, b2.xy, b2.theta, b2.reg, ray, ObservationConfig())
    assert float(clear.mean()) == pytest.approx(0.0, abs=1e-9)
    assert float(blocked.mean()) < -1.0


def test_one_ray_is_one_measurement_not_one_per_sample(plan):
    """Doubling the sampling of the SAME ray must not double the evidence.

    Summing per-sample likelihoods -- the obvious implementation -- makes the filter
    exactly as confident as the sample spacing says, and the sample spacing is a
    filter parameter with no information in it.
    """
    b = make_belief(plan, xy=(1.0, 0.0), sigma_xy_m=0.02, sigma_reg_m=0.0)
    cfg = ObservationConfig()
    ray = SweptFreeObs(np.stack([np.zeros(22), np.linspace(0.05, 1.10, 22)], axis=1))
    doubled = SweptFreeObs(np.concatenate([ray.points, ray.points], axis=0))
    once = swept_free_loglik(plan, b.xy, b.theta, b.reg, ray, cfg)
    twice = swept_free_loglik(plan, b.xy, b.theta, b.reg, doubled, cfg)
    assert float(once.mean()) < -0.5                     # the ray really is blocked
    np.testing.assert_allclose(twice, once, atol=1e-12)  # ...and 2x the samples adds nothing


def test_a_coarsely_sampled_path_is_still_a_corridor_not_a_string_of_points(plan):
    """The corridor is what closes the gap a zero-radius ray left open.

    Max-over-samples is independent of sample COUNT but not of sample
    PLACEMENT, so a spacing coarser than a slab used to straddle it and report
    no intrusion at all.  Dilating the path by the pad's own radius means the
    samples either side of the slab already reach it: sampling stays a
    resolution question, not a correctness one.
    """
    b = make_belief(plan, xy=(1.0, 0.0), sigma_xy_m=0.0001, sigma_reg_m=0.0)
    cfg = ObservationConfig()
    resolving = SweptFreeObs(np.stack([np.zeros(22), np.linspace(0.05, 1.10, 22)], axis=1))
    straddling = SweptFreeObs(np.array([[0.0, 0.40], [0.0, 0.79], [0.0, 0.96], [0.0, 1.10]]))
    fine = float(swept_free_loglik(plan, b.xy, b.theta, b.reg, resolving, cfg).mean())
    coarse = float(swept_free_loglik(plan, b.xy, b.theta, b.reg, straddling, cfg).mean())
    assert fine < -0.5
    assert coarse < -1e-3, "the corridor must see the slab the samples straddle"
    # ...and it is still only a corridor: the coarse path never went THROUGH
    # the slab, so it may not claim the evidence the fine one has
    assert coarse > fine


def test_the_corridor_refutes_a_wall_lying_on_the_swept_path(plan):
    """The defect the finite width exists to fix.  A hypothesis that puts a
    wall exactly where the pad passed is refuted by a miss -- the pad would
    have struck it -- yet a zero-radius ray scored that case at exactly
    nothing, because the path only ever touched the surface."""
    b = make_belief(plan, xy=(1.0, 0.0), sigma_xy_m=0.0001, sigma_reg_m=0.0)
    cfg = ObservationConfig()
    grazing = SweptFreeObs(np.array([[0.0, 0.7999]]))     # the north wall face
    corridor = float(swept_free_loglik(plan, b.xy, b.theta, b.reg, grazing, cfg).mean())
    point_ray = float(swept_free_loglik(
        plan, b.xy, b.theta, b.reg,
        SweptFreeObs(grazing.points, radius_m=0.0), cfg).mean())
    assert point_ray == pytest.approx(0.0, abs=1e-6)   # only the jitter
    assert corridor < -0.1


def test_the_corridor_does_not_invent_an_intrusion_at_a_convex_corner(plan):
    """A point just past a vertex is FURTHER from the body than the maximum of
    its faces' supporting-line distances says.  Scoring the dilation on that
    max would manufacture a phantom intrusion at every convex corner in the
    map -- exactly where the free channel is most tempted to over-claim."""
    from global_nav.geometry import ConvexPolygon
    box = ConvexPolygon(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
                        "box", "surface")
    r = 0.05
    # 45 degrees out from the (1,1) vertex, at 1.4 r: outside the disc
    d = 1.4 * r / math.sqrt(2.0)
    past_vertex = np.array([[1.0 + d, 1.0 + d]])
    assert float(box.penetration(past_vertex, None, r)[0]) == pytest.approx(0.0, abs=1e-12)
    # square onto a face at the same distance from the body: inside the disc
    past_face = np.array([[0.5, 1.0 + 1.4 * r / 2.0]])
    assert float(box.penetration(past_face, None, r)[0]) > 0.0
    # and with no radius it is exactly the old point-in-body depth
    inside = np.array([[0.5, 0.9]])
    assert float(box.penetration(inside, None, 0.0)[0]) == pytest.approx(0.1)


def test_penetration_is_zero_in_free_space_and_positive_inside_a_wall(plan):
    pts = np.array([[1.0, 0.0], [1.0, 0.87], [4.5, -0.45]])   # corridor, north wall, pillar
    pen = max_penetration(plan, pts, np.zeros((3, 2)))
    assert pen[0] == 0.0
    assert pen[1] > 0.0
    assert pen[2] > 0.0


# ── association ──────────────────────────────────────────────────────────────────


def test_an_ambiguous_gate_does_not_deadlock_it_splits_the_population(plan):
    """Two map features could have produced this touch.  Every particle commits to
    ONE of them and the weight update is the marginal over both, so the population
    carries the ambiguity.

    The alternative -- "more than one candidate in the gate, therefore make no
    update" -- discards the measurement exactly when it is most informative, and is
    a failure mode this codebase has paid for before.
    """
    b = make_belief(plan, n=1200, use_swept_free=False, use_structural=False)
    # two clusters, each 0.8 m short of one of the two identical wall termini
    b.xy = np.concatenate([
        np.array([1.40, 0.0]) + 0.05 * b.rng.standard_normal((600, 2)),
        np.array([4.40, 0.0]) + 0.05 * b.rng.standard_normal((600, 2))])
    b.theta, b.reg, b.logw = np.zeros(1200), np.zeros((1200, 2)), np.zeros(1200)

    report = b.update(contacts=[ContactObs(p_body=(0.80, 0.80), signature="north_terminus")])
    assert report["n_outliers"] == 0, "an ambiguous contact is not an impossible one"
    assert report["n_contacts"] == 1, "the update must actually happen"

    lm = b.landmark_probabilities()
    assert lm[0] > 0.30 and lm[1] > 0.30, "both candidates must keep real mass"
    assert lm[0] + lm[1] == pytest.approx(1.0, abs=1e-9)


def test_a_contact_the_map_cannot_produce_is_rejected_and_changes_nothing(plan):
    b = make_belief(plan, xy=(1.0, 0.0), use_swept_free=False, use_structural=False)
    before = b.logw.copy()
    report = b.update(contacts=[ContactObs(p_body=(0.35, 0.02), signature="door_handle")])
    assert report["n_outliers"] == 1
    assert report["n_contacts"] == 0
    assert np.allclose(b.logw, before)


def test_the_outlier_gate_is_mass_based_not_best_particle_based(plan):
    """One particle in a thousand is a 1-in-N extreme and can 'explain' anything.

    Here a phantom contact 0.78 m from any wall is met by a deliberately diffuse
    belief.  Some particle always fits it -- a best-particle gate accepts -- but
    only ~1% of the MASS does, and admitting it would move the other 99% onto that
    1% tail.  That is not a small error; it is a confident jump to wherever the
    spurious touch pointed.
    """
    b = make_belief(plan, n=2000, xy=(1.0, 0.0), sigma_xy_m=0.25, sigma_reg_m=0.10)
    cfg = ObservationConfig()
    phantom = ContactObs(p_body=(0.35, 0.02), signature="door_handle")

    ungated = contact_loglik(plan, b.xy, b.theta, b.reg, phantom,
                             replace(cfg, outlier_min_support=0.0), b.rng, weights=b.weights)
    assert float(-2.0 * ungated.loglik.max()) < cfg.outlier_gate_chi2, \
        "a best-particle gate would have accepted this phantom"

    upd = contact_loglik(plan, b.xy, b.theta, b.reg, phantom, cfg, b.rng, weights=b.weights)
    assert upd.support < 0.05
    assert upd.is_outlier


# ── multi-hypothesis preservation ────────────────────────────────────────────────


def test_symmetric_evidence_never_breaks_a_fifty_fifty_split(plan):
    """The decisive test for "do not collapse immediately to one Gaussian".

    Two clusters sit under the same featureless wall, 1.2 m apart.  A north-wall
    touch is EXACTLY as likely for either -- the wall is uniform in x -- so no
    amount of it may move the split.  Anything that does is the filter inventing
    information: resampling noise eating a mode, or a likelihood that is not
    actually symmetric.
    """
    b = make_belief(plan, n=1600, use_structural=False)
    b.xy = np.concatenate([
        np.array([0.50, 0.0]) + 0.08 * b.rng.standard_normal((800, 2)),
        np.array([1.70, 0.0]) + 0.08 * b.rng.standard_normal((800, 2))])
    b.theta, b.reg, b.logw = np.zeros(1600), np.zeros((1600, 2)), np.zeros(1600)

    for _ in range(12):
        b.update(contacts=[NORTH_WALL_TOUCH])

    modes = cluster_modes(b.xy, b.weights, radius_m=0.45)
    assert len(modes) >= 2
    assert 0.35 < modes[0]["weight"] < 0.65
    assert modes[0]["weight"] + modes[1]["weight"] > 0.95
    assert effective_sample_size(b.weights) > 0.45 * b.n


def test_systematic_resampling_cannot_delete_a_minority_mode():
    """A mode holding fraction f is guaranteed floor(f*N) survivors.  Multinomial
    resampling gives it zero often enough to lose it outright."""
    rng = np.random.default_rng(0)
    n = 1000
    w = np.concatenate([np.full(150, 0.15 / 150), np.full(850, 0.85 / 850)])
    for _ in range(200):
        idx = _systematic_resample(w, rng)
        assert int(np.sum(idx < 150)) >= int(math.floor(0.15 * n))


def test_resampling_only_fires_when_the_effective_sample_size_falls(plan):
    """Resampling every step is the fastest way to lose a minority hypothesis."""
    b = make_belief(plan, xy=(1.0, 0.0), use_contacts=False, use_swept_free=False,
                    use_structural=False)
    for _ in range(10):
        b.update()
    assert b.n_resamples == 0
    assert effective_sample_size(b.weights) == pytest.approx(b.n)


# ── route edge and wrong turns ───────────────────────────────────────────────────


def test_turning_into_the_decoy_marks_the_particle_off_route(plan):
    """The successor is SAMPLED from a von Mises over the branches' bearings, so a
    north-facing particle takes the decoy with probability exp(kappa)/(1+exp(kappa))
    -- about 98% at kappa=4, not 100%.  That residual is the model admitting a
    heading measurement is not a certainty, and it is deliberate."""
    b = make_belief(plan, n=400, xy=(3.05, 0.0), theta=math.pi / 2, edge=0,
                    sigma_xy_m=0.01)
    b.theta = np.full(400, math.pi / 2)
    b._transition_edges()
    assert np.mean(b.edge == 3) > 0.95
    assert b.wrong_turn_probability() > 0.95


def test_going_straight_keeps_the_particle_on_route(plan):
    b = make_belief(plan, n=400, xy=(3.05, 0.0), theta=0.0, edge=0, sigma_xy_m=0.01)
    b.theta = np.zeros(400)
    b._transition_edges()
    assert np.mean(b.edge == 1) > 0.95
    assert b.wrong_turn_probability() < 0.05


def test_a_particle_that_leaves_sideways_at_a_junction_still_transitions(plan):
    """Trigger on arclength alone and a particle that turns early never transitions
    at all: it drives up the decoy alcove still labelled with the corridor it left,
    and the wrong turn goes uncounted."""
    b = make_belief(plan, n=400, xy=(2.90, 1.00), theta=math.pi / 2, edge=0,
                    sigma_xy_m=0.01)
    b.theta = np.full(400, math.pi / 2)
    s, lat = plan.edge(0).project(b.xy)
    assert np.all(s < plan.edge(0).length_m), "arclength alone would NOT fire here"
    b._transition_edges()
    assert np.mean(b.edge == 3) > 0.95
    assert b.wrong_turn_probability() > 0.95


def test_mid_corridor_lateral_drift_does_not_fake_a_wrong_turn(plan):
    b = make_belief(plan, n=400, xy=(1.00, 1.00), theta=math.pi / 2, edge=0,
                    sigma_xy_m=0.01)
    b.theta = np.full(400, math.pi / 2)
    b._transition_edges()
    assert np.all(b.edge == 0)
    assert b.wrong_turn_probability() == pytest.approx(0.0)


def test_off_route_is_sticky(plan):
    """A particle that took the decoy stays wrong even if its pose later drifts back
    over the true corridor -- because it did in fact turn."""
    b = make_belief(plan, n=200, xy=(3.05, 0.0), theta=math.pi / 2, edge=0, sigma_xy_m=0.01)
    b.theta = np.full(200, math.pi / 2)
    b._transition_edges()
    before = b.wrong_turn_probability()
    b.xy = np.tile(np.array([[5.0, 0.0]]), (200, 1))
    assert before > 0.95
    assert b.wrong_turn_probability() == pytest.approx(before)


# ── motion ───────────────────────────────────────────────────────────────────────


def test_noiseless_odometry_moves_every_particle_exactly(plan):
    b = make_belief(plan, n=200, xy=(1.0, 0.0), sigma_xy_m=0.0, sigma_reg_m=0.0)
    b.theta = np.zeros(200)
    b.cfg = type(b.cfg)(**{**b.cfg.__dict__,
                           "motion": type(b.cfg.motion)(0.0, 0.0, 0.0, 0.0, 0.0)})
    b.predict(np.array([0.5, 0.0]), 0.0)
    assert b.xy == pytest.approx(np.tile([[1.5, 0.0]], (200, 1)))
    b.predict(np.zeros(2), math.pi / 2)
    b.predict(np.array([0.5, 0.0]), 0.0)
    assert b.xy == pytest.approx(np.tile([[1.5, 0.5]], (200, 1)), abs=1e-12)


def test_odometry_spread_grows_with_distance_walked(plan):
    b = make_belief(plan, n=2000, xy=(0.5, 0.0), sigma_xy_m=0.0, sigma_reg_m=0.0)
    spreads = []
    for _ in range(8):
        b.predict(np.array([0.25, 0.0]), 0.0)
        spreads.append(_std_xy(b)[0])
    assert all(a < c for a, c in zip(spreads, spreads[1:])), "drift must accumulate"
    assert spreads[-1] > 2.0 * spreads[0]
