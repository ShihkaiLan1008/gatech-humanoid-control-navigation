"""The unified IG family must EQUAL both existing implementations before
either is allowed to route through it -- the unification is an extraction,
not a behavior change."""
import math

import numpy as np
import pytest

from demo_map.economy import (TouchEconomy, drift_sigma, miss_truncation_ig,
                              unified_ig)


def test_rank1_reduces_to_the_wall_closed_form():
    """H = n̂ᵀ, R = σ_t²  ->  0.5·ln(1 + σ_n²/σ_t²), the exact expression
    `_next_wall_press` evaluates."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        A = rng.normal(size=(2, 2))
        Sigma = A @ A.T + 0.01 * np.eye(2)
        th = rng.uniform(0, 2 * math.pi)
        n_hat = np.array([math.cos(th), math.sin(th)])
        sigma_t = rng.uniform(0.005, 0.1)
        sigma_n2 = float(n_hat @ Sigma @ n_hat)
        want = 0.5 * math.log1p(sigma_n2 / sigma_t ** 2)
        got = unified_ig(n_hat, Sigma, sigma_t ** 2)
        assert got == pytest.approx(want, rel=1e-9)


def test_matches_pose_score_form_logdet_prior_plus_posterior():
    """Equal to ½[logdet Σ + logdet(Σ⁻¹ + HᵀR⁻¹H)] -- the exact form
    `tactile_explore.infogain._pose_score` computes -- for rank 1 and 2."""
    rng = np.random.default_rng(1)
    for m in (1, 2):
        for _ in range(25):
            A = rng.normal(size=(3, 3))
            Sigma = A @ A.T + 0.05 * np.eye(3)
            H = rng.normal(size=(m, 3))
            B = rng.normal(size=(m, m))
            R = B @ B.T + 0.01 * np.eye(m)
            info = H.T @ np.linalg.inv(R) @ H
            _, logdet_prior = np.linalg.slogdet(Sigma)
            _, logdet_post = np.linalg.slogdet(np.linalg.inv(Sigma) + info)
            want = 0.5 * (logdet_prior + logdet_post)
            got = unified_ig(H, Sigma, R)
            assert got == pytest.approx(want, rel=1e-9)


def test_rank2_beats_rank1_on_the_same_prior():
    """Two independent directions buy more than one -- the reason a corner
    or a second cylinder bearing is worth walking to."""
    Sigma = np.diag([0.04, 0.04])
    r = 0.018 ** 2
    one = unified_ig(np.array([[1.0, 0.0]]), Sigma, r * np.eye(1))
    two = unified_ig(np.eye(2), Sigma, r * np.eye(2))
    assert two > one
    assert two == pytest.approx(2 * one, rel=1e-9)  # symmetric prior


def test_parallel_touches_add_almost_nothing():
    """A second touch along the SAME normal is nearly redundant -- the
    geometric fact behind '2-3 touches suffice when shape is known'."""
    Sigma = np.diag([0.04, 0.04])
    r = 0.018 ** 2
    one = unified_ig(np.array([[1.0, 0.0]]), Sigma, r * np.eye(1))
    same = unified_ig(np.array([[1.0, 0.0], [1.0, 0.0]]), Sigma,
                      r * np.eye(2))
    ortho = unified_ig(np.eye(2), Sigma, r * np.eye(2))
    assert same < one + 0.5 * math.log(2.0) + 1e-9   # ~one bit of averaging
    assert ortho > same


def test_economy_defaults_are_the_currently_tuned_prices():
    eco = TouchEconomy()
    assert eco.lambda_motion == 0.5
    assert eco.lambda_backtrack == 2.5
    assert eco.lambda_coverage == 0.4
    assert eco.lambda_risk == 2.5
    assert eco.wall_floor_nats == 0.15
    assert eco.landmark_floor_nats == 0.06


def test_drift_sigma_uses_the_filter_motion_constant():
    assert drift_sigma(1.0) == pytest.approx(0.104)
    assert drift_sigma(-2.0, 0.05) == pytest.approx(0.10)


# ── one uncertainty cost, for every candidate class ──────────────────────────


def test_the_uncertainty_cost_prices_the_ACTION_not_the_belief():
    """Incremental, so a candidate that requires no travel costs nothing.

    An absolute price -- lambda_r * sigma_after -- is a toll every candidate
    at a station pays equally.  It therefore cancels out of the ranking, which
    is the only thing it looks like it is doing, and silently moves the
    accept/decline floor instead: at sigma 0.10 and lambda_r 2.5 that is
    0.25 nats against a 0.15-nat floor."""
    from demo_map.economy import TouchEconomy, drift_risk_nats, drift_sigma

    eco = TouchEconomy()
    for sig in (0.0, 0.02, 0.10, 0.40, 1.2):
        assert drift_risk_nats(eco, sig, 0.0) == pytest.approx(0.0)
        assert drift_risk_nats(eco, sig, -0.7) \
            == pytest.approx(drift_risk_nats(eco, sig, 0.7))
    # monotone in the hop, and strictly concave -- the curvature is what
    # makes an interior optimum exist at all
    d = [drift_risk_nats(eco, 0.10, x) for x in (0.5, 1.0, 1.5, 2.0)]
    assert d[0] < d[1] < d[2] < d[3]
    # quadratic near zero, asymptotically linear: the increments GROW, which
    # is the curvature that gives the wall objective an interior optimum
    assert (d[1] - d[0]) < (d[2] - d[1]) < (d[3] - d[2])
    # and it is exactly the growth the filter's own motion model predicts
    assert drift_risk_nats(eco, 0.10, 1.0) == pytest.approx(
        eco.lambda_risk * (math.hypot(0.10, drift_sigma(1.0)) - 0.10))


def test_wall_and_landmark_charge_the_same_price_for_the_same_hop():
    """The two candidate classes were not comparable: the wall charged the
    ABSOLUTE post-walk sigma and the landmark the INCREMENT, so a shared
    arbiter was ranking nats against nats-minus-a-constant."""
    import numpy as np
    from types import SimpleNamespace

    import demo_map.executor as X
    from demo_map.economy import drift_risk_nats
    from demo_map.executor import DemoExecCfg, DemoRouteExecutor
    from demo_map.landmark_ate import AteProbe, LandmarkATE
    from execution.wall_lane import WallSegment
    from tactile_explore.infogain import EIGTerms

    sigma, walk = 0.18, 0.60
    eco = DemoExecCfg().economy
    expected = drift_risk_nats(eco, sigma, walk)
    assert expected > 0.0

    # -- the landmark side: utility drop of a hop, information held fixed
    ate = LandmarkATE.cylinder("x", (0.0, 0.0), 0.2)
    ate.reaches_what_it_is_scored_for = lambda p: True
    ate.score_probe = lambda p: (None, p.terms)

    def _probe_at(s):
        q = AteProbe("f", float(s), np.zeros(2), np.array([1.0, 0.0]))
        q.terms = EIGTerms(ig_pose=2.0, ig_hyp=0.0, cost_s=1.0, p_hit=0.9,
                           ig_miss=0.0)
        return q

    here, there = _probe_at(1.0), _probe_at(1.0 + walk)
    ate.rank([here, there], progress_from={"f": 1.0}, pose_sigma_m=sigma)
    landmark_charge = (here.utility - there.utility) - eco.lambda_motion * walk
    assert landmark_charge == pytest.approx(expected, abs=1e-9)

    # -- the wall side: J, recomputed from its published parts
    ex = object.__new__(DemoRouteExecutor)
    # the price under test is the analytic objective's; the WM default would
    # need a checkpoint and a belief this stub does not carry
    ex.cfg = DemoExecCfg(ig_backend="analytic_marginal")
    ex.scene = SimpleNamespace(plan=SimpleNamespace(bodies=()))
    ex.be = SimpleNamespace(mj=SimpleNamespace(vf_touch_target_m=0.43,
                                               vf_press_stance_m=0.36))
    ex.be.mj.probe_hand_z_m = 0.25
    ex.be.mj.park_x_m = 0.14
    ex._cov_xy = lambda: (sigma ** 2) * np.eye(2)
    seg = WallSegment(p0=np.zeros(2), tangent=np.array([1.0, 0.0]),
                      normal=np.array([0.0, -1.0]), length=8.0, name="w")
    orig = X.face_present
    X.face_present = lambda *a, **k: True
    try:
        cand, j, _tr = ex._next_wall_press(seg, 1.0, covered=[])
    finally:
        X.face_present = orig
    assert cand is not None
    s_c = float(cand.target.s_m)
    k = float((ex.cfg.motion or {}).get("sigma_cross_per_m", 0.104))
    n_hat = np.asarray(seg.normal, float)
    cov = ex._cov_xy()
    sig_n = float(math.sqrt(max(n_hat @ cov @ n_hat, 1e-8)))
    w = float(s_c) - 1.0
    sig_pre = math.hypot(sig_n, k * w)
    margin = 0.43 - 0.36
    p_hit = 0.5 * (1.0 + math.erf(margin / (sig_pre * math.sqrt(2.0))))
    ig = (p_hit * unified_ig(n_hat, cov + (k * w) ** 2 * np.eye(2),
                             0.018 ** 2)
          + (1.0 - p_hit) * miss_truncation_ig(margin, sig_pre))
    rebuilt = (ig + eco.lambda_coverage - eco.lambda_motion * w
               - drift_risk_nats(eco, sig_n, w, k))
    assert j == pytest.approx(rebuilt, abs=1e-9)
    # the same function, the same arguments, the same number
    assert drift_risk_nats(eco, sig_n, walk, k) == pytest.approx(expected)
