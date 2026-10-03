"""The clean-finish contract: heading survives a touch, transit survives
belief error, and the corner does not get kissed for ceremony.

Each test pins one of the three seams the v14-v18 post-mortems named:
the contact update rotating the published heading (v14 t=48.3, v17
t=49.2), a fixed transit margin eaten by a healthy belief's own error
(v18 t=158), and the post-gate kiss detour no run ever profited from."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from demo_map.executor import (CARRY_CLEAR_M, DemoExecCfg, DemoRouteExecutor,
                               DemoRun)
from global_nav.geometry import ConvexPolygon


def test_the_imu_anchors_the_heading_after_the_contact_update():
    """v14/v17: the reweighting explained a rank-1 residual by rotating the
    elongated cloud (2.2 -> 19 deg in one press, truth 6.5 sigma out), and
    with the anchor upstream of `observe` that rotation was the published
    heading.  The anchor must therefore run AFTER the observation, so the
    contact spends its information on position and the IMU has the last
    word on heading."""
    ex = object.__new__(DemoRouteExecutor)
    ex.cfg = DemoExecCfg()
    ex.run_rec = DemoRun()
    ex.pf_log = None
    order = []
    ex.mb = SimpleNamespace(
        predict=lambda d, y: order.append("predict"),
        observe=lambda c, f: order.append("observe") or {},
        heading_anchor=lambda y: order.append("heading_anchor"),
        hypotheses=lambda: [],
        b=SimpleNamespace(last_contacts=[]))
    ex.be = SimpleNamespace(
        odometry_increment=lambda: ((0.0, 0.0), 0.0),
        heading_since_reset=lambda: 0.5,
        time_s=1.0,
        truth_pose=lambda: (np.zeros(3), 0.0),
        world_point_to_map=lambda p: np.zeros(2),
        world_yaw_to_map=lambda y: 0.0)
    ex._believed_xy = lambda: np.zeros(2)
    ex._believed_yaw = lambda: 0.0
    ex._sigma = lambda: 0.1
    ex._sigma_yaw = lambda: 0.05
    ex._cov_xy = lambda: 0.01 * np.eye(2)
    ex._absorb("press", observed={"contacts": [object()], "frees": []})
    assert order == ["predict", "observe", "heading_anchor"]


def _clearance_ex(cov):
    ex = object.__new__(DemoRouteExecutor)
    ex.cfg = DemoExecCfg()
    body = ConvexPolygon(np.array([[1.0, 0.5], [2.0, 0.5],
                                   [2.0, 1.5], [1.0, 1.5]]), "b", "surface")
    ex.scene = SimpleNamespace(plan=SimpleNamespace(bodies=(body,)))
    ex._believed_xy = lambda: np.zeros(2)
    ex._cov_xy = lambda: np.asarray(cov, float)
    return ex


def test_transit_clearance_grows_with_the_beliefs_own_sigma():
    """v18: a waypoint planned 0.40 m clear was walked, through a 0.27 m
    belief error at 1.2 sigma, to a true 0.14 m from the body, and the
    carried hand force-aborted a healthy run.  The margin must budget the
    error the covariance itself admits to: carry envelope + 2 sigma
    perpendicular to travel."""
    # target 0.45 below the body's south face: clear of the 0.40 carry
    # envelope on its own, so a tight belief leaves it alone...
    p = np.array([1.5, 0.05])
    tight = _clearance_ex(1e-8 * np.eye(2))._transit_target(p)
    assert np.allclose(tight, p, atol=1e-6)
    # ...and a belief with 0.15 sigma across the (x-axis) travel direction
    # pushes it out to carry + 2 sigma from the face
    wide = _clearance_ex(np.diag([1e-8, 0.15 ** 2]))._transit_target(p)
    assert wide[1] < p[1] - 1e-3, "must be pushed away from the body"
    assert 0.5 - wide[1] >= CARRY_CLEAR_M + 2.0 * 0.15 - 1e-3
    # sigma ALONG the travel direction must not inflate the margin
    along = _clearance_ex(np.diag([0.15 ** 2, 1e-8]))._transit_target(p)
    assert np.allclose(along, p, atol=1e-6)


def test_arrival_is_not_claimed_by_a_belief_that_cannot_support_it():
    """demo_room seed 1 reported `arrived` on a believed 0.591 m while
    standing 1.902 m away, carrying sigma 0.612 m.

    The wall was lost on the corridor, the leg closed with no candidate left
    to press, 10.7 m were walked with nothing to absorb, and sigma grew
    0.041 -> 1.34 into three modes before the corner touches re-converged it
    onto the wrong one.  A point estimate compared against a 0.35 m tolerance
    never asked any of that.  It is asked here."""
    ex = object.__new__(DemoRouteExecutor)
    ex.cfg = DemoExecCfg()
    ex.scene = SimpleNamespace(goal_radius_m=0.40)
    tol = ex.cfg.goal_tol_m + 0.40                       # 0.75 m

    ex._sigma = lambda: 0.612
    assert ex._arrival_outcome(0.591) == "arrived_unconfirmed"
    # the same stance under a belief that has earned the claim
    ex._sigma = lambda: 0.066
    assert ex._arrival_outcome(0.591) == "arrived"
    # and walking to the wrong place stays its own answer, at any sigma
    for sig in (0.02, 0.9):
        ex._sigma = lambda s=sig: s
        assert ex._arrival_outcome(tol + 0.1) == "route_done_off_goal"


def test_the_confidence_gate_is_one_sigma_not_a_second_tolerance():
    """The rule is `believed + sigma <= tolerance` -- one statement about
    where the goal is, at one sigma -- and not a separate sigma cap that
    would refuse a confident robot standing on the goal."""
    ex = object.__new__(DemoRouteExecutor)
    ex.cfg = DemoExecCfg()
    ex.scene = SimpleNamespace(goal_radius_m=0.40)
    tol = ex.cfg.goal_tol_m + 0.40

    # dead on the goal, sigma just under the whole tolerance: still arrived
    ex._sigma = lambda: tol - 1e-3
    assert ex._arrival_outcome(0.0) == "arrived"
    # a hair further out and the same sigma no longer covers it
    ex._sigma = lambda: tol - 1e-3
    assert ex._arrival_outcome(0.01) == "arrived_unconfirmed"


def test_a_confirmed_goal_marker_is_arrival_whatever_the_radius_says():
    """e36f754's rule, carried into the outcome: the ATE pressed the required
    goal marker and closed the visit `confirmed`, so the run arrived -- the
    confirming stance stands marker-radius + standoff from the goal POINT,
    and failing that geometry against a 0.75 m radius failed a semantically
    complete arrival (auto demo_room s0: belief error 0.116 m, `confirmed`,
    judged route_done_off_goal at 1.0 m)."""
    ex = object.__new__(DemoRouteExecutor)
    ex.cfg = DemoExecCfg()
    ex.scene = SimpleNamespace(goal_radius_m=0.40)
    ex._sigma = lambda: 0.05
    steps = [{"kind": "follow", "status": "done"},
             {"kind": "ate_pillar", "required": True, "status": "confirmed"}]
    assert ex._arrival_outcome(1.0, steps) == "arrived"
    # an unconfirmed or optional visit changes nothing
    assert ex._arrival_outcome(1.0, [{"kind": "ate_pillar", "required": True,
                                      "status": "not_felt"}]) \
        == "route_done_off_goal"
    assert ex._arrival_outcome(1.0, [{"kind": "ate_pillar",
                                      "status": "confirmed"}]) \
        == "route_done_off_goal"
