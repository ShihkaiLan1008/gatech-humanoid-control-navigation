import pytest
import numpy as np


def test_a_vertex_touch_can_decline_to_claim_a_normal(plan):
    """A corner kiss lands on the point where two faces meet.  There is no
    single surface normal there, but the filter scores the felt normal
    against the NEAREST FLAT FACE, 45 deg away.  Measured (room2 forward,
    seed 0, sigma_normal 0.10): that one observation rotated the belief from
    9 to 28.5 deg of yaw while position error read its best ever, 0.035 m --
    heading sold to buy position -- and 1.7 m later the hidden yaw surfaced
    as 0.936 m of position error.  Per-observation tolerance lets the touch
    contribute the part it knows (position) and withhold the part it does
    not."""
    from global_nav.models import ObservationConfig, contact_loglik
    from global_nav.observations import ContactObs

    cfg = ObservationConfig()
    rng = np.random.default_rng(0)
    xy = np.array([[1.0, 0.0], [1.0, 0.0]])
    theta = np.array([0.0, 0.15])           # one particle rotated 8.6 deg
    reg = np.zeros((2, 2))

    def spread(sig):
        """How hard this touch argues AGAINST the rotated particle."""
        o = ContactObs(p_body=(0.0, 0.80), normal_body=(0.0, -1.0),
                       normal_sigma_rad=sig)
        u = contact_loglik(plan, xy, theta, reg, o, cfg, rng)
        assert not u.is_outlier
        return float(u.loglik[0] - u.loglik[1])

    scored, declined = spread(0.10), spread(0.0)
    assert scored > 1.0               # scored: the rotated particle is punished
    assert declined < 0.1             # declined: heading no longer decides
    assert abs(spread(None) - spread(cfg.sigma_normal_rad)) < 1e-9


def _two_face_plan():
    """Two slabs whose faces look the same to a POSITION residual and
    opposite to a NORMAL one: a horizontal wall above the origin and a
    vertical wall the same distance to the right."""
    from global_nav.floorplan import Floorplan
    from global_nav.geometry import ConvexPolygon
    north = ConvexPolygon(np.array([[-2.0, 1.0], [2.0, 1.0],
                                    [2.0, 1.15], [-2.0, 1.15]]),
                          "north", "surface")
    east = ConvexPolygon(np.array([[1.0, -2.0], [1.15, -2.0],
                                   [1.15, 2.0], [1.0, 2.0]]),
                         "east", "surface")
    return Floorplan((north, east), (), (), route=(), name="two_face")


def test_the_normal_is_scored_against_the_primitive_the_position_was():
    """The correspondence defect, in one measurement.

    A hand at (0.7, 0.7) is equidistant from both faces, so POSITION alone
    cannot say which it touched -- that is what the correspondence marginal is
    for.  A felt normal pointing south (out of the north wall) settles it.
    Scoring that normal against `nearest_surface` -- whichever face happens to
    be nearest, computed once, added after the marginal -- shifts both
    candidates by the same amount and settles nothing.  Scored per primitive,
    the association posterior has to land on the north wall."""
    from global_nav.models import ObservationConfig, contact_loglik
    from global_nav.observations import ContactObs

    plan = _two_face_plan()
    cfg = ObservationConfig()
    n = 4000
    xy = np.zeros((n, 2))
    theta = np.zeros(n)
    reg = np.zeros((n, 2))
    rng = np.random.default_rng(0)
    p = (0.99, 0.99)                     # 0.01 m off BOTH faces
    i_north, i_east = plan.entity_id("body", 0), plan.entity_id("body", 1)
    facing_north = ContactObs(p_body=p, normal_body=(0.0, -1.0))
    facing_east = ContactObs(p_body=p, normal_body=(-1.0, 0.0))
    up = contact_loglik(plan, xy, theta, reg, facing_north, cfg, rng)
    side = contact_loglik(plan, xy, theta, reg, facing_east, cfg, rng)
    assert not up.is_outlier and not side.is_outlier
    assert (up.assoc == i_north).mean() > 0.95
    assert (side.assoc == i_east).mean() > 0.95
    # ...and the WEIGHT update, not only the label: a normal pointing INTO
    # the material agrees with neither face, and the marginal has to say so.
    away = ContactObs(p_body=p, normal_body=(0.0, 1.0))
    wrong = contact_loglik(plan, xy, theta, reg, away, cfg, rng)
    assert float(wrong.loglik[0]) < float(up.loglik[0]) - 1.0


def test_an_unambiguous_contact_reduces_to_its_one_compatible_primitive():
    """`Generic wall contacts reduce to a single primitive when unambiguous`
    is not a special case in the code -- it is what the marginal does on its
    own once one candidate dominates.  Against the north wall alone, the
    two-body marginal must equal that body's own residual, less the uniform
    prior's log(C)."""
    import math

    from global_nav.models import ObservationConfig, contact_loglik
    from global_nav.observations import ContactObs

    plan = _two_face_plan()
    cfg = ObservationConfig()
    rng = np.random.default_rng(1)
    xy = np.zeros((1, 2))
    theta = np.zeros(1)
    reg = np.zeros((1, 2))
    obs = ContactObs(p_body=(0.0, 0.99), normal_body=(0.0, -1.0))
    u = contact_loglik(plan, xy, theta, reg, obs, cfg, rng)
    sd, _n = plan.bodies[0].signed_distance(np.array([[0.0, 0.99]]), reg)
    alone = -0.5 * (abs(float(sd[0])) / cfg.sigma_contact_m) ** 2
    assert float(u.loglik[0]) == pytest.approx(alone - math.log(2.0), abs=1e-6)
    assert int(u.assoc[0]) == plan.entity_id("body", 0)
