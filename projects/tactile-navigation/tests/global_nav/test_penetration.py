"""A contact point inside a thin wall slab must be scored by how far it is
past the face the fist came through, not by its distance to the slab's back
face (`ConvexPolygon.penetration_along`, `ObservationConfig.penetration_along_press`)."""
from __future__ import annotations

import numpy as np
import pytest

from global_nav.floorplan import Floorplan, RouteEdge
from global_nav.geometry import ConvexPolygon
from global_nav.models import ObservationConfig, contact_loglik
from global_nav.observations import ContactObs


def _slab():
    # a wall slab from x = -0.69 to -0.59, tall in y; the room is at x > -0.59
    return ConvexPolygon(np.array([[-0.69, -1.0], [-0.59, -1.0], [-0.59, 5.0], [-0.69, 5.0]]), "wall_0", "surface")


def test_penetration_along_measures_depth_through_the_front_face():
    wall = _slab()
    pts = np.array([[-0.68, 1.0],      # 9 cm inside, 1 cm from the back face
                    [-0.60, 1.0],      # 1 cm inside
                    [-0.40, 1.0]])     # outside
    back = np.tile([[1.0, 0.0]], (3, 1))   # the fist came from +x
    depth = wall.penetration_along(pts, back)
    assert depth == pytest.approx([0.09, 0.01, 0.0], abs=1e-9)


def test_likelihood_scores_a_contact_as_a_range_along_the_press():
    wall = _slab()
    plan = Floorplan((wall,), (), (RouteEdge(0, (-1, -1), (5, 5), half_width_m=10.0, successors=(), name="f"),), route=(0,))
    # the fist touched the wall 0.45 m ahead of the base, pressing forward (+x body); the normal is the press reversed
    obs = ContactObs(p_body=(0.45, 0.0), signature="surface", normal_body=(-1.0, 0.0), normal_sigma_rad=0.0)   # normal points back at the robot
    theta = np.full(4, np.pi)                       # facing the wall (-x)
    xy = np.array([[-0.14, 1.0],                    # truth: 0.45 m from the face -> contact on the face
                   [-0.23, 1.0],                    # 9 cm too close -> contact 9 cm inside the slab
                   [-0.05, 1.0],                    # 9 cm too far  -> contact 9 cm short of the face
                   [-0.26, 1.0]])                   # 12 cm too close -> contact BEHIND the slab, 2 cm past its back face
    rng = np.random.default_rng(0)
    old = contact_loglik(plan, xy, theta, np.zeros((4, 2)), obs, ObservationConfig(sigma_contact_m=0.018, penetration_along_press=False), rng)
    new = contact_loglik(plan, xy, theta, np.zeros((4, 2)), obs, ObservationConfig(sigma_contact_m=0.018, penetration_along_press=True), rng)
    # old: the too-close particle looks almost as good as the truth (1 cm from the back face)
    assert old.loglik[1] > old.loglik[2] + 5.0
    assert old.loglik[3] > old.loglik[2] + 5.0             # ...and so does the one behind the slab
    # new: too close, too far and behind are all penalized by their range error; the truth wins clearly
    assert new.loglik[0] > max(new.loglik[1], new.loglik[2], new.loglik[3]) + 5.0
    assert new.loglik[1] == pytest.approx(new.loglik[2], abs=1e-6)
    assert new.loglik[3] < new.loglik[1]                     # 12 cm wrong is worse than 9 cm wrong
