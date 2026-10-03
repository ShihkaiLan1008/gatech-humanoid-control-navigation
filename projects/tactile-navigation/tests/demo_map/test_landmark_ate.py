"""LandmarkATE: the ported scorer really is the local planner.

Synthetic world, no simulator: a corner at a known true pose, a belief whose
prior is offset, candidates on both faces.  What is asserted is the BEHAVIOUR
the user asked the ATE to supply: it prefers the press that pins the axis the
belief has not seen, its posterior converges onto the true offset, redundant
re-presses lose value, and it says STOP on its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from demo_map.landmark_ate import AteProbe, LandmarkATE


def _probes(corner, n_a, n_b, along_a, along_b):
    """Candidates on the two faces of a CONCAVE corner: face A (inward normal
    n_a) spans the n_b side of the corner, face B spans the n_a side."""
    out = []
    t_a = np.asarray(n_b, float)           # along face A away from the corner
    t_b = np.asarray(n_a, float)
    for d in along_a:
        out.append(AteProbe("A", float(d),
                            np.asarray(corner) + d * t_a,
                            -np.asarray(n_a, float)))
    for d in along_b:
        out.append(AteProbe("B", float(d),
                            np.asarray(corner) + d * t_b,
                            -np.asarray(n_b, float)))
    return out


@pytest.fixture()
def world():
    corner = np.array([0.0, 0.0])
    n_a = np.array([1.0, 0.0])             # face A: west wall, room to +x
    n_b = np.array([0.0, -1.0])            # face B: north wall, room to -y
    true_off = np.array([0.06, -0.09])     # the robot's (negated) pose error
    return corner, n_a, n_b, true_off


def _touch(corner, n, true_off, along, t_dir):
    """Where a press at `along` on the face with inward normal n ACTUALLY
    lands: on the offset face plane, `along` down the face's own span."""
    p = corner + true_off + along * t_dir
    return p, n


def test_orthogonal_face_beats_a_redundant_re_press(world):
    corner, n_a, n_b, true_off = world
    ate = LandmarkATE.corner("c", corner, n_a, n_b, seed=3)
    probes = _probes(corner, n_a, n_b, [0.35, 0.7], [0.35, 0.7])
    ranked = ate.rank(probes)
    first = ranked[0]
    # press face A once (contact on the true, offset plane)
    p, n = _touch(corner, n_a, true_off, first.s, np.asarray(n_b))
    ate.feed_contact(p, n_a)
    # after one A-face touch, the BEST next press must be on face B: the
    # A-normal axis is pinned, the B-normal axis is not
    ranked2 = ate.rank(_probes(corner, n_a, n_b, [0.35, 0.7], [0.35, 0.7]))
    assert ranked2[0].seg_key == "B", [(q.seg_key, round(q.score_nats, 3))
                                       for q in ranked2]
    # and a repeat at the SAME face scores strictly below the fresh face
    best_b = max(q.score_nats for q in ranked2 if q.seg_key == "B")
    best_a = max(q.score_nats for q in ranked2 if q.seg_key == "A")
    assert best_b > best_a


def test_two_presses_pin_the_offset_and_stop(world):
    corner, n_a, n_b, true_off = world
    ate = LandmarkATE.corner("c", corner, n_a, n_b, seed=5)
    h0 = ate.entropy()
    for seg_key, along in (("A", 0.4), ("B", 0.4)):
        n = n_a if seg_key == "A" else n_b
        t = np.asarray(n_b) if seg_key == "A" else np.asarray(n_a)
        p, _ = _touch(corner, n, true_off, along, t)
        ate.feed_contact(p, n)
    est = ate.offset_estimate()
    # the ccorner state IS the corner point: it must land on the true corner
    corner_shift = np.asarray(est["mean"][:2]) - np.asarray(corner)
    assert np.allclose(corner_shift, true_off, atol=0.02), corner_shift
    assert est["sigma"][0] < 0.035 and est["sigma"][1] < 0.035
    assert ate.entropy() < h0 - 1.0        # nats, a real collapse
    # and the planner now says the well is dry
    ranked = ate.rank(_probes(corner, n_a, n_b, [0.35, 0.7], [0.35, 0.7]))
    assert not ate.worth_pressing(ranked[0]) or \
        ranked[0].score_nats < 0.25 * (h0 - ate.entropy())


def test_a_miss_is_evidence_too(world):
    corner, n_a, n_b, _ = world
    ate = LandmarkATE.corner("c", corner, n_a, n_b, seed=7)
    probes = ate.rank(_probes(corner, n_a, n_b, [0.4], [0.4]))
    h0 = ate.entropy()
    ate.feed_miss(probes[0])
    assert ate.entropy() <= h0 + 1e-9      # never manufactures uncertainty


def test_cylinder_landmark_builds_and_scores():
    ate = LandmarkATE.cylinder("c1", (2.0, 1.0), 0.30, seed=1)
    apex = np.array([2.0, 1.0 - 0.30])
    pr = [AteProbe("apex", 0.0, apex, np.array([0.0, 1.0])),
          AteProbe("flank", 0.2, apex + np.array([0.2, 0.07]),
                   np.array([-0.55, 0.83]))]
    ranked = ate.rank(pr)
    assert all(np.isfinite(q.score_nats) for q in ranked)
    ate.feed_contact(apex + np.array([0.03, -0.02]), np.array([0.0, -1.0]))
    assert np.isfinite(ate.entropy())
