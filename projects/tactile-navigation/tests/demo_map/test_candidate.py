"""One candidate structure, one covariance.

Two unifications, pinned where they are easiest to un-do by accident:

* every geometry generator -- wall station, corner face probe, curved
  landmark bracket, vertex kiss -- compiles to the SAME action structure, so
  the thing that scores a touch and the thing that executes one read one type;
* every planner-facing spread is a VIEW of a single SE(2) covariance, so an
  A/B between two scorers measures the scorers and not their accessors.
"""
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from demo_map.candidate import (TactileCandidate, hand_for_segment,
                                probe_candidate, wall_candidate)
from demo_map.executor import DemoExecCfg, DemoRouteExecutor
from demo_map.landmark_ate import AteProbe
from execution.wall_lane import WallSegment


def _seg(name="w", normal=(0.0, -1.0), tangent=(1.0, 0.0)):
    return WallSegment(p0=np.zeros(2), tangent=np.asarray(tangent, float),
                       normal=np.asarray(normal, float), length=8.0, name=name)


def _executor():
    ex = object.__new__(DemoRouteExecutor)
    ex.cfg = DemoExecCfg()
    ex.scene = SimpleNamespace(plan=SimpleNamespace(bodies=()))
    ex.be = SimpleNamespace(mj=SimpleNamespace(
        vf_touch_target_m=0.43, vf_press_stance_m=0.36,
        probe_hand_z_m=0.25, park_x_m=0.14))
    return ex


FIELDS = ("stance_pose", "end_effector", "tip_target_body",
          "approach_dir_body", "target")


def test_every_generator_compiles_to_the_same_action_structure():
    """A wall station used to be a bare float, a corner probe an `AteProbe`
    whose `.s` was the only field the call site read, and a kiss a hardcoded
    1.2 against a fabricated segment.  Downstream now sees one type."""
    ex = _executor()
    seg = _seg()
    curved = AteProbe("cyl", 1.8, np.array([1.8, 0.0]), np.array([0.0, 1.0]),
                      kind="edge")
    corner = AteProbe("face_a", 2.4, np.array([2.4, 0.0]),
                      np.array([0.0, 1.0]))
    made = {
        "wall": ex._candidate(seg, 2.0),
        "curved": ex._probe_candidate(curved, seg, primitive="curved",
                                      body="bulge_2"),
        "corner_face": ex._probe_candidate(corner, seg),
        "vertex": ex._candidate(seg, 1.2, primitive="corner_vertex",
                                normal_is_defined=False, kind="search"),
    }
    for name, c in made.items():
        assert isinstance(c, TactileCandidate), name
        for f in FIELDS:
            assert getattr(c, f) is not None, (name, f)
        assert len(c.stance_pose) == 3, name           # (x, y, yaw) in MAP
        assert len(c.tip_target_body) == 3, name       # (x, y, z) in BODY
        assert len(c.approach_dir_body) == 2, name
        assert c.end_effector in ("left", "right"), name
        assert c.target.ref is seg, name
    # the geometry-specific part survives: what each one is ABOUT differs
    assert {c.target.primitive for c in made.values()} == {
        "wall_face", "curved", "corner_vertex"}
    assert made["curved"].target.kind == "edge"
    assert made["curved"].target.body == "bulge_2"


def test_a_vertex_declares_that_it_has_no_normal_and_a_face_declares_that_it_has():
    """The kiss lands where two faces meet; there is no single surface normal
    there and the belief must be told so rather than handed a 45-degree
    assumption.  That statement is a field now, not a keyword argument one
    call site out of six remembered to pass."""
    ex = _executor()
    seg = _seg()
    assert ex._candidate(seg, 1.2, normal_is_defined=False) \
        .target.normal_is_defined is False
    assert ex._candidate(seg, 1.2).target.normal_is_defined is True


def test_the_stance_is_in_the_map_and_the_tip_target_is_in_the_body():
    """Mixing the frames is the bug this representation exists to prevent: a
    reach expressed in map coordinates folds the base's own error into a
    quantity the arm controller cannot know."""
    ex = _executor()
    seg = _seg(normal=(0.0, -1.0))          # face along +x, material at y>0
    c = ex._candidate(seg, 3.0)
    x, y, yaw = c.stance_pose
    assert x == pytest.approx(3.0)
    assert y < 0.0, "the stance stands off the face, on the free side"
    # facing the wall: body +x is the INWARD normal, which is +y here
    assert np.allclose(c.approach_map(), [0.0, 1.0], atol=1e-9)
    assert c.approach_dir_body == (1.0, 0.0)
    assert c.tip_target_body[0] == pytest.approx(0.43)   # the arm's reach
    assert c.tip_reach_m == pytest.approx(0.43)


def test_a_stance_may_expose_several_candidates_and_either_arm_but_never_both():
    """`A stance may expose multiple sequential candidates and candidates from
    either arm.  Do not add simultaneous bimanual probing yet.`"""
    ex = _executor()
    left_seg, right_seg = _seg(normal=(0.0, -1.0)), _seg(normal=(0.0, 1.0))
    assert hand_for_segment(left_seg) != hand_for_segment(right_seg)
    # several candidates at ONE stance: the acquisition ladder's rungs are
    # the same station at successively closer stances
    rungs = [ex._candidate(left_seg, 2.0, press_standoff_m=d, kind="search")
             for d in (0.36, 0.325, 0.29)]
    assert len({r.stance_pose for r in rungs}) == 3
    assert {r.target.s_m for r in rungs} == {2.0}
    # ...and one arm each, always: `end_effector` is a name, not a set
    for c in rungs:
        assert isinstance(c.end_effector, str)
    assert not any(f.startswith("hands") or f.endswith("_pair")
                   for f in TactileCandidate.__dataclass_fields__)


# ── one covariance, for every scorer ─────────────────────────────────────────


def test_every_planner_spread_is_a_view_of_one_covariance():
    ex = object.__new__(DemoRouteExecutor)
    cov = np.array([[0.09, 0.01, 0.002],
                    [0.01, 0.04, 0.001],
                    [0.002, 0.001, 0.0025]])
    ex.mb = SimpleNamespace(b=SimpleNamespace(planner_pose_cov=lambda: cov))
    assert np.allclose(ex._cov_xy(), cov[:2, :2])
    assert ex._sigma() == pytest.approx(np.sqrt(0.09 + 0.04))
    assert ex._sigma_yaw() == pytest.approx(np.sqrt(0.0025))


def test_no_scorer_computes_a_covariance_of_its_own():
    """The discrepancy this replaces was two accessors differing by
    1/(1 - sum w^2) -- 0.2% at full ESS and ~10% after a weight collapse,
    i.e. exactly when the numbers matter.  A third moment computed inline
    somewhere would re-open it silently, so the planning layer is asserted to
    contain exactly ONE place that touches the particle arrays."""
    src = (ROOT / "src" / "demo_map").glob("*.py")
    offenders = []
    for path in src:
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"\.b\.(weights|xy|theta)\b", line) \
                    or "pose_covariance()" in line:
                offenders.append(f"{path.name}:{i}: {line.strip()}")
    assert offenders == [], (
        "planner-facing spread must come from `_pose_cov`:\n"
        + "\n".join(offenders))
