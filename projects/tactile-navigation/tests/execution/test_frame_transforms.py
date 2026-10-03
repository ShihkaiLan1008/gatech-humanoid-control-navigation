"""The map<->world frame pair, and the harness that has to use both halves.

`PF_AUDIT.md` 4 reported that on MuJoCo the contact likelihood scored the true
pose 144 nats BELOW a pose 0.3 m away, and read that as the injected geometry
being misregistered by half a metre.  It was not.  `scripts/pf_audit.py` mapped
the truth POSITION into the belief's frame and handed over the truth YAW in
world.  On `r3_tpinch` the spawn heading is yaw0 = -0.302 rad, so every truth
pose the recorder scored was the real one rotated about itself by

    pi/2 - yaw0 = +1.873 rad = 107.3 deg

and a contact reached 0.3-0.4 m in front of the base therefore landed
2 |p| sin(alpha/2) = 0.48-0.62 m from where the hand actually touched -- the
exact residual band the audit reported.  The kinematic backend hid it: world IS
map there, so the missing conversion is the identity.

These tests pin the inverse pair on both backends and pin that the harness uses
it.  They are pure geometry: no simulator, no episode.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from execution.mujoco_backend import MujocoBackend  # noqa: E402

# r3_tpinch, seed 0: recovered exactly (0.00000 m fit residual) from the six
# contact updates of out/pf_audit/v3_r3_mj2 by Procrustes on (world, map) pairs.
R3_YAW0 = -0.302163
R3_ALPHA = math.pi / 2.0 - R3_YAW0          # +1.873 rad, the missing rotation


def _stub(yaw0: float) -> SimpleNamespace:
    """Enough of a MujocoBackend for the two yaw transforms (they read yaw0)."""
    return SimpleNamespace(yaw0=yaw0)


@pytest.mark.parametrize("yaw0", [0.0, R3_YAW0, 1.2, -2.9, math.pi])
@pytest.mark.parametrize("yaw_map", [0.0, math.pi / 2, -1.1, 2.7, 3.0])
def test_yaw_transforms_are_inverses(yaw0, yaw_map):
    be = _stub(yaw0)
    w = MujocoBackend.map_yaw_to_world(be, yaw_map)
    back = MujocoBackend.world_yaw_to_map(be, w)
    assert abs(math.remainder(back - yaw_map, 2 * math.pi)) < 1e-12


def test_start_heading_is_map_half_pi():
    """The map frame is DEFINED by the spawn pose: +y is the start heading."""
    be = _stub(R3_YAW0)
    assert abs(math.remainder(
        MujocoBackend.world_yaw_to_map(be, R3_YAW0) - math.pi / 2, 2 * math.pi)) < 1e-12


def test_unmapped_yaw_swings_a_reached_contact_off_the_wall():
    """The defect, quantified: the size of the error it manufactures.

    Reconstruct one body-frame contact at the true pose and at the same pose
    with the yaw left in world.  The gap must reproduce the 0.48-0.62 m band
    PF_AUDIT.md 4 attributed to geometry.
    """
    from global_nav.geometry import se2_apply

    xy_map = np.array([[-0.728, 1.830]])        # update 0 of v3_r3_mj2
    yaw_w = 1.4294                              # what the harness passed
    yaw_map = MujocoBackend.world_yaw_to_map(_stub(R3_YAW0), yaw_w)

    for reach in (0.30, 0.35, 0.40):
        p_body = np.array([reach, 0.0])
        good = se2_apply(xy_map, np.array([yaw_map]), p_body)[0]
        bad = se2_apply(xy_map, np.array([yaw_w]), p_body)[0]
        gap = float(np.linalg.norm(good - bad))
        assert abs(gap - 2 * reach * math.sin(R3_ALPHA / 2)) < 1e-9
        assert 0.45 < gap < 0.70, f"reach {reach} -> {gap:.3f} m"


def test_kinematic_stand_in_yaw_transform_is_the_identity():
    from execution.backends import BackendConfig, KinematicBackend
    from execution.scenes import build_scene

    be = KinematicBackend(build_scene("ambiguous_near_edge"), BackendConfig(),
                          np.random.default_rng(0))
    _, yaw = be.truth_pose()
    assert be.world_yaw_to_map(yaw) == pytest.approx(yaw)
    assert be.map_yaw_to_world(yaw) == pytest.approx(yaw)
    assert be.world_yaw_to_map(be.map_yaw_to_world(0.7)) == pytest.approx(0.7)


def test_audit_harness_maps_both_halves_of_the_truth_pose():
    """The regression that matters: _install_recorder must convert the yaw."""
    from scripts.pf_audit import _install_recorder

    class FakeBackend:
        yaw0 = R3_YAW0
        time_s = 0.0

        def truth_pose(self):
            return np.array([1.5432, 0.3024]), 1.4294

        world_point_to_map = MujocoBackend.world_point_to_map
        world_yaw_to_map = MujocoBackend.world_yaw_to_map

        base0 = np.array([-0.4206, 0.1522])
        u = np.array([math.sin(R3_YAW0), -math.cos(R3_YAW0)])
        v = np.array([math.cos(R3_YAW0), math.sin(R3_YAW0)])

    be = FakeBackend()
    mb = SimpleNamespace(b=SimpleNamespace(diagnostics=None))
    diag = _install_recorder(mb, be, lambda: 0)

    xy, yaw = diag.truth_fn()
    assert np.allclose(xy, [-0.728, 1.830], atol=2e-3)      # position, as before
    assert yaw == pytest.approx(-2.9808, abs=2e-3)          # yaw, now mapped too
    assert abs(math.remainder(yaw - 1.4294, 2 * math.pi)) > 1.8   # not world yaw
