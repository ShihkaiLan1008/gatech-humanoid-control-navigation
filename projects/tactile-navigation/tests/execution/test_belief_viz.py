"""The belief VISUALISATION, and the frame it is drawn in.

The r1_gallery demo displayed "believed vs TRUE (along-route)" rising past
12 m on an episode that ended `arrived` with a 0.098 m belief error and whose
belief was never worse than 0.496 m. `scripts/audit_belief_viz.py` attributed
it: `ExecutiveV3._snap` records `truth_xy` through `_truth_map()` -- already
map-frame -- while `tick_trace` and `probes` record WORLD truth.  The replay
applied the world->map transform to all three, so the hypothesis samples were
mapped twice.  Rotating an already-mapped point about the spawn by
`pi/2 - yaw0` produces an error that GROWS with distance travelled, which is
exactly what accumulating route drift would look like.

These tests pin the frame contract, so the artefact cannot come back quietly.
Nothing here touches the filter, the planner or the controller.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from lane_graph.replay_v2 import TelemetryV2
from lane_graph.scene3 import build_v3_scene

DEMO = Path("out/demo_v3/r1_gallery/episode.json")


def _episode(base0=(0.31, -0.13), yaw0=-2.1549, n=40, drift=0.05):
    """A synthetic episode whose belief is a fixed `drift` from the truth.

    The world frame is deliberately awkward -- offset origin, non-trivial spawn
    heading -- because that is what makes a double transform visible.
    """
    base0 = np.asarray(base0, float)
    v = np.array([math.cos(yaw0), math.sin(yaw0)])
    u = np.array([v[1], -v[0]])
    tick, hyp = [], []
    for k in range(n):
        s = 0.25 * k                                   # metres along map +x
        xy_map = np.array([s, -0.6])
        xy_w = base0 + xy_map[0] * u + xy_map[1] * v   # the true world pose
        tick.append({"t": 0.5 * k, "mode": "LOCOMOTION",
                     "xy": xy_w.tolist(), "yaw": yaw0})
        bel = xy_map + np.array([drift, 0.0])
        hyp.append({"t": 0.5 * k, "leg": 0, "mode": "WALK",
                    "s_hat": float(bel[0]), "s_sigma": 0.1,
                    "masses": [0.6, 0.4],
                    "hyp_s": [float(bel[0]), float(bel[0]) - 1.0],
                    "hyp_xy": [bel.tolist(), (bel - np.array([1.0, 0.0])).tolist()],
                    "belief_state": "TRACKING", "n_probes": 0,
                    "bel_xy": bel.tolist(), "bel_yaw": math.pi / 2,
                    "truth_xy": xy_map.tolist(), "truth_yaw": math.pi / 2})
    return {"tick_trace": tick, "hypothesis_trace": hyp, "probes": [],
            "directives": [], "arrived": True}


# ── the frame contract ──────────────────────────────────────────────────────────

def test_hypothesis_truth_is_not_mapped_twice():
    """`truth_xy` in the hypothesis trace is ALREADY map-frame."""
    tel = TelemetryV2(_episode(drift=0.05), build_v3_scene("r1_gallery"))
    assert np.allclose(tel.err_xy, 0.05, atol=1e-9)
    assert np.allclose(tel.err_v, 0.05, atol=1e-9)


@pytest.mark.parametrize("yaw0", [0.0, -2.1549, 1.3, math.pi])
@pytest.mark.parametrize("base0", [(0.0, 0.0), (0.31, -0.13), (-4.0, 2.5)])
def test_the_error_does_not_depend_on_the_world_frame(yaw0, base0):
    """A pose error is a property of the belief, not of where the robot
    happened to spawn.  The double transform made it depend on both."""
    tel = TelemetryV2(_episode(base0=base0, yaw0=yaw0, drift=0.08),
                      build_v3_scene("r1_gallery"))
    assert np.allclose(tel.err_xy, 0.08, atol=1e-9)


def test_the_artefact_grew_with_distance_and_the_fix_does_not():
    """Reproduce the old formula and show its signature: an error that ramps
    with travel while the true one stays flat."""
    ep = _episode(drift=0.05)
    tel = TelemetryV2(ep, build_v3_scene("r1_gallery"))
    base0 = np.asarray(ep["tick_trace"][0]["xy"], float)
    yaw0 = float(ep["tick_trace"][0]["yaw"])
    v = np.array([math.cos(yaw0), math.sin(yaw0)])
    u = np.array([v[1], -v[0]])
    old = np.array([float(h["bel_xy"][0])
                    - float((np.asarray(h["truth_xy"], float) - base0) @ u)
                    for h in ep["hypothesis_trace"]])
    assert abs(old[-1] - old[0]) > 5.0          # the old number ramps
    assert np.ptp(tel.err_xy) < 1e-9            # the fixed one does not


def test_euclidean_error_sees_lateral_disagreement():
    """The along-route scalar was blind to any error off the route axis."""
    ep = _episode(drift=0.0)
    for h in ep["hypothesis_trace"]:
        h["bel_xy"] = [h["truth_xy"][0], h["truth_xy"][1] + 0.30]
    tel = TelemetryV2(ep, build_v3_scene("r1_gallery"))
    assert np.allclose(tel.err_xy, 0.30, atol=1e-9)
    assert np.allclose(tel.err_v, 0.0, atol=1e-9)     # x component is zero


# ── the modes the panel draws ───────────────────────────────────────────────────

def test_modes_come_back_in_map_xy_heaviest_first():
    tel = TelemetryV2(_episode(), build_v3_scene("r1_gallery"))
    ms = tel.modes[5]
    assert [m["mass"] for m in ms] == sorted([m["mass"] for m in ms], reverse=True)
    assert ms[0]["mass"] == pytest.approx(0.6)
    assert np.allclose(ms[0]["xy"], tel.bel_map[5])
    assert ms[1]["xy"][0] == pytest.approx(tel.bel_map[5][0] - 1.0)


def test_modes_fall_back_when_an_old_episode_has_no_mode_xy():
    ep = _episode()
    for h in ep["hypothesis_trace"]:
        h.pop("hyp_xy")
    tel = TelemetryV2(ep, build_v3_scene("r1_gallery"))
    ms = tel.modes[5]
    assert len(ms) == 2
    # x from hyp_s, y borrowed from the leading pose so it lands on the lane
    assert ms[0]["xy"][0] == pytest.approx(tel.bel_map[5][0])
    assert ms[0]["xy"][1] == pytest.approx(tel.bel_map[5][1])


def test_executive_logs_mode_xy_consistent_with_mode_s():
    """The new logging field must agree with the one that already existed."""
    import numpy as _np

    from execution.backends import BackendConfig, KinematicBackend
    from lane_graph.executive3 import ExecutiveV3, ExecV3Config
    from lane_graph.graph3 import build_graph3
    from lane_graph.run_v3 import PRESS_TARGET_N

    sc = build_v3_scene("r5_twinbay")
    be = KinematicBackend(sc, BackendConfig(probe_force_n=PRESS_TARGET_N),
                          _np.random.default_rng(0))
    try:
        r = ExecutiveV3(sc, be, build_graph3(sc), ExecV3Config(seed=0)).run()
    finally:
        be.close()
    for h in r["hypothesis_trace"]:
        assert len(h["hyp_xy"]) == len(h["hyp_s"]) == len(h["masses"])
        for xy, x in zip(h["hyp_xy"], h["hyp_s"]):
            assert xy[0] == pytest.approx(x, abs=1e-3)


# ── the episode that started it ─────────────────────────────────────────────────

@pytest.mark.skipif(not DEMO.exists(), reason="r1_gallery demo not rendered here")
def test_r1_gallery_demo_error_matches_the_episodes_own_number():
    ep = json.loads(DEMO.read_text())
    tel = TelemetryV2(ep, build_v3_scene("r1_gallery"))
    assert float(np.max(tel.err_xy)) < 0.60         # panel showed 12.10 m
    assert float(tel.err_xy[-1]) == pytest.approx(ep["global_pos_err_m"], abs=0.05)
