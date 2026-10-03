"""Every force a decision reads must come through the hardware model."""
import ast
import pathlib


def _decision_reads_are_sensed():
    src = pathlib.Path("src/execution/mujoco_backend.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_probe_force_n")
    calls = {c.func.attr for c in ast.walk(fn)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)}
    return "_sensed_force_n" in calls


def test_the_press_latch_reads_a_force_hardware_can_supply():
    """`_probe_force_n` feeds `contacted = axis_n >= contact_threshold_n` and
    the press-mean sum.  Raw MuJoCo contact force is exact, unbounded and at
    the physics rate; SensX is 0-20 N, sigma 1 N, ~100 Hz.  The docstring
    claimed the model was applied and the code did not apply it."""
    assert _decision_reads_are_sensed()


def test_the_model_saturates_and_holds():
    import sys
    sys.path.insert(0, "src")
    from execution.mujoco_backend import MujocoBackend, MujocoConfig

    b = MujocoBackend.__new__(MujocoBackend)
    b.mj = MujocoConfig()
    b._t = 0.0
    import numpy as np
    b.rng = np.random.default_rng(0)
    assert b._sensed_force_n(100.0, key="k") <= b.mj.force_sat_n + 4.0
    v = b._sensed_force_n(3.0, key="h")
    b._t = 0.001                                   # inside one 100 Hz sample
    assert b._sensed_force_n(50.0, key="h") == v   # held, not instantaneous
    # ...and there is no raw path left to fall back to: the model is the
    # only one, so even a fresh key saturates rather than passing 100 N
    # through (the flag that used to bypass it is gone).
    assert b._sensed_force_n(100.0, key="fresh") <= b.mj.force_sat_n + 4.0


def test_the_wrist_does_not_aim_itself_with_the_simulators_answer():
    """`_face_quat` yawed the hand toward the face using `heading_w` -- the
    TRUE base heading -- so the hand was steered by the very quantity the
    felt normal is supposed to help estimate.  Under honest sensing the
    command must come from the believed heading."""
    import inspect
    import sys
    sys.path.insert(0, "src")
    from execution.mujoco_backend import MujocoBackend

    src = inspect.getsource(MujocoBackend._face_quat)
    assert "working_pose()" in src
    body = src.split('"""')[-1]          # the code, not the docstring
    assert "heading_w" not in body


def test_a_railed_wrist_announces_that_it_stopped_measuring():
    """The wrist may yaw only +-face_quat_clamp_rad toward the face, so once
    the base sits further off square than that, the reading is the wrist's
    own stop rather than the wall.  Measured: base 36 deg off square, wrist
    railed at 6.9 deg, filter concluded the BASE was 6.9 deg off and rotated
    the belief the 28 deg difference."""
    import math
    import sys
    sys.path.insert(0, "src")
    import numpy as np
    from execution.mujoco_backend import MujocoBackend, MujocoConfig

    b = MujocoBackend.__new__(MujocoBackend)
    b.mj = MujocoConfig()
    lim = b.mj.face_quat_clamp_rad

    b.mj = MujocoConfig(measured_normal_axis="x", measured_normal_sign=1)
    b.truth_pose = lambda: (np.zeros(2), 0.0)

    def railed(off_rad):
        # the wrist reports the face's outward normal, nominally straight
        # back along the body -x; `off_rad` is how far off that it points
        b._wrist_axes_body = lambda: {
            "x": [-math.cos(off_rad), -math.sin(off_rad)]}
        _p, _n = b._contact_body_frame(np.zeros(3))
        return b.last_normal_railed

    assert not railed(0.0)
    assert not railed(lim * 0.5)
    assert railed(lim)


def test_the_force_channel_survives_a_clock_that_restarts():
    """`install_substep_monitor` makes a new counter on every install -- once
    per leg, once per hand swap -- so the substep clock drops back to zero
    mid-run.  A hold keyed on "has an interval elapsed?" can then never fire
    again and the channel freezes.  Measured cost: every press after the
    first leg change reported peak 0.0 N, and the in-press brake and 25 N
    abort were reading that same frozen constant."""
    import sys
    sys.path.insert(0, "src")
    import numpy as np
    from execution.mujoco_backend import MujocoBackend, MujocoConfig

    b = MujocoBackend.__new__(MujocoBackend)
    b.mj = MujocoConfig()
    b._t = 0.0
    b.rng = np.random.default_rng(0)

    for i in range(50):                       # run the clock well forward
        b._sensed_force_n(0.0, key="sub", t=0.02 * i, latency_s=0.0)
    # ... then restart it, as a reinstall does
    seen = [b._sensed_force_n(18.0, key="sub", t=0.02 * i, latency_s=0.0)
            for i in range(20)]
    assert max(seen) > 10.0, "the channel went dead after the clock reset"


def test_the_summary_notices_a_channel_that_stopped_updating():
    """The existing assert pairs `peak_force_n` against the audit peak, but
    both are fed from the same samples: freeze the channel and they agree at
    zero.  Deadness needs its own check."""
    import inspect
    import sys
    sys.path.insert(0, "src")
    from execution.mujoco_backend import MujocoBackend

    src = inspect.getsource(MujocoBackend.physical_summary)
    assert "_force_samples" in src
    assert "sub_probe" in src


def test_air_does_not_latch_as_contact():
    """`contact_threshold_n` is 2 N and SensX noise is sigma 1 N, so a single
    sample crosses it on 2.3 % of reads of PURE AIR.  While the latch read raw
    MuJoCo force (air = exactly 0.0) that never showed; routing it through the
    sensor model -- which is what hardware hands it -- makes a one-sample
    test on a 2 N threshold meaningless.  Measured before the fix: a press
    whose base stood 0.62 m from the wall, far beyond any reach, returned
    `bounded_press_contact` at 2.8 N."""
    import sys
    sys.path.insert(0, "src")
    import numpy as np
    from execution.mujoco_backend import MujocoBackend, MujocoConfig

    b = MujocoBackend.__new__(MujocoBackend)
    b.mj = MujocoConfig()
    b.cfg = type("C", (), {"contact_threshold_n": 2.0})()
    b._t = 0.0
    b.rng = np.random.default_rng(0)

    false_latches = 0
    for press in range(200):
        b._above_thresh = 0
        for i in range(150):                       # a press-length sweep of AIR
            b._t = 0.02 * i
            f = b._sensed_force_n(0.0, key=("air", press), t=b._t, latency_s=0.0)
            if b._contact_debounced(f):
                false_latches += 1
                break
    assert false_latches <= 4, f"{false_latches}/200 air presses latched"

    # and a REAL sustained contact must still latch, quickly
    b._above_thresh = 0
    n = next(i for i in range(20)
             if b._contact_debounced(b._sensed_force_n(
                 14.0, key="real", t=100.0 + 0.02 * i, latency_s=0.0)))
    assert n <= b.mj.contact_debounce_samples
def test_the_true_peak_says_which_hand_made_it():
    """A run-wide maximum of 41-43 N could not be attributed, and the two
    candidates need opposite fixes: a PROBE peak means our own touch lands
    too hard (a closing-speed problem), a PARK peak means a carried hand
    collided while walking (a lane problem).  The sensed channel cannot tell
    them apart either -- SensX saturates at 20 N, so both read ~21."""
    import inspect
    import sys
    sys.path.insert(0, "src")
    from execution.mujoco_backend import MujocoBackend

    sub = inspect.getsource(MujocoBackend._on_substep)
    assert "true_peak_probe_n" in sub and "true_peak_park_n" in sub
    assert '"hand": "probe" if total_p >= total_k else "park"' in sub
    summ = inspect.getsource(MujocoBackend.physical_summary)
    for k in ("true_peak_probe_n", "true_peak_park_n", "true_peak_detail"):
        assert k in summ, k
