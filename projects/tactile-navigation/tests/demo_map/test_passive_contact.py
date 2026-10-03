"""An unasked-for contact is a measurement, and it is worth exactly one.

demo_room v19, t=223-246: the base rode 0.231 m from the second wall against
a 0.276-0.317 m realized parked-hand radius, so a hand was 45-86 mm inside
the face for 23.5 s.  The belief held a 0.394 m wall-normal sigma and a
39%-mass hypothesis standing 0.98 m out -- a pose at which that contact is
geometrically impossible.  The measurement that would have killed that mode
was being felt continuously and collected nowhere.

These pin the two halves of collecting it: the backend emits ONE record per
contact episode (a rub is thousands of ticks of the same contact, and feeding
each one would collapse sigma on a single measurement repeated), and the
executor turns that record into the same `ContactObs` a deliberate tap makes.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from demo_map.executor import DemoExecCfg, DemoRouteExecutor, DemoRun
from execution.mujoco_backend import MujocoBackend, MujocoConfig


def _backend(force_by_hand):
    be = object.__new__(MujocoBackend)
    be.mj = MujocoConfig()
    be._t = 1.0
    be._passive_armed = True
    be._hand_forces = dict(force_by_hand)
    ee = SimpleNamespace(detach=lambda: SimpleNamespace(
        cpu=lambda: SimpleNamespace(numpy=lambda: np.zeros(3))))
    hand = SimpleNamespace(measured_ee_pose_w=lambda: [[ee]])
    be.hands = {"left": hand, "right": hand}
    be._contact_body_frame = lambda ee_w: ((0.30, 0.0), (-1.0, 0.0))
    be.working_pose = lambda: (np.zeros(2), 0.0)
    be.world_point_to_map = lambda p: np.asarray(p, float)
    be.world_yaw_to_map = lambda y: float(y)
    be._believed_contact_map = lambda pb: [0.30, 0.0]
    return be


def test_a_rub_reports_one_contact_and_not_one_per_tick():
    """The wedge lasted 23.5 s at ~50 Hz.  A latch that re-armed on force
    alone every tick would hand the filter the same measurement ~1000 times
    and collapse sigma on evidence it does not have."""
    be = _backend({"left": 8.0, "right": 0.0})
    first = be._passive_contact()
    assert first is not None and first["hand"] == "left"
    assert first["force_n"] == 8.0
    # the hand stays loaded for the rest of the rub: nothing new to report
    for _ in range(50):
        assert be._passive_contact() is None


def test_the_latch_re_arms_only_after_the_hand_comes_off():
    be = _backend({"left": 8.0, "right": 0.0})
    assert be._passive_contact() is not None
    # still touching, below the report threshold but above the release one
    be._hand_forces = {"left": 3.0, "right": 0.0}
    assert be._passive_contact() is None
    be._hand_forces = {"left": 0.5, "right": 0.0}      # off the wall
    assert be._passive_contact() is None
    be._hand_forces = {"left": 9.0, "right": 0.0}      # a NEW contact
    assert be._passive_contact() is not None


def test_a_light_graze_is_not_a_contact():
    """`wp_tap_contact_n` (2 N) is where a touch begins to exist and
    `wp_feeler_touch_n` (4 N) is where the hand has demonstrably MET the
    wall.  Under the SensX model a pure air sweep peaks at 2.5-3.2 N, so
    reporting at the lower threshold would feed the filter its own noise."""
    be = _backend({"left": 3.0, "right": 0.0})
    assert be._passive_contact() is None


def test_a_retract_disarms_the_latch():
    """Every deliberate press exits through `_retract`, and the hand can
    still be loaded when it does.  The belief has already absorbed that
    touch; it must not also receive it as an accident."""
    be = _backend({"left": 8.0, "right": 0.0})
    be._set_goal = lambda *a, **k: None
    be._park_hand = lambda *a, **k: None
    be.shield = SimpleNamespace(set_normal=lambda *a, **k: None,
                                close_window=lambda *a, **k: None)
    be.probe_term = SimpleNamespace(pose_command_b=np.zeros((1, 7)))
    be.park_term = None
    be._step = lambda: None
    be._guard_tripped = lambda: True          # exit the retract loop at once
    be._retract()
    assert be._passive_armed is False
    assert be._passive_contact() is None


def _executor():
    ex = object.__new__(DemoRouteExecutor)
    ex.cfg = DemoExecCfg()
    ex.cfg.touch_noise = False
    ex.run_rec = DemoRun()
    ex.noise_rng = np.random.default_rng(0)
    ex.be = SimpleNamespace(
        time_s=2.0,
        working_pose=lambda: (np.array([5.0, 2.0]), 0.0),
        world_point_to_map=lambda p: np.asarray(p, float),
        world_yaw_to_map=lambda y: float(y))
    ex._sigma = lambda: 0.3
    return ex


def test_the_passive_record_and_a_tap_make_the_same_observation():
    """One contact model, not two.  The passive record carries the same
    fields a tap does -- a hand pose from the encoders and the pose the robot
    BELIEVED it held -- so it must compile to the same `ContactObs`."""
    ex = _executor()
    fields = {"p_body": [0.3, 0.0], "n_body": [-1.0, 0.0],
              "press_pose_map": [5.0, 2.0, 0.0]}
    passive = ex._contact_of(dict(fields))
    tap = ex._contact_of(dict(fields, contacted=True))
    assert passive is not None
    assert passive.p_body == tap.p_body
    assert passive.normal_body == tap.normal_body
    assert passive.signature == tap.signature


def test_absorbing_a_passive_contact_feeds_a_contact_and_no_free_space():
    """A rub sweeps no volume.  Claiming free space from it would assert the
    robot proved emptiness exactly where it just proved the opposite."""
    ex = _executor()
    seen = {}
    ex._absorb = lambda event, observed=None: seen.update(
        event=event, observed=observed)
    ex._on_passive_contact({"p_body": [0.3, 0.0], "n_body": [-1.0, 0.0],
                            "press_pose_map": [5.0, 2.0, 0.0],
                            "hand": "left", "force_n": 8.0})
    assert seen["event"] == "passive_contact"
    assert len(seen["observed"]["contacts"]) == 1
    assert list(seen["observed"]["frees"]) == []
    row = ex.run_rec.replans[-1]
    assert row["reason"] == "passive_contact_absorbed"
    assert row["force_n"] == 8.0


# ── the wiring, at the layer the demo actually runs ──────────────────────────

ROOT = Path(__file__).resolve().parents[2]


def _src_of(src: str, name: str) -> str:
    i = src.index(f"def {name}")
    j = src.find("\n    def ", i + 1)
    return src[i:j if j > 0 else len(src)]


def test_the_walk_the_demo_actually_runs_is_the_one_that_collects():
    """The first cut put this harvest in `MujocoBackend.wall_parallel_touch`.
    `VForceBackend` -- the controller every demo runs -- OVERRIDES that method
    without calling super, so the collection ran nowhere: demo_room v20w
    recorded zero passive contacts through a whole episode.  The harvest
    belongs in the walk the shipped backend actually executes, which is
    `_walk_to_lane`."""
    src = (ROOT / "src" / "execution" / "vforce_backend.py").read_text()
    walk = _src_of(src, "_walk_to_lane")
    assert "self._passive_contact()" in walk
    assert "passive_contact_sink" in walk
    assert "target_w = None" in walk, "a moved belief makes the aim stale"
    # ...and the deliberate press disarms the latch, so a tap the belief has
    # already absorbed cannot arrive again as an accident
    assert "self._passive_armed = False" in src


def test_the_sink_exists_on_every_backend_however_it_builds_itself():
    """`VForceBackend.__init__` does not call `super().__init__`, so the
    contract cannot live in the base initializer: `hasattr` returned False
    and the executor silently skipped installing the sink."""
    from execution.mujoco_backend import MujocoBackend
    assert "passive_contact_sink" in vars(MujocoBackend)
    assert MujocoBackend.passive_contact_sink is None
    assert vars(MujocoBackend)["_passive_armed"] is True

    class Bare(MujocoBackend):
        def __init__(self):          # deliberately no super().__init__
            pass

    be = Bare()
    assert hasattr(be, "passive_contact_sink")
    assert be.passive_contact_sink is None
    assert be._passive_armed is True


def test_a_rub_the_base_stood_still_through_is_one_measurement():
    """demo_room v20 absorbed 11 passive contacts that were 4 contact events:
    the sensed force of a rub crosses the 2 N release and the 4 N report
    thresholds repeatedly, so a force-only latch re-arms on noise.  Across
    one burst of SEVEN in 1.3 s sigma fell 0.432 -> 0.073 on a single patch
    of wall, and the two updates after it grew the error while shrinking
    sigma further.  Whether a contact is new INFORMATION is a belief
    question, so the executor answers it: below the travel at which odometry
    drift reaches the contact sensor's own sigma, the filter cannot tell two
    contacts apart."""
    ex = _executor()
    absorbed = []
    ex._absorb = lambda event, observed=None: absorbed.append(event)
    base = {"p_body": [0.3, 0.0], "n_body": [-1.0, 0.0],
            "press_pose_map": [5.0, 2.0, 0.0], "hand": "left",
            "force_n": 8.0}
    floor = ex._passive_travel_floor_m()
    assert 0.10 < floor < 0.30                     # 0.018 / 0.104
    ex._on_passive_contact({**base, "odom_travel_m": 10.0})
    for extra in (0.0, 0.01, 0.05, floor * 0.99):  # the burst, base still
        ex._on_passive_contact({**base, "odom_travel_m": 10.0 + extra})
    assert len(absorbed) == 1, "a rub is one measurement"
    ex._on_passive_contact({**base, "odom_travel_m": 10.0 + floor + 1e-6})
    assert len(absorbed) == 2, "...and a new patch of wall is a new one"
    dropped = [r for r in ex.run_rec.replans
               if r["reason"] == "passive_contact_same_patch"]
    assert len(dropped) == 4 and dropped[0]["force_n"] == 8.0


def test_the_travel_mark_cannot_be_moved_by_a_localizer_correction():
    """The counter is accumulated from per-tick odometry inside
    `_integrate_estimator` and nowhere else, so `reset_estimator_to` -- which
    is how every belief correction reaches the estimator -- cannot make the
    robot look like it walked."""
    src = (ROOT / "src" / "execution" / "mujoco_backend.py").read_text()
    assert "_odom_travel" in _src_of(src, "_integrate_estimator")
    assert "_odom_travel" not in _src_of(src, "reset_estimator_to")
