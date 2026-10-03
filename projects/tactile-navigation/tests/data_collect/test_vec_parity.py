"""The batched collector must be the single-world one, faster -- not different.

`vec_env`/`vec_touch`/`vec_collector` cannot be exercised without mjlab, so
what is pinned here is the part that would silently diverge: the physical
constants of the press, the ordering of its phases, and the fact that the two
collectors share one sampling implementation rather than two copies of it.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "data_collect"
VEC_TOUCH = (SRC / "vec_touch.py").read_text()
VEC_ENV = (SRC / "vec_env.py").read_text()
VEC_COL = (SRC / "vec_collector.py").read_text()
PROBE = (SRC / "probe.py").read_text()


def _consts(src: str) -> dict:
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:                   # noqa: BLE001
                pass
    return out


def test_the_two_presses_use_the_same_timings():
    """A dwell or a hold that differs between the two paths would make the
    dataset's labels depend on how many worlds it was collected in."""
    a, b = _consts(PROBE), _consts(VEC_TOUCH)
    for k in ("TARGET_DWELL_S", "CONTACT_HOLD_S", "PRESS_MARGIN_S"):
        assert a[k] == b[k], k
    assert b["SETTLE_CAP_S"] == a["SETTLE_CAP_S"]


def test_the_batched_press_takes_its_physics_from_the_config():
    """Not restated as literals: the creep rate, the intent, the debounce, the
    contact threshold and the back-off all come from `VForceConfig`."""
    for knob in ("vf_press_rate_m_s", "vf_press_intent_n", "wp_tap_contact_n",
                 "contact_debounce_samples", "vf_contact_backoff_m",
                 "vf_release_s"):
        assert knob in VEC_TOUCH, knob
    assert "vf_press_intent_n" in VEC_ENV


def test_the_batched_press_keeps_the_phase_order():
    """settle -> creep -> (contact hold | target dwell) -> retract, with the
    back-off applied the instant the latch fires."""
    i_settle = VEC_TOUCH.index("if s.phase == SETTLE:")
    i_creep = VEC_TOUCH.index("if s.phase == CREEP:")
    i_hold = VEC_TOUCH.index("if s.phase == CONTACT_HOLD:")
    i_dwell = VEC_TOUCH.index("if s.phase == TARGET_DWELL:")
    i_ret = VEC_TOUCH.index("if s.phase == RETRACT:")
    assert i_settle < i_creep < i_hold < i_dwell < i_ret
    creep = VEC_TOUCH[i_creep:i_hold]
    assert "s.contacted = True" in creep
    assert "s.u - mj.vf_contact_backoff_m" in creep
    assert creep.index("s.contacted = True") < creep.index(
        "s.u - mj.vf_contact_backoff_m")


def test_the_guard_keeps_its_teeth_per_environment():
    """`_safety_stop` freezes a hold, backs the BASE off along the measured
    reaction force, and waits until clear AND still.  One env tripping must
    not stall the other 127, and must not be answered by parking a hand."""
    assert "SAFETY_BACKOFF" in VEC_TOUCH and "SAFETY_CLEAR" in VEC_TOUCH
    fn = VEC_TOUCH[VEC_TOUCH.index("def _enter_safety"):
                   VEC_TOUCH.index("def _enter_retract")]
    # the retreat is the NEGATIVE of the hand-on-surface force
    assert "- SAFETY_BACKOFF_M * f / n" in fn
    assert "guard_abort_force_n" in VEC_TOUCH
    assert "guard_abort_hold_s" in VEC_TOUCH
    assert "safety_clear_force_n" in VEC_TOUCH
    clear = VEC_TOUCH[VEC_TOUCH.index("if s.phase == SAFETY_CLEAR:"):]
    assert "still" in clear and "stable_since" in clear


def test_the_parked_hand_is_the_guarded_one():
    """In PROBE the pressing hand is exempt -- it is regulated against its
    intent -- but the PARKED hand has no business touching anything."""
    assert 'park_hand = "right" if s.hand == "left" else "left"' in VEC_TOUCH
    assert "f_total[park_hand][ei]" in VEC_TOUCH


def test_the_batched_collector_reuses_the_sampling_code():
    """`_draw_pose`, `propose_target`, `_capability` and the coverage quotas
    must not be reimplemented, or the two datasets are sampled differently."""
    tree = ast.parse(VEC_COL)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
               and n.name == "VecCollector")
    assert [b.id for b in cls.bases] == ["Collector"]
    defined = {f.name for f in cls.body if isinstance(f, ast.FunctionDef)}
    for shared in ("_draw_pose", "propose_target", "_capability",
                   "_pick_by_deficit", "_episode_meta", "_action_record",
                   "_wanted_outcome"):
        assert shared not in defined, f"{shared} was reimplemented"


def test_the_batched_collector_keeps_the_belief_contract():
    """One PF per segment, one ordinary update per valid touch, and nothing
    folded in on an invalid attempt."""
    fn = VEC_COL[VEC_COL.index("def _on_touch_end"):
                 VEC_COL.index("def _odom")]
    assert 'if out.label != "invalid_attempt":' in fn
    assert "s.belief.predict(" in fn and "s.belief.update(" in fn
    assert fn.index("s.belief.predict(") < fn.index("s.belief.update(")
    # b_{t+1} is absent on an invalid attempt, and the row is not advanced
    assert '"belief_after": (None if nxt is None' in fn
    assert "return" not in fn.split("self.cov.add_invalid")[1][:120]


def test_the_batched_env_never_walks():
    """No lane, no station, no goto: a base is placed and held."""
    from test_no_planner_in_the_loop import symbols
    names = symbols(SRC / "vec_env.py") | symbols(SRC / "vec_touch.py")
    for banned in ("_walk_to_lane", "wall_parallel_touch", "goto_map",
                   "turn_to_map", "lane_point", "wall_follow_map",
                   "tap_stations", "WallSegment"):
        assert banned not in names, banned
    assert "def hold(" in VEC_ENV
    assert "arrived[:] = True" in VEC_ENV


def test_the_stale_wall_index_is_cleared_in_both_paths():
    """`maze_twist.follow_wall` is chosen at reset from the SIX maze walls and
    indexes `wall_map.hl` every tick; Room 2 installs four.  Measured: fine at
    8 environments, a device-side assert at 32."""
    vf = (Path(__file__).resolve().parents[2] / "src" / "execution"
          / "vforce_backend.py").read_text()
    for src, name in ((VEC_ENV, "vec_env"), (vf, "vforce_backend")):
        assert 'hasattr(self.fsm, "follow_wall")' in src, name
        assert "self.fsm.follow_wall[:] = 0" in src, name


@pytest.mark.parametrize("mod", ["vec_env.py", "vec_touch.py",
                                 "vec_collector.py"])
def test_the_batched_path_has_no_planner_either(mod):
    from test_no_planner_in_the_loop import (PLANNER_MODULES, PLANNER_NAMES,
                                             symbols)
    s = symbols(SRC / mod)
    for m in PLANNER_MODULES:
        assert not any(x == m or x.startswith(m + ".") for x in s), m
    for fn in PLANNER_NAMES:
        assert fn not in s, fn
