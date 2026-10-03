"""P4: one J across a leg's touches -- the pooling and the ordering.

The contexts themselves are exercised through `_ate_pillar` (stub-driven in
test_landmark_ate_loop) and the sim runs; what THESE tests pin is the new
brain: which steps pool into a leg, and who presses next once they have."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import demo_map.executor as X
from demo_map.executor import DemoExecCfg, DemoRouteExecutor, DemoRun


def _bare_executor():
    ex = object.__new__(DemoRouteExecutor)
    ex.cfg = DemoExecCfg()
    ex.run_rec = DemoRun()
    ex.be = SimpleNamespace(
        time_s=0.0,
        world_point_to_map=lambda p: np.zeros(2),
        truth_pose=lambda: (np.zeros(3), 0.0))
    ex.scene = SimpleNamespace(goal_map_xy=(0.0, 0.0), goal_radius_m=0.0)
    ex._believed_xy = lambda: np.zeros(2)
    ex._sigma = lambda: 0.0            # the arrival gate reads the spread too
    return ex


def _steps():
    return [{"kind": "goto", "p": np.zeros(2), "yaw": 0.0, "label": "in"},
            {"kind": "follow", "seg": "SEG", "label": "w:rhythm"},
            {"kind": "ate_pillar", "label": "post", "centre": (0, 0)},
            {"kind": "goto", "p": np.zeros(2), "yaw": 0.0, "label": "out"}]


def test_a_follow_and_its_visit_pool_into_one_leg():
    ex = _bare_executor()
    pooled = []
    ex._leg_eig = lambda chunk: (pooled.append(
        [c["kind"] for c in chunk]) or ("ok", ""))
    ex._goto = lambda *a, **k: "arrived"
    ex.run(_steps())
    assert pooled == [["follow", "ate_pillar"]]
    assert ex.run_rec.outcome in ("arrived", "arrived_unconfirmed",
                                  "route_done_off_goal")


def test_the_pool_needs_both_kinds_and_an_eig_follow():
    # a lone follow, a lone visit, or an exact-stations follow all run the
    # sequential handlers -- the arbiter only exists where there is a choice
    ex = _bare_executor()
    ex._leg_eig = lambda chunk: (_ for _ in ()).throw(AssertionError("pooled"))
    ex._goto = lambda *a, **k: "arrived"
    seen = []
    ex._follow_eig = lambda seg, label, **k: (seen.append(label) or "arrived")
    ex._follow = lambda *a, **k: seen.append("exact") or "arrived"
    ex._ate_pillar = lambda st: seen.append(st["label"]) or "ok"
    steps = _steps()
    steps[1]["exact_stations"] = True
    steps[1]["stations"] = []
    ex.run(steps)
    assert seen == ["exact", "post"]

    ex2 = _bare_executor()
    ex2.cfg.eig_arbiter = False
    ex2._leg_eig = ex._leg_eig
    ex2._goto = lambda *a, **k: "arrived"
    seen2 = []
    ex2._follow_eig = lambda seg, label, **k: (seen2.append(label)
                                               or "arrived")
    ex2._ate_pillar = lambda st: seen2.append(st["label"]) or "ok"
    ex2.run(_steps())
    assert seen2 == ["w:rhythm", "post"]


def test_a_fatal_from_the_leg_becomes_the_outcome():
    ex = _bare_executor()
    ex._goto = lambda *a, **k: "arrived"
    ex._leg_eig = lambda chunk: ("gate_failed", "post")
    ex.run(_steps())
    assert ex.run_rec.outcome == "gate_failed:post"


class _ScriptedWall:
    PRIORITY = float("inf")

    def __init__(self, js):
        self.js = list(js)          # a J per propose; None closes
        self.label = "wall"
        self.closed = False
        self.executed = []

    def propose(self):
        j = self.js.pop(0) if self.js else None
        if j is None:
            self.closed = True
        return j

    def execute(self):
        self.executed.append("press")
        return "ok"

    def close(self):
        self.closed = True
        return "arrived"


class _ScriptedVisit:
    def __init__(self, js):
        self.js = list(js)          # (j, why) per propose; (None,"x") closes
        self.label = "visit"
        self.closed = False
        self.executed = []

    def propose(self):
        j, why = self.js.pop(0) if self.js else (None, "eig_below_threshold")
        if j is None and why != "acquiring":
            self.closed = True
        return j, why

    def execute(self):
        self.executed.append("press")
        return "ok"

    def close(self):
        self.closed = True
        return "ok"


def _run_leg(ex, wall, visit, monkeypatch):
    monkeypatch.setattr(X, "WallCtx", lambda *a, **k: wall)
    monkeypatch.setattr(X, "VisitCtx", lambda *a, **k: visit)
    # isinstance checks in _leg_eig must see the scripted classes
    monkeypatch.setattr(X.WallCtx, "PRIORITY", float("inf"), raising=False)
    return ex._leg_eig([{"kind": "follow", "seg": None, "label": "wall"},
                        {"kind": "ate_pillar", "label": "visit"}])


def test_the_highest_j_presses_next_and_a_tie_goes_to_the_wall(monkeypatch):
    ex = _bare_executor()
    order = []
    wall = _ScriptedWall([1.0, 2.0, None])
    visit = _ScriptedVisit([(1.5, ""), (2.0, ""), (None, "done")])
    wall.execute = lambda: order.append("wall") or "ok"
    visit.execute = lambda: order.append("visit") or "ok"
    st = _run_leg(ex, wall, visit, monkeypatch)
    # round 1: visit 1.5 beats wall 1.0; round 2: tie at 2.0 -> wall
    assert order[:2] == ["visit", "wall"]
    assert st == ("ok", "")


def test_a_pending_recovery_preempts_a_better_scored_visit(monkeypatch):
    ex = _bare_executor()
    order = []
    wall = _ScriptedWall([float("inf"), None])
    visit = _ScriptedVisit([(99.0, ""), (None, "done")])
    wall.execute = lambda: order.append("recover") or "ok"
    visit.execute = lambda: order.append("visit") or "ok"
    _run_leg(ex, wall, visit, monkeypatch)
    assert order[0] == "recover"


def test_acquiring_runs_only_when_nothing_scores(monkeypatch):
    ex = _bare_executor()
    order = []
    wall = _ScriptedWall([1.0, None])
    visit = _ScriptedVisit([(None, "acquiring"), (None, "acquiring"),
                            (None, "done")])
    wall.execute = lambda: order.append("wall") or "ok"
    visit.execute = lambda: order.append("acquire") or "ok"
    _run_leg(ex, wall, visit, monkeypatch)
    assert order == ["wall", "acquire"]
