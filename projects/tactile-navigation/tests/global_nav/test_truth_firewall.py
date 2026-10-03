"""Subtask 1 -- no planner decision may read simulator ground truth.

The existing freeze checks guard `Executive._update_global` and
`Executive.probe_station`.  This file widens the guard to EVERY method that
participates in a route decision, on BOTH executives, because a leak anywhere
in the decision chain makes every benchmark number meaningless: a planner that
can see truth cannot demonstrate that tactile evidence corrected anything.

Truth is legitimately used for three things, and only these:
  * the simulator itself (backends must know where the robot really is);
  * logging / plotting (`_snap`, `_sample_truth`, path-length accounting);
  * scoring after the fact (wrong-turn checks, final error).
Everything that CHOOSES is fenced off.
"""

from __future__ import annotations

import inspect
import re

import pytest

from execution import executive as ex_mod
from lane_graph import executive3 as v3_mod

# Reading the robot's true pose, or converting a true world point into the map
# frame, are the two ways truth has actually leaked into a decision here before.
TRUTH_TOKENS = ("truth_pose", "_truth_map", "world_point_to_map", "_last_truth")


def _code_only(fn) -> str:
    """Source with docstring and comments stripped: prose may discuss truth."""
    src = inspect.getsource(fn)
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    return "\n".join(line.split("#")[0] for line in src.splitlines())


# Every method that selects a target, a station, a probe, a speed, or a turn.
DECISION_METHODS = [
    ("Executive." + n, getattr(ex_mod.Executive, n)) for n in (
        "plan", "_plan_inner", "_choose_station", "_maintenance_due",
        "_maintenance_station", "_fan_budget", "_believed_xy", "_believed_yaw",
        "_tracked_s", "_tracked_sigma", "_hypotheses", "_believed_standoff",
        "_bias_vector", "_hyp_min_clearance_m", "_clear_translation_target",
        "_governed_speed", "_ensure_turn_clearance", "_brackets_terminus",
        "_anchors_along", "_station_gate_met", "_fallback_candidate",
        "_update_global", "_at_goal",
    )
] + [
    ("ExecutiveV3." + n, getattr(v3_mod.ExecutiveV3, n)) for n in (
        "_hyps", "_believed_xy", "_believed_yaw", "_bel", "_relatch",
        "_stance", "_choose_offset", "_ray", "_corner_verified", "_plan",
        "_nearest_node", "_speed", "_hands_clear",
    )
]


@pytest.mark.parametrize("name,fn", DECISION_METHODS, ids=[n for n, _ in DECISION_METHODS])
def test_decision_methods_do_not_read_truth(name, fn):
    src = _code_only(fn)
    for tok in TRUTH_TOKENS:
        assert tok not in src, f"{name} reads simulator truth via `{tok}`"


def test_the_truth_readers_are_only_logging_and_scoring():
    """Truth IS read in this codebase.  Pin down exactly where, so that a new
    reader has to be added to this list deliberately rather than by accident."""
    allowed = {
        # logging / plotting
        "Executive._sample_truth", "ExecutiveV3._snap", "ExecutiveV3._truth_map",
        # odometry + path-length accounting (the simulator's own bookkeeping)
        "Executive._absorb_odometry", "ExecutiveV3._absorb",
        # after-the-fact scoring
        "Executive._check_wrong_turn", "Executive.summary", "ExecutiveV3._summary",
        # measures the realized turn (the turn is COMMANDED from the belief)
        "Executive.run",
        # the press record notes where the press physically happened
        "ExecutiveV3._press",
    }
    found = set()
    for cls, mod in ((ex_mod.Executive, "Executive"), (v3_mod.ExecutiveV3, "ExecutiveV3")):
        for n, fn in vars(cls).items():
            if not callable(fn) or n.startswith("__"):
                continue
            src = _code_only(fn)
            if any(t in src for t in TRUTH_TOKENS):
                found.add(f"{mod}.{n}")
    unexpected = found - allowed
    assert not unexpected, (
        f"new truth reader(s) {sorted(unexpected)} -- if this is logging or scoring, "
        f"add it to `allowed`; if it is a decision, it is a firewall breach")


def test_diagnostics_recorder_is_not_wired_by_the_pipeline():
    """The PF audit feeds truth to its recorder from the analysis harness.  If a
    future change made the executive supply it instead, the firewall tests above
    would start passing vacuously."""
    for cls in (ex_mod.Executive, v3_mod.ExecutiveV3):
        for n, fn in vars(cls).items():
            if callable(fn) and not n.startswith("__"):
                assert "diagnostics" not in _code_only(fn), (
                    f"{cls.__name__}.{n} wires the PF recorder; it must stay external")
