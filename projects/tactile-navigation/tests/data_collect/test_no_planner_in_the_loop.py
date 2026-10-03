"""Section 9: the planner takes no part in data collection.

The route the base poses are drawn along IS the planner's -- section 1 asks
for that -- so the rule cannot be "never import A*".  It is: A* runs ONCE, in
`rooms.planned_centreline`, and its output is a polyline frozen into the
manifest.  Nothing that chooses a pose, a target or an outcome may reach the
planner, the expected-information-gain machinery or the demo executor.

Checked on the parsed CODE, not the text: these modules discuss the planner
at length in their docstrings, and a grep would be pinning prose.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "data_collect"

PLANNER_MODULES = ("lane_graph.graph3", "lane_graph.executive2",
                   "lane_graph.executive3", "lane_graph.run_v3",
                   "tactile_explore.infogain", "demo_map.executor",
                   "demo_map.landmark_ate", "demo_map.scene_room2",
                   "demo_map.scene_demo", "execution.executive")
PLANNER_NAMES = ("astar3", "build_graph3", "Belief3", "DemoRouteExecutor",
                 "plan_steps_room2", "plan_room2_route",
                 "plan_steps_demo_room", "build_demo_scene",
                 "build_room2_scene", "LandmarkATE", "expected_information")

DECISION_MODULES = ("action_space.py", "coverage.py", "probe.py",
                    "collector.py", "records.py", "diagnostics.py",
                    "route.py")


def symbols(path: Path) -> set:
    """Every name the CODE could resolve: imports, references, attributes."""
    tree = ast.parse(path.read_text())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            out.add(n.module or "")
            out.update(a.name for a in n.names)
        elif isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            # a name reached through __import__/getattr would hide in a string
            if not n.value.strip().startswith(("\n", " ")) and len(n.value) < 60:
                out.add(n.value)
    return out


@pytest.mark.parametrize("name", DECISION_MODULES)
def test_no_decision_module_can_reach_the_planner(name):
    s = symbols(SRC / name)
    for mod in PLANNER_MODULES:
        assert not any(x == mod or x.startswith(mod + ".") for x in s), \
            f"{name} imports {mod}"
    for fn in PLANNER_NAMES:
        assert fn not in s, f"{name} references {fn}"


def test_only_the_route_builder_runs_the_planner_and_it_runs_it_once():
    src = (SRC / "rooms.py").read_text()
    tree = ast.parse(src)
    callers = [f.name for f in ast.walk(tree)
               if isinstance(f, ast.FunctionDef)
               and "astar3(" in (ast.get_source_segment(src, f) or "")]
    assert callers == ["_astar_polyline"], callers
    users = [f.name for f in ast.walk(tree)
             if isinstance(f, ast.FunctionDef)
             and "_astar_polyline(" in (ast.get_source_segment(src, f) or "")
             and f.name != "_astar_polyline"]
    assert users == ["planned_centreline"], users


def test_the_collector_cannot_rebuild_a_route_mid_run():
    """`planned_centreline` is called by the run script at start-up.  If the
    collector could call it, an episode could re-plan."""
    s = symbols(SRC / "collector.py")
    assert "planned_centreline" not in s
    assert "data_collect.rooms" not in s and "rooms" not in s


def test_the_execution_path_cannot_read_the_balancing_geometry():
    """Section 4 permits simulator geometry to the SAMPLER, for hit/miss
    balance only.  `swept_hit` is that read, and the module that drives the
    controller must not be able to call it."""
    assert "swept_hit" not in symbols(SRC / "probe.py")


def test_the_label_is_never_taken_from_the_prediction():
    """`predicted_hit` is recorded beside the outcome and never as it."""
    src = (SRC / "collector.py").read_text()
    tree = ast.parse(src)
    fn = next(f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
              and f.name == "_observation_record")
    body = ast.get_source_segment(src, fn) or ""
    assert "predicted" not in body
    assert '"label": res.label' in body
