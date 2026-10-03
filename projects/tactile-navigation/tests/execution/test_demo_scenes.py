"""Scene parity and a global run that cannot import legacy task policies."""
from dataclasses import asdict
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from execution.demo_scenes import load_demo_scene


@pytest.mark.parametrize("room", ["room2", "room2_turn", "demo_room", "room830"])
@pytest.mark.parametrize("standoff", [.35, .40])
def test_scene_matches_legacy_setup_without_importing_tasks(monkeypatch, room, standoff):
    from execution.punch_tasks import TASKS

    old = TASKS[room](standoff_m=standoff)
    monkeypatch.setitem(sys.modules, "execution.punch_tasks", None)
    new = load_demo_scene(room, standoff_m=standoff)
    np.testing.assert_equal(asdict(new.scene.spec), asdict(old.scene.spec))
    np.testing.assert_equal(new.scene.start_map_xy, old.scene.start_map_xy)
    np.testing.assert_equal(new.scene.goal_map_xy, old.scene.goal_map_xy)
    assert new.face_yaw_map == pytest.approx(old.face_yaw_map)
    assert new.scene.goal_radius_m == old.scene.goal_radius_m
    assert "features" not in new.describe()


def test_room2_global_completes_without_legacy_tasks(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    monkeypatch.setitem(sys.modules, "execution.punch_tasks", None)
    main = runpy.run_path(str(root / "scripts/punch_demo.py"))["main"]
    assert main(["--room", "room2", "--policy", "global", "--controller", "kinematic",
                 "--seed", "0", "--no-video", "--out", str(tmp_path)]) == 0
    record = json.loads(next(tmp_path.glob("*.eval.json")).read_text())
    assert record["outcome"] == "route_complete"
    assert record["metrics"]["success"] is True


@pytest.mark.parametrize("script", ["punch_eval.py", "punch_video.py"])
def test_eval_and_replay_entrypoints_do_not_import_tasks(monkeypatch, script):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setitem(sys.modules, "execution.punch_tasks", None)
    main = runpy.run_path(str(root / "scripts" / script))["main"]
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
