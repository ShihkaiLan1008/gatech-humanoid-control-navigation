"""Authored coordinates and map-to-simulator geometry for imported JSON maps."""
import json
import math
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from demo_map.manifest import FORMAT, build_manifest_scene
from execution.demo_scenes import REFERENCE_MAPS, load_demo_scene
from lane_graph.maps_v3 import clearance, in_rooms
from lane_graph.mj_assets import build_obstacles_v3


@pytest.mark.parametrize("name,goal,count", [
    ("ref1_bend", (7., 5.25), 20),
    ("ref2_hall", (9., -.25), 11),
    ("ref3_zsuite", (7.1, 9.9), 11),
    ("ref4_room", (7., 4.7), 9),
])
def test_reference_coordinates_and_geometry(monkeypatch, tmp_path, name, goal, count):
    monkeypatch.chdir(tmp_path)  # registered paths must not depend on shell cwd
    monkeypatch.setitem(sys.modules, "execution.punch_tasks", None)
    setup = load_demo_scene(name)
    s = setup.scene
    np.testing.assert_allclose(s.start_map_xy, (0., 0.))
    np.testing.assert_allclose(s.goal_map_xy, goal, atol=1e-12)
    assert s.goal_radius_m == .30
    assert setup.face_yaw_map == math.pi / 2
    endpoints = np.array([s.start_map_xy, s.goal_map_xy])
    assert np.all(in_rooms(endpoints, s.spec))
    assert np.all(clearance(endpoints, s.spec) >= .60)
    obstacles = build_obstacles_v3(s.spec)
    assert len(obstacles) == len(s.plan.bodies) == count
    assert all(o.half_height == .75 for o in obstacles)


def test_manifest_controls_start_goal_and_heading():
    doc = dict(format=FORMAT, name="empty", room=dict(width_m=8., depth_m=6.),
               start_xy=[2., 1.], start_heading_deg=0., goal_xy=[5., 4.],
               goal_radius_m=.45, obstacles=[])
    scene = build_manifest_scene(doc)
    # Starting east becomes map north: room displacement (3,3) becomes (-3,3).
    np.testing.assert_allclose(scene.goal_map_xy, [-3., 3.])
    assert scene.goal_radius_m == .45
    np.testing.assert_allclose(scene.spec.rooms, [(-5., -2., 1., 6.)], atol=1e-12)
    assert len(scene.plan.bodies) == 4


@pytest.mark.parametrize("field,value", [
    ("start_heading_deg", 45.), ("wall_thickness_m", .2), ("body_height_m", 3.),
])
def test_incompatible_geometry_is_not_silently_changed(field, value):
    doc = json.loads(REFERENCE_MAPS["ref4_room"].read_text())
    doc[field] = value
    with pytest.raises(ValueError):
        build_manifest_scene(doc)


def test_reference_room_requires_global_policy(capsys):
    root = Path(__file__).resolve().parents[2]
    main = runpy.run_path(str(root / "scripts/punch_demo.py"))["main"]
    with pytest.raises(SystemExit) as error:
        main(["--room", "ref4_room", "--policy", "contact"])
    assert error.value.code == 2
    assert "add --policy global" in capsys.readouterr().err
