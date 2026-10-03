"""New map geometry must agree between planning and simulation."""
import numpy as np
import pytest

from execution.demo_scenes import load_demo_scene
from lane_graph.maps_v3 import clearance, in_rooms
from lane_graph.mj_assets import build_obstacles_v3


@pytest.mark.parametrize("name,bodies,goal", [
    ("sky_offset_gate", 5, [4.4, 2.4]),
    ("sky_staggered_baffles", 6, [4.4, 2.4]),
    ("sky_island_choice", 5, [0.0, 3.6]),
])
def test_new_map_endpoints_and_simulator_geometry(name, bodies, goal):
    scene = load_demo_scene(name).scene
    np.testing.assert_allclose(scene.start_map_xy, [0., 0.])
    np.testing.assert_allclose(scene.goal_map_xy, goal, atol=1e-12)
    endpoints = np.array([scene.start_map_xy, scene.goal_map_xy])
    assert np.all(in_rooms(endpoints, scene.spec))
    assert np.all(clearance(endpoints, scene.spec) >= .6)
    obstacles = build_obstacles_v3(scene.spec)
    assert len(obstacles) == len(scene.plan.bodies) == bodies
