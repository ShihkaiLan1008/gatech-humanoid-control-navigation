"""The live viewer draws the HOST copy, so the room has to be copied there."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

mujoco = pytest.importorskip("mujoco")

from execution.mujoco_backend import MujocoBackend

XML = """
<mujoco>
  <worldbody>
    <body name="free"><freejoint/><geom size=".05"/></body>
    <body name="wall_a" mocap="true"><geom size=".5 .05 1"/></body>
    <body name="wall_b" mocap="true"><geom size=".5 .05 1"/></body>
  </worldbody>
</mujoco>
"""


class _Warp:
    """A warp-style device array: `.numpy()` is the way off the device."""
    def __init__(self, a):
        self._a = np.asarray(a, np.float32)

    def numpy(self):
        return self._a


class _Cuda:
    """A torch-style CUDA tensor: `.numpy()` EXISTS and RAISES; only
    `.detach().cpu().numpy()` works.  mjlab hands out these."""
    def __init__(self, a):
        self._a = np.asarray(a, np.float32)

    def numpy(self):
        raise TypeError("can't convert cuda:0 device type tensor to numpy")

    def detach(self):
        return self

    def cpu(self):
        return _Warp(self._a)


class _FakeViewer:
    def __init__(self):
        self.synced = 0

    def is_running(self):
        return True

    def sync(self):
        self.synced += 1

    def close(self):
        pass


def _backend_with(model, data, dev_mocap_pos, dev_mocap_quat, qpos,
                  kind=_Cuda):
    be = MujocoBackend.__new__(MujocoBackend)
    be.view_on = True
    be._viewer = _FakeViewer()
    dev = type("D", (), {"qpos": kind(qpos[None, :]),
                         "mocap_pos": kind(dev_mocap_pos[None, ...]),
                         "mocap_quat": kind(dev_mocap_quat[None, ...])})()
    sim = type("S", (), {"mj_model": model, "mj_data": data, "data": dev})()
    be.env = type("E", (), {"sim": sim})()
    return be


@pytest.mark.parametrize("kind", [_Cuda, _Warp],
                         ids=["torch-cuda", "warp"])
def test_the_room_reaches_the_window_not_just_the_robot(kind):
    """Walls, pillars and boxes are MOCAP bodies placed with
    `write_mocap_pose_to_sim`, which writes `sim.data.mocap_*` on the DEVICE.
    Copying only qpos left the host copy holding whatever the model shipped
    with -- and inactive maze walls ship PARKED 25 m up -- so the viewer showed
    a robot walking an empty floor at every camera distance, while the replay
    video (which rebuilds the scene device-side) looked correct."""
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    assert m.nmocap == 2

    d.mocap_pos[:] = [[0.0, 0.0, 25.0], [0.0, 0.0, 25.5]]      # parked
    placed = np.array([[1.5, 0.0, 1.0], [0.0, 2.0, 1.0]])      # the real room
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1))
    q = np.zeros(m.nq)
    q[:7] = [0.3, 0.4, 0.5, 1, 0, 0, 0]

    be = _backend_with(m, d, placed, quat, q, kind=kind)
    be._sync_viewer()

    assert np.allclose(d.mocap_pos, placed), "the room never reached the host"
    assert np.allclose(d.qpos[:7], q[:7]), "the robot must still be copied"
    # ...and mj_forward ran, so the wall bodies are actually somewhere to draw
    wall = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wall_a")
    assert np.allclose(d.xpos[wall], placed[0])
    assert be._viewer.synced == 1


def test_a_viewer_that_cannot_open_never_takes_the_run_down():
    """The run is the product; the window is a convenience."""
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    be = _backend_with(m, d, np.zeros((2, 3)), np.zeros((2, 4)),
                       np.zeros(m.nq))
    be.env.sim.data = None                    # anything at all going wrong
    be._sync_viewer()                         # must not raise
    assert be.view_on is False                # and must stop trying
