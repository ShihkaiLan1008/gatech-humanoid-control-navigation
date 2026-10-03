"""Replay a recorded wall-parallel run into a video.  Physics does not re-run.

    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
      tactile_controller/.venv/bin/python scripts/wall_parallel_video.py \
        --states out/hw/wpx/states_r5_twinbay_s0_s0g0.npz \
        --episode out/hw/wpx/wp_r5_twinbay_s0_s0g0.json \
        --out out/hw/wpx/r5_seg0.mp4

The camera tracks the base along the followed wall and looks at it from the room
side, which is the view the behaviour is about: constant standoff, and a hand
that goes out and comes back.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def render(states: Path, episode: Path, out: Path, *, stride: int = 1,
           fps: float = 25.0, distance: float = 2.6,
           elevation: float = -16.0) -> None:
    import torch

    from execution.backends import BackendConfig
    from execution.mujoco_backend import MujocoConfig
    from lane_graph.mj_backend2 import MujocoBackendV2
    from lane_graph.run_v3 import PRESS_TARGET_N
    from lane_graph.scene3 import build_v3_scene

    ep = json.loads(episode.read_text())
    rec = np.load(states)
    ts, qs = rec["t"], rec["qpos"]
    scene = build_v3_scene(ep["scene"])

    class _V3(MujocoBackendV2):
        def _press_target_n(self):
            return PRESS_TARGET_N

    be = _V3(scene, BackendConfig(probe_force_n=PRESS_TARGET_N),
             np.random.default_rng(0),
             MujocoConfig(seed=11 + int(ep.get("seed", 0)),
                          video_width=960, video_height=640,
                          camera_distance=distance, camera_elevation=elevation),
             render=True, video_path=None)

    # look at the wall from the room side: the camera sits on the segment's
    # outward normal, so the standoff and the hand extension are both in frame
    seg = ep["runs"][0]["segment"] if "runs" in ep else ep["segment"]
    n_map = np.asarray(seg["normal"], float)
    t_map = np.asarray(seg["tangent"], float)
    n_w = n_map[0] * be.u + n_map[1] * be.v
    t_w = t_map[0] * be.u + t_map[1] * be.v
    # three-quarter view from the room side, tilted along the direction of
    # travel: the standoff, the wall running away and the hand are all in frame.
    # Keep `distance` under the room's depth -- these rooms are 4.4 m across, so
    # a 4.2 m orbit puts the camera inside the opposite wall.
    look = -0.55 * n_w - 0.84 * t_w
    azimuth = math.degrees(math.atan2(look[1], look[0]))
    cam = getattr(getattr(be.env, "_offline_renderer", None), "_cam", None)
    if cam is not None:
        import mujoco
        # the default viewer origin is WORLD, i.e. a FREE camera whose lookat
        # never moves: the robot walks straight out of frame.  Track the base.
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
        cam.trackbodyid = be.robot.indexing.root_body_id
        cam.fixedcamid = -1
        cam.azimuth = azimuth
        cam.elevation = elevation
        cam.distance = distance
    enc = None
    try:
        for n in range(0, len(ts), stride):
            be.env.sim.data.qpos[0] = torch.from_numpy(qs[n]).to(be.device)
            be.env.sim.forward()
            rgb = np.ascontiguousarray(
                np.asarray(be.env.render(), np.uint8)[:, :, :3])
            rgb = rgb[:rgb.shape[0] // 2 * 2, :rgb.shape[1] // 2 * 2]
            if enc is None:
                h, w = rgb.shape[:2]
                enc = subprocess.Popen(
                    ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
                     "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r",
                     str(fps / stride), "-i", "-", "-an", "-c:v", "libx264",
                     "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
                     str(out)], stdin=subprocess.PIPE)
            enc.stdin.write(rgb.tobytes())
    finally:
        if enc is not None:
            enc.stdin.close()
            enc.wait()
        be.close()
    print(f"{len(ts) // stride} frames -> {out}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--distance", type=float, default=2.6)
    ap.add_argument("--elevation", type=float, default=-16.0)
    a = ap.parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    render(a.states, a.episode, a.out, stride=a.stride, distance=a.distance,
           elevation=a.elevation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
