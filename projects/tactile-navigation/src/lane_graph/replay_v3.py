"""Offline replay for V3 contact-rich episodes (recorded states -> video).

Reuses the V1/V2 overlay stack; physics never re-runs.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from execution.replay_video_v2 import _draw_left, _draw_right, _left_panel, _right_panel

from .replay_v2 import TelemetryV2
from .scene3 import build_v3_scene


def render(episode_dir: Path, map_name: str, out: Path, stride: int = 1,
           fps: float = 25.0) -> None:
    import torch
    from execution.backends import BackendConfig
    from execution.mujoco_backend import MujocoConfig

    from .mj_backend2 import MujocoBackendV2
    from .run_v3 import PRESS_TARGET_N

    class _V3(MujocoBackendV2):
        def _press_target_n(self):
            return PRESS_TARGET_N

    ep = json.loads((episode_dir / "episode.json").read_text())
    rec = np.load(episode_dir / "states.npz")
    ts, qs = rec["t"], rec["qpos"]
    scene = build_v3_scene(map_name)
    be = _V3(scene, BackendConfig(probe_force_n=PRESS_TARGET_N),
             np.random.default_rng(0),
             MujocoConfig(seed=11 + int(ep.get("seed", 0)),
                          video_width=720, video_height=720,
                          camera_distance=2.4, camera_elevation=-20.0),
             render=True, video_path=None)
    tel = TelemetryV2(ep, scene)
    lfig, lart = _left_panel(scene, tel, 620, 720)
    rfig, rart = _right_panel(620, 720)
    cam = getattr(getattr(be.env, "_offline_renderer", None), "_cam", None)
    az, enc = None, None
    try:
        for n in range(0, len(ts), stride):
            t = float(ts[n])
            be.env.sim.data.qpos[0] = torch.from_numpy(qs[n]).to(be.device)
            be.env.sim.forward()
            if cam is not None:
                i = tel.at(t)["i"]
                target = math.degrees(tel.yaw[i])
                az = target if az is None else az + float(
                    np.clip((target - az + 180.0) % 360.0 - 180.0, -4, 4))
                cam.azimuth = az
            rgb = np.ascontiguousarray(
                np.asarray(be.env.render(), np.uint8)[:, :, :3])
            left = _draw_left(lfig, lart, tel, t)
            right = _draw_right(rfig, rart, tel, t)
            hh = min(left.shape[0], rgb.shape[0], right.shape[0])
            comp = np.hstack([left[:hh], rgb[:hh], right[:hh]])
            comp = comp[:comp.shape[0] // 2 * 2, :comp.shape[1] // 2 * 2]
            if enc is None:
                H, W = comp.shape[:2]
                enc = subprocess.Popen(
                    ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
                     "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r",
                     str(fps / stride), "-i", "-", "-an", "-c:v", "libx264",
                     "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
                     str(out)], stdin=subprocess.PIPE)
            enc.stdin.write(comp.tobytes())
    finally:
        if enc is not None:
            enc.stdin.close()
            enc.wait()
        be.close()
    print(f"v3 replayed {len(ts) // stride} frames -> {out}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args(argv)
    render(args.episode, args.map, args.out, stride=args.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
