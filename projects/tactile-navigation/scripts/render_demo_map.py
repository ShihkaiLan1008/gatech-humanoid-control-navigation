"""3D views and a route flythrough of the reconstructed demo room.

    MUJOCO_GL=egl PYTHONPATH=src python3 scripts/render_demo_map.py

WHAT THIS IS NOT.  No controller runs here and no physics is stepped.  The G1 is
placed at poses taken from the sparse reference route, at its nominal standing
pose, yawed along the route tangent -- a kinematic preview of the geometry, the
same placement the Phase 5 clearance sweep uses.  A real episode needs the mjlab
environment and the trained checkpoint, which is separate work.

Outputs (under --out):
    room_overview.png     the whole room from above, robot at START
    room_views.png        four eye-level views: START, C1/C2, corner, doorway
    route_flythrough.mp4  chase camera following the route end to end
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np

from demo_map import build_demo_map, build_reference_route
from demo_map.mujoco_scene import G1_SCENE, combined_with_g1

ROBOT_Z_KEY = 2   # freejoint qpos index of the root height


def _place(model, data, xy, yaw: float):
    import mujoco

    data.qpos[:] = model.qpos0
    data.qpos[0:2] = xy
    data.qpos[3:7] = [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]
    mujoco.mj_forward(model, data)


def _cam(lookat, distance, azimuth, elevation):
    """Free camera.  ``azimuth`` is the compass bearing the camera LOOKS ALONG
    (0 = +x, 90 = +y); the camera sits ``distance`` back along that bearing, so
    a chase view uses the robot's own heading.  ``elevation`` must be steep
    enough to lift the camera over the 1.8 m walls or it renders from inside a
    slab -- in a 1.63 m corridor that is most of the frame."""
    import mujoco

    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.distance = distance
    c.azimuth = azimuth
    c.elevation = elevation
    return c


def _route_frames(r, ds: float):
    """Arc-length-spaced (xy, yaw) samples along the reference route."""
    poly = r.polyline(0.01)
    keep = [0]
    for i in range(1, len(poly)):
        if float(np.linalg.norm(poly[i] - poly[keep[-1]])) >= ds:
            keep.append(i)
    if keep[-1] != len(poly) - 1:
        keep.append(len(poly) - 1)
    out = []
    for i in keep:
        j = min(i + 1, len(poly) - 1)
        k = i - 1 if j == i else i
        t = poly[j] - poly[k]
        out.append((poly[i], math.atan2(float(t[1]), float(t[0]))
                    if np.any(t) else 0.0))
    return out


def _unwrap_deg(vals):
    """Continuous azimuth track: a chase camera must not spin 359 deg at a turn."""
    return np.degrees(np.unwrap(np.radians(np.asarray(vals, float))))


def _smooth(a, k):
    a = np.asarray(a, float)
    if k < 3 or len(a) < k:
        return a
    k |= 1
    pad = k // 2
    return np.convolve(np.pad(a, pad, mode="edge"), np.ones(k) / k, "valid")[:len(a)]


def main() -> None:
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/demo_map"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--ds", type=float, default=0.08,
                    help="route spacing between video frames [m]")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--map", choices=("tape", "scan"), default="tape")
    ap.add_argument("--walls", type=Path, default=None,
                    help="scan walls JSON (implies --map scan)")
    ap.add_argument("--tag", default=None, help="output filename suffix")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.map

    if not Path(G1_SCENE).exists():
        raise SystemExit(f"G1 scene not found: {G1_SCENE}")

    if args.walls:
        args.map = "scan"
    if args.map == "scan":
        from demo_map.scan_map import (build_scan_map, build_scan_route,
                                       build_scan_traverse)
        m = build_scan_map(args.walls) if args.walls else build_scan_map()
        r = (build_scan_route(m) if m.key_points else build_scan_traverse(m))
        kp = dict(_generic=not m.key_points,
                  x_west=min(p[0] for w in m.walls for p in (w.p0, w.p1)),
                  x_upper_e=max(p[0] for w in m.walls for p in (w.p0, w.p1)),
                  y_bottom=min(p[1] for w in m.walls for p in (w.p0, w.p1)),
                  y_upper_n=max(p[1] for w in m.walls for p in (w.p0, w.p1)),
                  x_c1_w=0.0, x_c1_e=0.5969,
                  x_corner=m.key_points.get("branch_w_x", 0.0),
                  x_corr_e=m.key_points.get("branch_e_x", 0.0),
                  y_upper_s=m.key_points.get("room_s_y", 0.0))
    else:
        m = build_demo_map()
        r = build_reference_route(m)
        kp = m.key_points
    model = combined_with_g1(m, at=m.start.center, yaw=m.start.heading_rad)
    # the shipped scenes size their offscreen buffer for their own cameras
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.height, args.width)

    def shot(xy, yaw, cam):
        _place(model, data, xy, yaw)
        renderer.update_scene(data, camera=cam)
        return renderer.render()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    # ── overview ────────────────────────────────────────────────────────────
    cx = 0.5 * (kp["x_west"] + kp["x_upper_e"])
    cy = 0.5 * (kp["y_bottom"] + kp["y_upper_n"])
    # frame the map you actually have, not the one the constant was tuned for
    span = max(kp["x_upper_e"] - kp["x_west"], kp["y_upper_n"] - kp["y_bottom"])
    over = shot(m.start.center, m.start.heading_rad,
                _cam([cx, cy, 0.0], 1.45 * span + 3.0, 90.0, -62.0))
    if plt is not None:
        fig, ax = plt.subplots(figsize=(args.width / 145, args.height / 145))
        ax.imshow(over); ax.set_axis_off()
        ax.set_title(f"{tag}: MuJoCo from the scan -- {len(m.walls)} walls, "
                     f"{len(m.curves)} curved features, {len(m.boxes)} clutter "
                     "boxes, 1.8 m high, G1 at START", fontsize=10)
        fig.savefig(args.out / f"room_overview_{tag}.png", dpi=145, bbox_inches="tight")
        plt.close(fig)

    # ── four eye-level views ────────────────────────────────────────────────
    frames = _route_frames(r, max(r.length / 4.5, 0.05))
    c2 = m.curve("C2") if m.curves else None
    views = [
        ("START, looking east along the long wall past C1",
         m.start.center, 0.0,
         _cam([kp["x_c1_e"] + 0.3, -0.7, 0.8], 5.0, 0.0, -25.0)),
        ("C2 -- 0.47 m proud, the feature the lane must bow around",
         (c2.center[0] - 1.5, -0.56) if c2 else (0.0, 0.0), 0.0,
         _cam([c2.center[0] if c2 else 0.0, -0.5, 0.6], 4.5, 20.0, -28.0)),
        ("corridor mouth: the long wall ends, transit to the east wall",
         (kp["x_corner"] - 0.4, -0.56), 0.0,
         _cam([kp["x_corner"] + 1.0, 0.5, 0.6], 6.5, 30.0, -30.0)),
        ("convex corner into the upper room, doorway GOAL beyond",
         (kp["x_corr_e"] - 0.56, kp["y_upper_s"] - 1.2), math.pi / 2,
         _cam([kp["x_corr_e"] + 0.3, kp["y_upper_s"] + 0.3, 0.8], 7.0, 75.0, -30.0)),
    ]
    if kp.get("_generic"):
        # no named landmarks in a room the tape never measured: look in from
        # each side of the traverse instead
        span = max(kp["x_upper_e"] - kp["x_west"], kp["y_upper_n"] - kp["y_bottom"])
        cx = 0.5 * (kp["x_west"] + kp["x_upper_e"])
        cy = 0.5 * (kp["y_bottom"] + kp["y_upper_n"])
        views = [(f"looking {name}", frames[i][0], frames[i][1],
                  _cam([cx, cy, 0.9], 1.35 * span, az, -28.0))
                 for i, (name, az) in enumerate(
                     [("east", 0.0), ("north", 90.0), ("west", 180.0),
                      ("south", 270.0)])]
    tiles = [(t, shot(xy, yaw, cam)) for t, xy, yaw, cam in views]
    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.6))
        for ax, (title, img) in zip(axes.ravel(), tiles):
            ax.imshow(img); ax.set_axis_off()
            ax.set_title(title, fontsize=8.5)
        fig.suptitle(f"{tag} -- G1 posed along the reference path "
                     "(kinematic preview, no controller)", fontsize=11)
        fig.tight_layout()
        fig.savefig(args.out / f"room_views_{tag}.png", dpi=145, bbox_inches="tight")
        plt.close(fig)

    # ── chase-camera flythrough ─────────────────────────────────────────────
    frames = _route_frames(r, args.ds)
    az = _smooth(_unwrap_deg([math.degrees(y) for _, y in frames]), 21)
    xs = _smooth([p[0] for p, _ in frames], 9)
    ys = _smooth([p[1] for p, _ in frames], 9)

    w2, h2 = args.width // 2 * 2, args.height // 2 * 2
    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w2}x{h2}", "-r", str(args.fps), "-i", "-", "-an",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", str(args.out / f"route_flythrough_{tag}.mp4")],
        stdin=subprocess.PIPE)
    try:
        for i, (xy, yaw) in enumerate(frames):
            img = shot(xy, yaw, _cam([xs[i], ys[i], 0.85], 4.6, float(az[i]), -20.0))
            enc.stdin.write(img[:h2, :w2].tobytes())
    finally:
        enc.stdin.close()
        enc.wait()

    print(f"{len(frames)} frames, {r.length:.2f} m route, "
          f"{len(frames) / args.fps:.1f} s at {args.fps} fps")
    for f in (f"room_overview_{tag}.png", f"room_views_{tag}.png",
              f"route_flythrough_{tag}.mp4"):
        p = args.out / f
        if p.exists():
            print(f"  {p}  ({p.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
