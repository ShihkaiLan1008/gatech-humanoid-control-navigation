#!/usr/bin/env python3
"""Replay a punch-demo state log into a MuJoCo video with a small HUD.

    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \\
      tactile_controller/.venv/bin/python scripts/punch_video.py \\
        --run out/punch/room2/room2_vforce_s0.json \\
        --states out/punch/room2/states_room2_vforce_s0.npz \\
        --out out/punch/room2/room2_vforce_s0.mp4

Nothing is re-simulated: the recorded qpos is written into a render env
built by the same controller factory, with the run's own spawn frame so the
room stands where it stood.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

FPS = 25.0
PANEL_PX = 720


def _belief_panel(doc, scene):
    """The 2D half: map, TRUE path vs BELIEVED path, the live 2-sigma
    ellipse, every punch drawn where the belief put the fist, and a strip
    of sigma and |truth - belief| against time.  Returns (fig, update(t))
    where update renders one frame as an RGB array."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    nav = doc["nav_trace"]
    nav_t = np.array([r["t"] for r in nav])
    bel = np.array([r["bel_xy"] for r in nav])
    cov = np.array([np.asarray(r["cov_xy"], float) for r in nav])
    sig = np.array([r["sigma_m"] for r in nav])
    err = np.array([r["err_m"] for r in nav])
    # PF rows are primitive-boundary snapshots. The evaluation observer also
    # records the live posterior+odometry estimate at every control tick; show
    # that separately instead of depicting a stationary estimate during a move.
    live = [row for row in doc.get("evaluation_samples", [])
            if row.get("estimate") is not None]
    live_t = np.array([row["t"] for row in live])
    live_pose = np.array([row["estimate"] for row in live])
    live_err = np.array([np.linalg.norm(np.asarray(row["estimate"])[:2]
                                      - np.asarray(row["truth"])[:2])
                         if row.get("truth") is not None else np.nan for row in live])
    trace = np.asarray(doc.get("trace", []), float)
    touches = doc.get("touches", [])

    fig = plt.figure(figsize=(PANEL_PX / 100, PANEL_PX / 100), dpi=100)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 1.3], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    axs = fig.add_subplot(gs[1])
    for b in scene.plan.bodies:
        v = np.asarray(b.verts, float)
        ax.fill(v[:, 0], v[:, 1], color="0.82", ec="0.35", lw=0.8, zorder=1)
    ax.plot(0, 0, marker="*", ms=11, color="tab:green", zorder=6)
    pts = np.concatenate([np.asarray(b.verts, float) for b in scene.plan.bodies]
                         + [trace[:, 1:3] if len(trace) else np.zeros((1, 2))])
    lo, hi = pts.min(axis=0) - 0.6, pts.max(axis=0) + 0.6
    # A WINDOW THAT FOLLOWS THE ROBOT.  A fixed frame of the whole run
    # shrinks a 12 m hallway's robot, ellipse and punch marks to dots; 6 m
    # around the true base keeps them legible in every room, clamped to the
    # room so the window never shows only void.
    WIN = 6.0
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_title("Live estimate (cyan); PF snapshots + 2-sigma ellipse (blue)",
                 fontsize=10)
    (l_truth,) = ax.plot([], [], color="k", lw=1.6, zorder=4, label="truth")
    (l_bel,) = ax.plot([], [], color="tab:blue", lw=1.4, ls="--", zorder=5,
                       label="PF snapshots")
    (l_live,) = ax.plot([], [], color="tab:cyan", lw=1.3, zorder=5,
                        label="live estimate" if live else None)
    (m_live,) = ax.plot([], [], marker="o", ms=5, color="tab:cyan", zorder=7)
    (m_truth,) = ax.plot([], [], marker="o", ms=6, color="k", zorder=7)
    (m_bel,) = ax.plot([], [], marker="o", ms=6, color="tab:blue", zorder=7)
    ell = Ellipse((0, 0), 0.1, 0.1, angle=0, fc="tab:blue", alpha=0.18,
                  ec="tab:blue", lw=1.2, zorder=3)
    ax.add_patch(ell)
    hit_sc = ax.scatter([], [], s=28, color="tab:red", zorder=8, label="punch landed")
    miss_sc = ax.scatter([], [], s=26, facecolors="none", edgecolors="0.4",
                         zorder=8, label="punch missed")
    ax.legend(loc="upper right", fontsize=8)
    hud = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=9, family="monospace",
                  bbox=dict(fc="white", ec="0.6", alpha=0.85))
    axs.plot(nav_t, sig, color="tab:blue", lw=1.2, label="PF snapshot sigma [m]")
    axs.plot(live_t if live else nav_t, live_err if live else err,
             color="k", lw=1.2, label="estimation error [m]")
    cursor = axs.axvline(0, color="tab:red", lw=1)
    axs.set_xlim(0, float(nav_t[-1]) if len(nav_t) else 1.0)
    error_max = float(np.nanmax(live_err)) if np.isfinite(live_err).any() else float(err.max())
    axs.set_ylim(0, max(0.3, float(max(sig.max(), error_max)) * 1.15))
    axs.set_xlabel("t [s]", fontsize=9)
    axs.legend(loc="upper left", fontsize=8)
    axs.grid(alpha=0.25)
    fig.canvas.draw()

    def update(t):
        k = int(np.searchsorted(trace[:, 0], t)) if len(trace) else 0
        k = min(max(k, 1), len(trace))
        l_truth.set_data(trace[:k, 1], trace[:k, 2])
        m_truth.set_data([trace[k - 1, 1]], [trace[k - 1, 2]])
        cx, cy = float(trace[k - 1, 1]), float(trace[k - 1, 2])
        cx = float(np.clip(cx, lo[0] + WIN / 2, max(hi[0] - WIN / 2, lo[0] + WIN / 2)))
        cy = float(np.clip(cy, lo[1] + WIN / 2, max(hi[1] - WIN / 2, lo[1] + WIN / 2)))
        ax.set_xlim(cx - WIN / 2, cx + WIN / 2)
        ax.set_ylim(cy - WIN / 2, cy + WIN / 2)
        i = int(np.searchsorted(nav_t, t, side="right")) - 1
        i = min(max(i, 0), len(nav) - 1)
        l_bel.set_data(bel[:i + 1, 0], bel[:i + 1, 1])
        m_bel.set_data([bel[i, 0]], [bel[i, 1]])
        if live:
            j_live = max(0, int(np.searchsorted(live_t, t, side="right")) - 1)
            l_live.set_data(live_pose[:j_live + 1, 0], live_pose[:j_live + 1, 1])
            m_live.set_data([live_pose[j_live, 0]], [live_pose[j_live, 1]])
        w, v = np.linalg.eigh(cov[i])
        w = np.maximum(w, 1e-9)
        ell.set_center((bel[i, 0], bel[i, 1]))
        ell.width, ell.height = 4.0 * math.sqrt(w[1]), 4.0 * math.sqrt(w[0])
        ell.angle = math.degrees(math.atan2(float(v[1, 1]), float(v[0, 1])))
        hits = [tc["p_map_believed"] for tc in touches
                if tc["t"] <= t and tc["contact"] and tc["p_map_believed"]]
        hit_sc.set_offsets(np.asarray(hits, float).reshape(-1, 2))
        misses = []
        for tc in touches:
            if tc["t"] <= t and not tc["contact"]:
                # a miss is drawn at the fist's full reach along the axis
                # from the believed base
                j = int(np.searchsorted(nav_t, tc["t"], side="right")) - 1
                b = bel[max(j, 0)]
                a = np.asarray(tc["axis_map"], float)
                misses.append(b + 0.54 * a)
        miss_sc.set_offsets(np.asarray(misses, float).reshape(-1, 2))
        current_err = live_err[j_live] if live else err[i]
        hud.set_text(f"t {t:6.1f} s\nPF age {max(0., t-nav_t[i]):.1f} s\n"
                     f"PF sigma {sig[i]:.3f} m\nerr {current_err:.3f} m\n"
                     f"{nav[i].get('belief_state', '')}")
        cursor.set_xdata([t, t])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        return np.ascontiguousarray(buf)

    return fig, update


def _hud(rgb: np.ndarray, lines: list[tuple[str, tuple]]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    y = 12
    for text, color in lines:
        w = dr.textlength(text, font=font)
        dr.rectangle([8, y - 4, 16 + w, y + 26], fill=(0, 0, 0, 140))
        dr.text((12, y), text, fill=color, font=font)
        y += 32
    return np.asarray(im)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", help="checkpoint used by the recorded controller, if overridden")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=float, default=None,
                    help="default keeps real time (25/stride)")
    ap.add_argument("--distance", type=float, default=3.8)
    ap.add_argument("--elevation", type=float, default=-32.0)
    a = ap.parse_args(argv)
    if a.stride < 1:
        ap.error("--stride must be positive")
    # Compute nodes may not have a system ffmpeg; imageio ships one in the
    # controller environment already used for video export.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()

    import torch
    from execution.controllers import make_controller
    from execution.demo_scenes import load_demo_scene

    doc = json.loads(a.run.read_text())
    evaluation = a.run.with_suffix(".eval.json")
    if doc.get("evaluation_file"):
        evaluation = a.run.parent / doc["evaluation_file"]
    if evaluation.is_file():
        doc["evaluation_samples"] = json.loads(evaluation.read_text()).get("samples", [])
    rec = np.load(a.states)
    ts, qs = rec["t"], rec["qpos"]
    task = doc["task"]
    scene = load_demo_scene(doc["room"], standoff_m=task.get("standoff_m", 0.40)).scene
    be = make_controller(doc["controller"], scene, seed=int(doc["seed"]),
                         face_yaw_map=float(task["face_yaw_map"]), render=True,
                         video_width=960, video_height=720,
                         camera_distance=a.distance,
                         camera_elevation=a.elevation,
                         **({"checkpoint": a.checkpoint} if a.checkpoint else {}))
    frame = doc.get("spawn_frame")
    if frame:
        be.base0 = np.asarray(frame["base0"], float)
        be.u = np.asarray(frame["u"], float)
        be.v = np.asarray(frame["v"], float)
        be.yaw0 = float(frame["yaw0"])
        be._install_world()
    cam = getattr(getattr(be.env, "_offline_renderer", None), "_cam", None)
    if cam is not None:
        import mujoco
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
        cam.trackbodyid = be.env.scene["robot"].indexing.root_body_id
        cam.fixedcamid = -1
        cam.elevation = a.elevation
        cam.distance = a.distance
        # THIRD-PERSON, ELEVATED, LOOKING ALONG THE WALL from behind the
        # direction of travel.  Compared on recorded punch frames against
        # behind, behind-left/right, side and overhead views: the robot is
        # in full profile with both fists against the surface, both walls
        # of the hallway stay in frame, the pillars and the corner appear
        # ahead before the robot reaches them, and nothing ever stands
        # between the camera and the robot -- a camera off to the room side
        # had the demo room's opposite wall cutting the robot at the knees.
        def _azimuth(look_map):
            look_map = np.asarray(look_map, float)
            look_map = look_map / np.linalg.norm(look_map)
            look = look_map[0] * be.u + look_map[1] * be.v
            return math.degrees(math.atan2(float(look[1]), float(look[0])))
        route = doc.get("route")
        if isinstance(route, dict) and route.get("legs"):
            along0 = np.asarray(route["legs"][0]["along"], float)
        else:
            along0 = np.asarray(task.get("along_map", [math.cos(task["face_yaw_map"]),
                                                       math.sin(task["face_yaw_map"])]), float)
        az_first = _azimuth(along0)
        # AT EVERY CORNER THE CAMERA TURNS WITH THE ROBOT.  Held on the
        # first wall's direction it looked along that wall's back once the
        # robot had rounded the corner and the final segment showed only
        # walls.  Each rounding move logs its heading change, and the map
        # frame is a rigid motion of the world, so the azimuth target turns
        # by the same angle; it slews there from the moment the move begins.
        cam.azimuth = az_first
        # THE CAMERA MUST STAND INSIDE THE ROOM.  Looking along the wall
        # from behind puts it distance*cos(elevation) behind the robot, and
        # after room2's concave turn that is beyond the west wall: the
        # replay showed that wall's back with the robot's head above it.
        # Per frame, the look direction is the along-wall one rotated by
        # the turns so far; if the camera it implies would be outside the
        # room, the look swings toward the wall the robot faces (from the
        # room side) until it is inside.
        _pts = np.concatenate([np.asarray(b.verts, float)
                               for b in scene.plan.bodies])
        room_lo, room_hi = _pts.min(axis=0) - 0.3, _pts.max(axis=0) + 0.3
        horiz = float(a.distance) * math.cos(math.radians(abs(a.elevation)))
        heading0 = np.array([math.cos(task["face_yaw_map"]),
                             math.sin(task["face_yaw_map"])])

        def _rot(v, deg):
            r = math.radians(deg)
            return np.array([math.cos(r) * v[0] - math.sin(r) * v[1],
                             math.sin(r) * v[0] + math.cos(r) * v[1]])

        def _look_for(xy_map, turn_deg):
            along = _rot(along0, turn_deg)
            head = _rot(heading0, turn_deg)
            for w in (0.0, 0.5, 0.8, 1.0):
                look = (1.0 - w) * along + w * head
                look /= max(np.linalg.norm(look), 1e-9)
                cam_xy = np.asarray(xy_map) - horiz * look
                if np.all(cam_xy > room_lo) and np.all(cam_xy < room_hi):
                    return look
            return look

    punches = doc.get("punches", [])
    trace = np.asarray(doc.get("trace", []), float)
    panel = None
    if doc.get("nav_trace"):
        _fig, panel = _belief_panel(doc, scene)
    stop = doc.get("stop")
    t_stop = float(stop["t"]) if stop else None
    banner = {"right_hand_on_second_wall": "RIGHT HAND ON WALL 2 -> STOP",
              "both_hands_lost_wall": "BOTH HANDS LOST THE WALL -> STOP",
              "turned_to_second_wall": "TURNED, WALL 2 UNDER THE FISTS -> STOP",
              "loop_complete": "LOOP COMPLETE -> STOP",
              "advanced_along_second_wall": "2 M ALONG WALL 2 -> STOP"}.get(
        stop["reason"] if stop else "", f"STOP: {stop['reason']}" if stop else "")
    events = [e for e in doc.get("events", [])
              if e["event"] in ("MOVE", "PUNCH", "GUARD", "REPLAN", "PASSIVE_CONTACT",
                                "ARM_TRANSITION", "ARM_CARRY")]
    marks = [e for e in doc.get("events", [])
             if e["event"] in ("FOUND", "PASSED", "REACQUIRED", "CORNER",
                               "SECOND_WALL", "CORNERS_DONE", "ADVANCED")]
    enc = None
    try:
        for n in range(0, len(ts), a.stride):
            t = float(ts[n])
            be.env.sim.data.qpos[0] = torch.from_numpy(qs[n]).to(be.device)
            be.env.sim.forward()
            if cam is not None:
                # A partial turn followed by its remainder is not two full
                # command turns. Camera motion follows recorded realized yaw.
                row = (trace[min(int(np.searchsorted(trace[:, 0], t)), len(trace) - 1)]
                       if len(trace) else None)
                turned = math.degrees(float(row[3]) - task["face_yaw_map"]) if row is not None else 0.0
                xy_map = row[1:3] if row is not None else np.zeros(2)
                az_target = _azimuth(_look_for(xy_map, turned))
                d_az = (az_target - cam.azimuth + 180.0) % 360.0 - 180.0
                cam.azimuth += float(np.clip(d_az, -2.2, 2.2))
            orr = be.env._offline_renderer
            orr.update(be.env.sim.data)
            rgb = np.ascontiguousarray(np.asarray(orr.render(), np.uint8)[:, :, :3])
            thrown = sum(1 for p in punches if p["t"] <= t)
            state = ""
            for e in events:
                if e["t"] <= t:
                    where = (f"station {e['k']} " if "k" in e else "")
                    state = f"{e['event']}  {where}({e.get('kind', '')})"
            # PUNCH events are logged when the primitive finishes. Its
            # recorded interval identifies the action while it is underway.
            active_punch = next((p for p in punches
                                 if p["t"] <= t < p["t"] + p.get("punch_s", 0.0)), None)
            if active_punch is not None:
                phase = ("ARM TRANSITION" if t < active_punch["t"] + active_punch.get("transition_s", 0.)
                         else "PUNCH")
                state = f"{phase} {active_punch['hand']} ({active_punch.get('kind', '')})"

            fl = fr = 0.0
            if len(trace):
                k = int(np.searchsorted(trace[:, 0], t))
                k = min(max(k, 0), len(trace) - 1)
                fl, fr = float(trace[k, 4]), float(trace[k, 5])
            lines = [(f"t = {t:5.1f} s   {doc['room']} / {doc['controller']}   "
                      f"punches {thrown}", (255, 255, 255)),
                     (state, (170, 230, 255)),
                     (f"left hand {fl:5.1f} N   right hand {fr:5.1f} N",
                      (255, 220, 120))]
            for e in marks:
                if e["t"] <= t <= e["t"] + 3.0:
                    lines.append((f"{e['event']}: {e.get('kind', '')} "
                                  f"{e.get('by', '') or ''}", (120, 255, 160)))
            if t_stop is not None and t >= t_stop:
                lines.append((banner, (255, 80, 80)))
            comp = _hud(rgb, lines)
            if panel is not None:
                right = panel(t)
                hh = min(comp.shape[0], right.shape[0])
                comp = np.hstack([comp[:hh], right[:hh]])
            comp = comp[:comp.shape[0] // 2 * 2, :comp.shape[1] // 2 * 2]
            if enc is None:
                H, W = comp.shape[:2]
                enc = subprocess.Popen(
                    [ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo",
                     "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r",
                     str(a.fps if a.fps else FPS / a.stride),
                     "-i", "-", "-an", "-c:v", "libx264",
                     "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
                     str(a.out)], stdin=subprocess.PIPE)
            enc.stdin.write(np.ascontiguousarray(comp).tobytes())
    finally:
        if enc is not None:
            enc.stdin.close()
            enc.wait()
        be.close()
    if enc is None or enc.returncode != 0:
        raise RuntimeError("Video encoder failed or no frames were recorded")
    n_out = len(range(0, len(ts), a.stride))
    print(f"{n_out} frames -> {a.out}  ({float(ts[-1]):.0f} s of episode)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
