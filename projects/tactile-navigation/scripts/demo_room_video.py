"""V3-style demo video: MuJoCo follow view LEFT, live belief map RIGHT.

    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \\
      tactile_controller/.venv/bin/python scripts/demo_room_video.py \\
        --run out/demo_map/run/demo_run_s0.json \\
        --states out/demo_map/run/states_demo_s0.npz \\
        --out out/demo_map/run/demo_room.mp4

LEFT.  Offline replay of the recorded qpos (physics does not re-run) through a
TRACKING camera.  On followed walls the camera stands in the ROOM, on the
lane's open side, looking at the wall -- the robot, the standoff and the hand
contact are all in frame and no wall ever stands between camera and robot.
Between walls it falls back to a slewed follow-from-behind.

RIGHT.  The scan-frame map with everything the run logged: the drawn route,
the realized trajectory (truth, evaluation only), the belief trail, the PF
modes sized by mass, the live UNCERTAINTY AREA, and each touch flashing at the
moment it lands with its peak force.

THE UNCERTAINTY AREA is the particle cloud's 2-sigma covariance ELLIPSE, drawn
filled and updated every frame.  It replaces a circle of radius 2-sigma, and
the difference is the whole point of the demo: after a flat-wall press the
cloud is squeezed hard along the face normal and left untouched along the face
-- the rank-1 result -- so the honest picture is a long thin ellipse lying
ALONG the wall.  A circle reports the average of the two axes and shows
neither, so it hid exactly the thing a landmark touch then fixes: watch the
ellipse collapse from a slot into a disc when the hand finds a corner, a bulge
or a protrusion.  The strip below plots its AREA (pi*a*b, m^2) against time on
its own axis, so the collapse is legible as a number as well as a shape.

A ZOOM INSET follows the robot, because on a 20 m room map a 0.7 m ellipse is
3% of the frame and the collapse is invisible at that scale -- the number in
the HUD would be the only evidence of the thing the picture exists to show.
The inset is the same artists over a 2.4 m window, so nothing in it is a
second rendering that could disagree with the map.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FPS = 25.0


def _load_run(run_path: Path):
    doc = json.loads(run_path.read_text())
    nav = doc["nav_trace"]
    nav_t = np.array([x["t"] for x in nav])
    bel = np.array([x["bel_xy"] for x in nav])
    sig = np.array([x["sigma_m"] for x in nav])
    # Runs recorded before the covariance was logged still play: an isotropic
    # cloud with the same sigma is the only honest reconstruction from a
    # scalar, and it reproduces exactly the circle those runs used to draw.
    cov = np.array([np.asarray(x["cov_xy"], float) if x.get("cov_xy")
                    else np.eye(2) * (float(x["sigma_m"]) ** 2 / 2.0)
                    for x in nav])
    return doc, nav, nav_t, bel, sig, cov


def _ellipse_axes(cov, n_sigma: float = 2.0):
    """(width, height, angle_deg, area) of the n-sigma covariance ellipse."""
    w, v = np.linalg.eigh(np.asarray(cov, float))
    w = np.maximum(w, 1e-9)
    a, b = n_sigma * np.sqrt(w[1]), n_sigma * np.sqrt(w[0])   # eigh: ascending
    ang = math.degrees(math.atan2(float(v[1, 1]), float(v[0, 1])))
    return 2.0 * a, 2.0 * b, ang, float(np.pi * a * b)


def _interp(ts, xs, t):
    i = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
    a, b = ts[i - 1], ts[i]
    w = 0.0 if b <= a else float(np.clip((t - a) / (b - a), 0.0, 1.0))
    return xs[i - 1] * (1 - w) + xs[i] * w, i


def _scene_for(doc):
    if doc.get("walls_json"):
        # a scanned room: the record names the map it ran, so the replay
        # must rebuild THAT scene, not the demo room's
        from demo_map.scene_scan import build_corner_room
        return build_corner_room(doc["walls_json"])[1]
    if str(doc.get("scene", "")).startswith("room2"):
        from demo_map.scene_room2 import build_room2_scene
        return build_room2_scene(bool(doc.get("reverse")))
    from demo_map.scene_demo import build_demo_scene
    return build_demo_scene()


def _run_backend(doc, *, seed: int, standoff_m: float, **cfg_over):
    """The backend built by the script that produced this run."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    scene = _scene_for(doc)
    controller = str(doc.get("controller") or "hand_force")
    if str(doc.get("scene", "")).startswith("room2"):
        from room2_run import backend as make
        return make(scene, seed, standoff_m,
                    crab_ff_m=0.20 if doc.get("dense_showcase") else 0.0,
                    render=True, video_path=None, controller=controller,
                    **cfg_over)
    from demo_room_run import backend as make
    return make(scene, seed, standoff_m, render=True, video_path=None,
                controller=controller, **cfg_over)


def build_right_panel(doc, d, size=(7.2, 7.2)):
    """Static map figure + the dynamic artists, scan frame.  A thin strip
    under the map plots sigma and |truth-belief| against time -- the
    belief-vs-truth EVOLUTION, not just its instantaneous circle."""
    scene = _scene_for(doc)
    # Three cells, not two.  The zoom inset used to sit INSIDE the map, and
    # no corner works for both rooms: the demo room's corridor runs along the
    # bottom and its branch up the right, room2 is an L in the lower left, and
    # the two free corners are already the legend's and the HUD's.  Giving the
    # inset its own cell costs a little map height and can never cover the
    # route.
    fig = plt.figure(figsize=size, dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[4.2, 1.45],
                          width_ratios=[3.05, 1.0], hspace=0.24, wspace=0.16)
    ax = fig.add_subplot(gs[0, :])
    axs = fig.add_subplot(gs[1, 0])
    axz = fig.add_subplot(gs[1, 1])
    for b in scene.plan.bodies:
        v = np.asarray(b.verts, float) + d
        ax.fill(v[:, 0], v[:, 1],
                color="#8a8072" if "wall" in b.name
                else "#9c5f8f" if "bulge" in b.name else "#b98a4b", lw=0)
    for f in scene.plan.features:
        q = np.asarray(f.xy, float) + d
        ax.plot(*q, "o", ms=9, mfc="none", mec="#2ca02c", mew=1.8, alpha=0.9)
    route = np.asarray(doc["route_polyline_map"], float) + d
    ax.plot(route[:, 0], route[:, 1], color="#d62728", lw=1.6, alpha=0.75,
            label="planned route")
    s = np.asarray([0.0, 0.0]) + d
    g = np.asarray(doc.get("goal_map", scene.goal_map_xy), float) + d
    ax.plot(*s, "o", ms=13, color="#2ca02c", zorder=6)
    ax.plot(*g, "o", ms=13, color="#d62728", alpha=0.75, zorder=6)

    truth_ln, = ax.plot([], [], color="#222222", lw=2.4,
                        label="realized trajectory")
    bel_ln, = ax.plot([], [], color="#1f77b4", lw=1.8, ls=(0, (3, 2)),
                      label="belief trail")
    bel_dot, = ax.plot([], [], "o", ms=8, color="#1f77b4", zorder=7)
    truth_dot, = ax.plot([], [], "o", ms=7, color="#222222", zorder=7)
    sig_fill = Ellipse((0, 0), 0.2, 0.2, angle=0.0, facecolor="#1f77b4",
                       alpha=0.16, edgecolor="none", zorder=3,
                       label="belief uncertainty (2$\\sigma$)")
    sig_edge = Ellipse((0, 0), 0.2, 0.2, angle=0.0, fill=False,
                       edgecolor="#1f77b4", lw=1.3, alpha=0.85, zorder=4)
    ax.add_patch(sig_fill)
    ax.add_patch(sig_edge)
    modes = ax.scatter([], [], s=[], facecolors="none", edgecolors="#9467bd",
                       linewidths=1.6, zorder=6, label="PF modes")
    touch_done = ax.scatter([], [], s=44, marker="X", color="#2ca02c",
                            zorder=8, label="touch")
    flash = ax.scatter([], [], s=420, marker="o", facecolors="none",
                       edgecolors="#ff7f0e", linewidths=2.6, zorder=9)
    hud = ax.text(0.985, 0.985, "", transform=ax.transAxes, va="top",
                  ha="right", fontsize=9.5, family="monospace",
                  bbox=dict(fc="white", alpha=0.85, ec="none"))
    note = ax.text(0.985, 0.02, "", transform=ax.transAxes, va="bottom",
                   ha="right", fontsize=10, color="#ff7f0e",
                   fontweight="bold",
                   bbox=dict(fc="white", alpha=0.85, ec="none"))
    ax.set_aspect("equal")
    ax.set_xlim(route[:, 0].min() - 3.4, route[:, 0].max() + 3.6)
    ax.set_ylim(route[:, 1].min() - 2.6, route[:, 1].max() + 3.2)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=7.6, framealpha=0.9)
    ax.set_title("belief vs truth", fontsize=11)

    # ── the zoom inset: the SAME artists over a 2.4 m window, so nothing in
    # it is a second rendering that could disagree with the map
    for b in scene.plan.bodies:
        v = np.asarray(b.verts, float) + d
        axz.fill(v[:, 0], v[:, 1],
                 color="#8a8072" if "wall" in b.name
                 else "#9c5f8f" if "bulge" in b.name else "#b98a4b", lw=0)
    axz.plot(route[:, 0], route[:, 1], color="#d62728", lw=1.2, alpha=0.7)
    z_fill = Ellipse((0, 0), 0.2, 0.2, angle=0.0, facecolor="#1f77b4",
                     alpha=0.30, edgecolor="none", zorder=3)
    z_edge = Ellipse((0, 0), 0.2, 0.2, angle=0.0, fill=False,
                     edgecolor="#1f77b4", lw=1.6, zorder=4)
    axz.add_patch(z_fill)
    axz.add_patch(z_edge)
    z_bel, = axz.plot([], [], "o", ms=7, color="#1f77b4", zorder=7)
    z_truth, = axz.plot([], [], "o", ms=6, color="#222222", zorder=7)
    z_modes = axz.scatter([], [], s=[], facecolors="none", edgecolors="#9467bd",
                          linewidths=1.4, zorder=6)
    axz.set_aspect("equal")
    axz.set_xticks([])
    axz.set_yticks([])
    for sp in axz.spines.values():
        sp.set_edgecolor("#1f77b4")
        sp.set_linewidth(1.2)
    axz.set_title("belief, 2.4 m window", fontsize=7.5, color="#1f77b4",
                  pad=2)
    axz.set_facecolor("#fbfbfd")

    nav_t = np.array([x["t"] for x in doc["nav_trace"]])
    sig_all = np.array([x["sigma_m"] for x in doc["nav_trace"]])
    err_all = np.array([x["err_m"] for x in doc["nav_trace"]])
    axs.plot(nav_t, sig_all, color="#1f77b4", lw=1.6, label="sigma (belief)")
    axs.plot(nav_t, err_all, color="#222222", lw=1.4,
             label="|truth - belief|  (eval)")
    for tc in doc["touches"]:
        if tc.get("contacted"):
            axs.axvline(tc["t"], color="#2ca02c", lw=0.8, alpha=0.5)
    # NO 2-SIGMA AREA CURVE.  It was a second trace on a twin m^2 axis under
    # the same panel as sigma and the error (owner, 2026-08-31: removed).  The
    # ellipse on the map already carries the area, and two y-axes under one
    # time series made the panel read as three quantities where the question
    # is only ever "is the error inside the belief".
    cursor = axs.axvline(0.0, color="#ff7f0e", lw=1.6, zorder=5)
    axs.set_xlim(0.0, max(float(nav_t[-1]), 1.0))
    axs.set_ylim(0.0, max(float(sig_all.max()), float(err_all.max())) * 1.15)
    axs.set_xlabel("t  [s]  (green ticks = touches)", fontsize=8)
    axs.set_ylabel("m", fontsize=8)
    axs.tick_params(labelsize=7.5)
    axs.grid(alpha=0.25)
    axs.legend(loc="upper right", fontsize=7)
    fig.subplots_adjust(left=0.085, right=0.955, top=0.955, bottom=0.09)
    art = dict(truth_ln=truth_ln, bel_ln=bel_ln, bel_dot=bel_dot,
               truth_dot=truth_dot, sig_fill=sig_fill, sig_edge=sig_edge,
               modes=modes, touch_done=touch_done, flash=flash, hud=hud,
               note=note, cursor=cursor, axz=axz, z_fill=z_fill,
               z_edge=z_edge, z_bel=z_bel, z_truth=z_truth, z_modes=z_modes)
    return fig, art


def draw_right(fig, art, doc, d, nav, nav_t, bel, sig, cov, truth_hist, t):
    truth = np.asarray(truth_hist, float)
    art["truth_ln"].set_data(truth[:, 0] + d[0], truth[:, 1] + d[1])
    art["truth_dot"].set_data([truth[-1, 0] + d[0]], [truth[-1, 1] + d[1]])
    k = int(np.searchsorted(nav_t, t))
    bl = bel[:max(k, 1)]
    art["bel_ln"].set_data(bl[:, 0] + d[0], bl[:, 1] + d[1])
    b_now, i = _interp(nav_t, bel, t)
    s_now, _ = _interp(nav_t, sig, t)
    # elementwise interpolation of two covariances is their convex
    # combination, which stays positive semi-definite -- so the ellipse can be
    # animated at video rate between belief updates without ever going
    # imaginary on a frame that falls between two absorbs
    c_now, _ = _interp(nav_t, cov.reshape(len(cov), 4), t)
    w_e, h_e, ang, area = _ellipse_axes(c_now.reshape(2, 2))
    art["bel_dot"].set_data([b_now[0] + d[0]], [b_now[1] + d[1]])
    for art_key in ("sig_fill", "sig_edge", "z_fill", "z_edge"):
        e = art[art_key]
        e.set_center((b_now[0] + d[0], b_now[1] + d[1]))
        e.set_width(max(w_e, 0.02))
        e.set_height(max(h_e, 0.02))
        e.set_angle(ang)
    # the inset window follows the BELIEF, not the truth: it is the belief's
    # own picture of where it is, and centring it on truth would quietly leak
    # the answer into the panel that is supposed to be judging it
    half = 1.2
    art["axz"].set_xlim(b_now[0] + d[0] - half, b_now[0] + d[0] + half)
    art["axz"].set_ylim(b_now[1] + d[1] - half, b_now[1] + d[1] + half)
    art["z_bel"].set_data([b_now[0] + d[0]], [b_now[1] + d[1]])
    art["z_truth"].set_data([truth[-1, 0] + d[0]], [truth[-1, 1] + d[1]])
    md = nav[i].get("modes", [])
    if md:
        art["modes"].set_offsets(np.array([[m["xy"][0] + d[0],
                                            m["xy"][1] + d[1]] for m in md]))
        art["modes"].set_sizes([60 + 500 * m["mass"] for m in md])
        art["z_modes"].set_offsets(np.array([[m["xy"][0] + d[0],
                                              m["xy"][1] + d[1]] for m in md]))
        art["z_modes"].set_sizes([40 + 340 * m["mass"] for m in md])
    done, flash_pts, note = [], [], ""
    for tc in doc["touches"]:
        if tc["t"] <= t and tc.get("contacted"):
            done.append([tc["xy_map"][0] + d[0], tc["xy_map"][1] + d[1]])
        if abs(tc["t"] - t) < 3.0 and tc.get("contacted"):
            flash_pts.append([tc["xy_map"][0] + d[0], tc["xy_map"][1] + d[1]])
            note = (f"TOUCH  {tc['label']}  "
                    f"{tc.get('peak_force_n', 0) or 0:.1f} N")
    art["touch_done"].set_offsets(np.array(done) if done
                                  else np.empty((0, 2)))
    art["flash"].set_offsets(np.array(flash_pts) if flash_pts
                             else np.empty((0, 2)))
    art["note"].set_text(note)
    art["cursor"].set_xdata([t, t])
    err = float(np.linalg.norm(np.asarray(truth[-1]) - b_now))
    n_hit = sum(1 for tc in doc["touches"]
                if tc["t"] <= t and tc.get("contacted"))
    # the ANISOTROPY, not just the size: a 5:1 ellipse says the last touch
    # pinned one axis and left the other alone, which is the rank-1 story
    ratio = (w_e / h_e) if h_e > 1e-6 else float("inf")
    art["hud"].set_text(f"t {t:6.1f} s\n"
                        f"sigma {float(s_now):5.3f} m\n"
                        f"2sig axes {w_e / 2:5.3f} x {h_e / 2:5.3f} m ({ratio:.1f}:1)\n"
                        f"2sig area {area:6.4f} m2\n"
                        f"|truth-belief| {err:5.3f} m\n"
                        f"touches {n_hit}")
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(),
                         np.uint8).reshape(h, w, 4)[:, :, :3]




def _scene_marker(scn, pos, size, rgba, geom_type):
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, geom_type, np.asarray(size, float),
                        np.asarray(pos, float), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1


def _scene_arrow(scn, p_from, p_to, rgba, width=0.016):
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                        np.zeros(3), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, float(width),
                         np.asarray(p_from, float), np.asarray(p_to, float))
    scn.ngeom += 1


def _touch_normals(doc):
    """Per touch: (t, physical map point, outward face normal, force).

    The touched face's outward normal comes from the planned follow segment
    whose name appears in the touch label; the map point is shifted by the
    registration offset so the arrow lands on the PHYSICAL wall the viewer
    sees."""
    segs = _camera_faces(doc)
    delta = float(doc.get("true_delta_m", 0.0) or 0.0)
    out = []
    for tc in doc.get("touches", []):
        if not tc.get("contacted"):
            continue
        n = None
        for sg in segs:
            if sg.get("name") and sg["name"] in str(tc.get("label", "")):
                n = np.asarray(sg["normal"], float)
                break
        if n is None and segs:
            p = np.asarray(tc["xy_map"], float)
            sg = min(segs, key=lambda q: abs(
                (p - np.asarray(q["p0"], float))
                @ np.asarray(q["normal"], float)))
            n = np.asarray(sg["normal"], float)
        if n is None:
            continue
        out.append((float(tc["t"]),
                    np.asarray(tc["xy_map"], float) + np.array([delta, 0.0]),
                    n, float(tc.get("peak_force_n") or 0.0)))
    return out


def _fit_camera(doc, distance, elevation):
    """Frame the ROOM, not just the robot.

    The camera tracks the base, so its distance decides how much of the
    floorplan is ever visible.  At the old fixed 2.9 m it stands INSIDE a
    4.6 m room: the nearest wall fills the frame, the rest is off-screen, and
    the video looks like a G1 walking on an empty checkerboard -- the room is
    fully built and placed, it is simply not in shot.

    So the default is read off the scene's own footprint.  An explicit
    --distance / --elevation still wins.
    """
    if distance is not None and elevation is not None:
        return float(distance), float(elevation)
    span = 6.0
    try:
        scene = _scene_for(doc)
        pts = np.concatenate([np.asarray(b.verts, float)
                              for b in scene.plan.bodies])
        span = float(max(pts[:, 0].ptp(), pts[:, 1].ptp()))
    except Exception:                        # noqa: BLE001 - fall back to 6 m
        pass
    d = float(distance) if distance is not None else \
        float(np.clip(1.5 * span, 5.0, 16.0))
    e = float(elevation) if elevation is not None else -40.0
    return d, e


def _camera_faces(doc):
    """Every face the camera may frame the robot against, followed or not.

    Not just the `follow` segments.  Under the ATE compile the robot spends
    whole minutes at a landmark -- pressing a bulge, feeling around a corner
    -- with no followed wall within reach, and the camera then fell back to
    look-from-behind and filled the frame with the wall it was standing next
    to.  A landmark visit gets the same treatment as a wall: its apex-tangent
    line IS the face the robot is working against, so hand it to the same
    room-side rule.  Corner episodes contribute both of their faces."""
    out = []
    for st in doc.get("planned_steps", []):
        k = st.get("kind")
        if k == "follow" and isinstance(st.get("seg"), dict):
            out.append(st["seg"])
        elif k == "ate_corner":
            out += [f["seg"] for f in st.get("faces", [])
                    if isinstance(f.get("seg"), dict)]
        elif k in ("ate_landmark", "ate_pillar"):
            n = np.asarray(st.get("normal", (0.0, -1.0)), float)
            n = n / max(float(np.linalg.norm(n)), 1e-9)
            t = np.array([-n[1], n[0]])
            reach = float(st.get("reach_m") or st.get("radius") or 0.0)
            apex = np.asarray(st["centre"], float) + reach * n
            out.append({"p0": (apex - 1.13 * t).tolist(),
                        "tangent": t.tolist(), "normal": n.tolist(),
                        "length_m": 2.03,
                        "name": f"{st.get('label', 'landmark')}:apex"})
    return out


def follow_azimuth(doc, be, xy_map, yaw_map, *, distance: float,
                   elevation: float):
    """Camera azimuth target: from the room side of the face being worked on,
    else from behind the direction of travel."""
    from execution.wall_lane import raycast

    # EVERY face in range, not the first one.  At a corner the robot is close
    # to BOTH walls, and taking whichever came first in plan order put the
    # camera on the room side of one wall and behind the other -- the reverse
    # demo lost the robot entirely for two stretches, once through the corner
    # kiss, which is the moment the video exists to show.  Averaging the
    # normals gives the bisector, which points into the open corner rather
    # than through either face.
    hits = []
    for seg in _camera_faces(doc):
        p0 = np.asarray(seg["p0"], float)
        tg = np.asarray(seg["tangent"], float)
        nm = np.asarray(seg["normal"], float)
        s = float((np.asarray(xy_map) - p0) @ tg)
        dlat = float((np.asarray(xy_map) - p0) @ nm)
        if -0.6 <= s <= seg["length_m"] + 0.6 and 0.0 < dlat < 1.2:
            hits.append((tg, nm, dlat))
    if not hits:
        look = np.array([math.cos(yaw_map), math.sin(yaw_map)])
        look = look[0] * be.u + look[1] * be.v
        return math.degrees(math.atan2(float(look[1]), float(look[0])))

    nm = np.sum([h[1] for h in hits], axis=0)
    n_len = float(np.linalg.norm(nm))
    nm = nm / n_len if n_len > 1e-9 else hits[0][1]
    # the tangent of the NEAREST face still sets the look-down-the-lane
    # component; only the room side is shared between them
    tg = min(hits, key=lambda h: h[2])[0]

    # HOW FAR THE CAMERA MAY STAND OFF THE LANE IS THE ROOM'S ANSWER, NOT A
    # CONSTANT.  A tracking camera sits `distance * cos(elevation)` away
    # horizontally -- 2.49 m at the defaults -- so a fixed normal weight fixes
    # the LATERAL offset too, and the previous 0.50/0.87 split put it 1.25 m
    # off the lane.  The demo room's hallway is 1.68 m wide and the robot
    # follows it at 0.49, leaving 1.19 m: the camera stood 0.37 m THROUGH the
    # far wall and every frame showed that wall's back with the robot's head
    # above it.  Measure the free room along the camera's own bearing instead
    # and stay inside it.
    horiz = max(float(distance) * math.cos(math.radians(abs(elevation))), 0.3)
    d_room, _n, _h = raycast(np.asarray(xy_map, float), nm, horiz + 1.0,
                             tuple(be.scene.plan.bodies))
    room = horiz if d_room is None else float(d_room)
    lat = float(np.clip(room - 0.45, 0.30, 1.35))
    sin_t = float(np.clip(lat / horiz, 0.10, 0.98))
    cos_t = math.sqrt(max(1.0 - sin_t * sin_t, 0.0))
    # the camera sits `lat` into the room and `cos_t*horiz` along the lane,
    # looking back at the robot and the face it is touching
    look_map = -(sin_t * nm + cos_t * tg)
    look = look_map[0] * be.u + look_map[1] * be.v
    return math.degrees(math.atan2(float(look[1]), float(look[0])))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=1,
                    help="render every Nth recorded frame")
    ap.add_argument("--fps", type=float, default=None,
                    help="output frame rate.  The default keeps REAL TIME "
                         "(25/stride), so a strided render still plays at the "
                         "speed the robot moved.  Setting it higher plays the "
                         "episode faster -- a 27 m route at 0.3 m/s is a long "
                         "watch, and the touches are still legible at 2-4x.")
    ap.add_argument("--distance", type=float, default=None,
                    help="tracking-camera distance [m].  Default FITS THE "
                         "ROOM: a fixed 2.9 m sits inside these rooms and "
                         "frames the nearest wall only, which reads as 'the "
                         "room was never rendered'.")
    ap.add_argument("--elevation", type=float, default=None,
                    help="tracking-camera elevation [deg].  Default -40 with "
                         "a fitted distance: looking further down is what "
                         "puts the floor plan in frame.")
    a = ap.parse_args(argv)

    import torch

    doc, nav, nav_t, bel, sig, cov = _load_run(a.run)
    d = np.asarray(doc["scan_start_offset"], float)
    rec = np.load(a.states)
    ts, qs = rec["t"], rec["qpos"]

    # THE REPLAY MUST REBUILD THE RUN'S OWN MAP FRAME.  `base0` (where the env
    # reset) and `u`/`v` (its facing) define every world<->map conversion in
    # this file: the realized trajectory drawn on the right panel, the goal
    # disc, the touch arrows, and the tracking camera's bearing.  Rebuilding
    # the backend from a config of the video's own devising put base0 0.64 m
    # off and the heading 173 deg around -- the camera then framed the far
    # side of a wall for whole stretches, with the robot behind it.  So the
    # backend comes from the RUN SCRIPT'S factory, and anything the run
    # recorded about its frame overrides even that.
    cam_d, cam_e = _fit_camera(doc, a.distance, a.elevation)
    print(f"camera: distance {cam_d:.1f} m  elevation {cam_e:.0f} deg"
          f"{'' if a.distance else '  (fitted to the room)'}", flush=True)
    be = _run_backend(doc, seed=int(doc.get("seed", 0)),
                      standoff_m=float(doc.get("standoff_m", 0.49)),
                      video_width=960, video_height=720,
                      camera_distance=cam_d,
                      camera_elevation=cam_e)
    frame = doc.get("spawn_frame")
    if frame:
        be.base0 = np.asarray(frame["base0"], float)
        be.u = np.asarray(frame["u"], float)
        be.v = np.asarray(frame["v"], float)
    print(f"map frame: base0 {np.round(np.asarray(be.base0), 4)} "
          f"u {np.round(np.asarray(be.u), 3)}"
          f"{'  (from the run record)' if frame else '  (rebuilt)'}",
          flush=True)
    cam = getattr(getattr(be.env, "_offline_renderer", None), "_cam", None)
    if cam is not None:
        import mujoco
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
        cam.trackbodyid = be.robot.indexing.root_body_id
        cam.fixedcamid = -1
        cam.elevation = cam_e
        cam.distance = cam_d

    rfig, art = build_right_panel(doc, d)
    delta = float(doc.get("true_delta_m", 0.0) or 0.0)
    start_w = np.asarray(be.base0, float)          # the physical spawn
    goal_map = doc.get("goal_map")
    if goal_map is None:                           # older run jsons
        goal_map = be.scene.goal_map_xy
    goal_w = np.asarray(be.map_point_to_world(
        np.asarray(goal_map, float)
        + np.array([delta, 0.0])), float)          # the physical goal
    touch_arrows = _touch_normals(doc)
    az, enc, truth_hist = None, None, []
    try:
        for n in range(0, len(ts), a.stride):
            t = float(ts[n])
            be.env.sim.data.qpos[0] = torch.from_numpy(qs[n]).to(be.device)
            be.env.sim.forward()
            xy_w = qs[n][:2]
            xy_map = be.world_point_to_map(xy_w)
            qw, qx, qy, qz = qs[n][3:7]
            yaw_w = math.atan2(2 * (qw * qz + qx * qy),
                               1 - 2 * (qy * qy + qz * qz))
            yaw_map = be.world_yaw_to_map(yaw_w)
            truth_hist.append(xy_map)
            if cam is not None:
                target = follow_azimuth(doc, be, xy_map, yaw_map,
                                        distance=cam_d,
                                        elevation=cam_e)
                az = target if az is None else az + float(
                    np.clip((target - az + 180.0) % 360.0 - 180.0, -2.2, 2.2))
                cam.azimuth = az
            orr = be.env._offline_renderer
            orr.update(be.env.sim.data)
            scn = orr.renderer.scene
            import mujoco as _mj
            # start (green) and goal (red) FLOOR DISCS -- decoration geoms.
            # These were translucent pillars, and the goal pillar stood at
            # the exact spot the robot finishes on, so the render showed the
            # robot "penetrating" a cylinder (owner).  A 1 cm disc underfoot
            # cannot intersect anything.
            _scene_marker(scn, [*start_w, 0.012], (0.16, 0.012, 0),
                          (0.15, 0.75, 0.25, 0.9), _mj.mjtGeom.mjGEOM_CYLINDER)
            _scene_marker(scn, [*goal_w, 0.012], (0.16, 0.012, 0),
                          (0.90, 0.15, 0.15, 0.9), _mj.mjtGeom.mjGEOM_CYLINDER)
            # each touch: an arrow along the face's outward normal, length
            # and colour scaled by the peak force (25 N = the hard limit)
            for t_tc, p_tc, n_tc, f_tc in touch_arrows:
                if not (t_tc - 3.0 <= t <= t_tc + 3.0):
                    continue
                p_w = np.asarray(be.map_point_to_world(p_tc), float)
                n_w = (np.asarray(be.map_point_to_world(p_tc + n_tc), float)
                       - p_w)
                frac = min(f_tc / 25.0, 1.0)
                # AT THE HAND, not on the floor.  `probe_hand_z_m` is 0.15 in
                # the PELVIS frame and was being used as an absolute world
                # height, which drew every touch arrow down by the pelvis's
                # own 0.75 m -- the force showed up at the robot's shin
                # instead of where the hand met the wall.
                z_touch = (float(be.robot.data.root_link_pos_w[0, 2])
                           + be.mj.probe_hand_z_m)
                a_from = np.array([p_w[0], p_w[1], z_touch])
                a_to = a_from + (0.12 + 0.020 * f_tc) * np.array(
                    [n_w[0], n_w[1], 0.0])
                _scene_arrow(scn, a_from, a_to,
                             (0.2 + 0.8 * frac, 0.85 * (1.0 - frac) + 0.1,
                              0.10, 0.95))
            rgb = np.ascontiguousarray(
                np.asarray(orr.render(), np.uint8)[:, :, :3])
            right = draw_right(rfig, art, doc, d, nav, nav_t, bel, sig,
                               cov, truth_hist, t)
            hh = min(rgb.shape[0], right.shape[0])
            comp = np.hstack([rgb[:hh], right[:hh]])
            comp = comp[:comp.shape[0] // 2 * 2, :comp.shape[1] // 2 * 2]
            if enc is None:
                H, W = comp.shape[:2]
                enc = subprocess.Popen(
                    ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
                     "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r",
                     str(a.fps if a.fps else FPS / a.stride),
                     "-i", "-", "-an", "-c:v", "libx264",
                     "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
                     str(a.out)], stdin=subprocess.PIPE)
            enc.stdin.write(comp.tobytes())
    finally:
        if enc is not None:
            enc.stdin.close()
            enc.wait()
        be.close()
    rate = a.fps if a.fps else FPS / a.stride
    n_out = len(ts) // a.stride
    print(f"{n_out} frames @ {rate:g} fps -> {a.out}  "
          f"({n_out / rate:.0f} s of video for {float(ts[-1]):.0f} s of "
          f"episode, {float(ts[-1]) / max(n_out / rate, 1e-6):.1f}x)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
