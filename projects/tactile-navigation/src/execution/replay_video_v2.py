"""Clarity re-render (v2) of a recorded contact-rich episode.  REPLAY ONLY:
states come from ``states.npz``, the render env runs kinematics (``forward``),
physics never re-runs, no result changes.

Three-panel layout (1920x720):

    LEFT   overhead floorplan: planned route, RAW realized path (faint) +
           SMOOTHED visualization trace (bold, labeled), numbered touch
           flashes, old/new planned-turn points with the changed route
           segment highlighted
    CENTER close-follow camera (mjlab tracking camera, distance ~1.9 m,
           azimuth slewed to look over the robot's shoulder at the wall)
    RIGHT  simplified status: current mode, belief (masses + sigma),
           planned turn, valid/invalid observation counters, and the
           "touch contradicts wrong hypothesis" banner

Smoothing (visualization ONLY -- raw data untouched, and the raw path is
drawn alongside): tick-trace xy resampled uniformly at 10 Hz, then a
centered moving average over a 2.5 s window.  The window is far shorter
than any leg or stop, so turns, stops and route changes survive; gait
oscillation (~1-2 Hz) and stabilize wiggles do not.

    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
      uv run --project tactile_controller python -m execution.replay_video_v2 \
        --episode out/contact_rich/_keep/mj_s0 --scene contact_rich_gap \
        --out out/contact_rich/_keep/mj_s0/demo_v2_full.mp4
    # --still-only writes overhead_routes.png without building the render env
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

MODE_NAMES = {"LOCOMOTION": "WALK", "STABILIZE": "STABILIZE", "PROBE": "PROBE",
              "TURN": "TURN", "ARRIVED": "ARRIVE", "FAILED": "FAILED"}
MODE_COLOR = {"WALK": "#2b8cbe", "STABILIZE": "#a6bddb", "PROBE": "#e34a33",
              "TURN": "#31a354", "ARRIVE": "#756bb1", "FAILED": "#636363",
              "TOUCH": "#e6873c", "LANDMARK_PROBE": "#c51b8a"}
FLASH_S = 1.6
FLIP_SHOW_S = 2.0


def _fit_w2m(ep):
    """world->map affine from the episode's own truth samples (lstsq, the
    same method as plots_contact_rich; residual sub-mm)."""
    tr = ep["tick_trace"]
    tt = np.asarray([r["t"] for r in tr])
    txy = np.asarray([r["xy"] for r in tr])
    tmap = np.asarray(ep.get("truth_map_xy", []))
    ht = np.asarray([h["t"] for h in ep["hypothesis_trace"]])
    if len(tmap) == len(ht) and len(tmap) >= 3:
        tw = np.asarray([txy[max(np.searchsorted(tt, t + 1e-9) - 1, 0)] for t in ht])
        A = np.c_[tw, np.ones(len(tw))]
        sx, *_ = np.linalg.lstsq(A, tmap[:, 0], rcond=None)
        sy, *_ = np.linalg.lstsq(A, tmap[:, 1], rcond=None)
        return lambda p: np.array([np.r_[np.asarray(p, float), 1.0] @ sx,
                                   np.r_[np.asarray(p, float), 1.0] @ sy])
    return lambda p: np.asarray(p, float)


class Telemetry2:
    def __init__(self, ep: dict, scene):
        tr = ep["tick_trace"]
        self.t = np.asarray([r["t"] for r in tr])
        self.xy_w = np.asarray([r["xy"] for r in tr])
        self.yaw = np.asarray([r.get("yaw", 0.0) for r in tr])
        self.mode = [MODE_NAMES.get(r["mode"], r["mode"]) for r in tr]
        self.hyp = ep["hypothesis_trace"]
        self.hyp_t = np.asarray([h["t"] for h in self.hyp])
        self.directives = ep.get("directives", [])
        self.dir_t = np.asarray([d["t"] for d in self.directives])
        self.disambig = set(getattr(scene, "disambiguating_stations", ()) or ())
        self.scene, self.ep = scene, ep
        self.w2m = _fit_w2m(ep)
        self.path_map = np.asarray([self.w2m(p) for p in self.xy_w])

        # smoothed visualization trace: uniform 10 Hz resample + 2.5 s
        # centered moving average
        tu = np.arange(self.t[0], self.t[-1], 0.1)
        rx = np.interp(tu, self.t, self.path_map[:, 0])
        ry = np.interp(tu, self.t, self.path_map[:, 1])
        k = 25
        ker = np.ones(k) / k
        pad = k // 2
        sx = np.convolve(np.pad(rx, pad, mode="edge"), ker, "valid")[:len(tu)]
        sy = np.convolve(np.pad(ry, pad, mode="edge"), ker, "valid")[:len(tu)]
        self.smooth_t, self.smooth = tu, np.c_[sx, sy]

        # presses: the probes list records VALID presses only (the invalid
        # branch continues past the append); invalid attempts are recovered
        # from their controller-owned failure records, placed at the station
        # the directive of that moment was probing
        self.presses = []
        nv = 0
        for p in ep["probes"]:
            st = scene.stations.get(p["station"])
            if st is None:
                continue
            nv += 1
            self.presses.append({
                "t": p["t"], "kind": p["kind"], "valid": True, "num": nv,
                "force": p["peak_force_n"], "facing": st.facing,
                "travel": st.travel,
                "xy": st.xy + float(p["offset_m"]) * st.travel})
        ni = 0
        invalid_reasons = ("probe_sweep_stopped_short",
                           "sweep_stopped_short_of_mapped_face",
                           "invalid_probe")
        for f in ep.get("failures", []):
            if f.get("owner") != "controller" \
                    or f.get("reason") not in invalid_reasons:
                continue
            tf = float(f.get("time_s", 0.0))
            k = int(np.clip(np.searchsorted(self.dir_t, tf + 1e-9) - 1, 0,
                            max(len(self.directives) - 1, 0))) \
                if self.directives else -1
            stn = scene.stations.get(self.directives[k].get("station")) \
                if k >= 0 else None
            i = max(np.searchsorted(self.t, tf + 1e-9) - 1, 0)
            xy = stn.xy if stn is not None else self.path_map[i]
            ni += 1
            f_dir = stn.facing if stn is not None else np.array([0.0, 1.0])
            t_dir = stn.travel if stn is not None else np.array([1.0, 0.0])
            self.presses.append({"t": tf, "kind": "invalid", "valid": False,
                                 "num": ni, "force": 0.0, "xy": xy,
                                 "facing": f_dir, "travel": t_dir})
        self.presses.sort(key=lambda p: p["t"])
        self.n_valid = nv
        self.n_invalid = ni

        # planned-turn series + flip events (jump > 0.20 m on the same leg)
        self.turn_pts, self.flips = [], []
        prev = None
        for h in self.hyp:
            if not h.get("masses") or h["leg"] >= len(scene.legs) - 1:
                self.turn_pts.append(None)
                prev = None
                continue
            leg = scene.legs[h["leg"]]
            a, u = np.asarray(leg["origin"], float), np.asarray(leg["u"], float)
            turn_s = float(leg["terminus"]) + scene.standoff_m
            s_trk = float(h["hyp_s"][int(np.argmax(h["masses"]))])
            i = max(np.searchsorted(self.t, h["t"] + 1e-9) - 1, 0)
            s_phys = float((self.path_map[i] - a) @ u)
            pt = a + (s_phys + (turn_s - s_trk)) * u
            self.turn_pts.append(pt)
            if prev is not None and prev[0] == h["leg"] \
                    and float(np.linalg.norm(pt - prev[1])) > 0.20:
                self.flips.append({"t": h["t"], "old": prev[1], "new": pt,
                                   "leg": h["leg"]})
            prev = (h["leg"], pt)

        # believed-vs-true ALONG-ROUTE error, per hypothesis sample (the
        # lateral belief is not recorded per tick, so the estimated-pose
        # marker shows the along-route component only, riding the true
        # track).  Interpolated between samples for display; evaluation-side
        # visualization only -- the belief never sees truth.
        et, ev = [], []
        for h, pt in zip(self.hyp, self.turn_pts):
            if not h.get("masses"):
                continue
            leg = scene.legs[min(h["leg"], len(scene.legs) - 1)]
            a, u = np.asarray(leg["origin"], float), np.asarray(leg["u"], float)
            s_trk = float(h["hyp_s"][int(np.argmax(h["masses"]))])
            i = max(np.searchsorted(self.t, h["t"] + 1e-9) - 1, 0)
            s_phys = float((self.path_map[i] - a) @ u)
            et.append(h["t"])
            ev.append(s_trk - s_phys)
        self.err_t = np.asarray(et)
        self.err_v = np.asarray(ev)

        # a press CONTRADICTS the wrong hypothesis if the minor mass dropped
        # >= 0.05 across it, or it sits within 2 s of a planned-turn flip
        for p in self.presses:
            j0 = max(np.searchsorted(self.hyp_t, p["t"] - 1e-9) - 1, 0)
            j1 = min(np.searchsorted(self.hyp_t, p["t"] + 1e-9),
                     len(self.hyp) - 1)
            minor0 = 1.0 - max(self.hyp[j0]["masses"])
            minor1 = 1.0 - max(self.hyp[j1]["masses"])
            near_flip = any(abs(p["t"] - f["t"]) < 2.0 for f in self.flips)
            p["contradicts"] = p["valid"] and (minor0 - minor1 >= 0.05
                                               or near_flip)

    def at(self, t: float) -> dict:
        i = int(np.clip(np.searchsorted(self.t, t + 1e-9) - 1, 0, len(self.t) - 1))
        j = int(np.clip(np.searchsorted(self.hyp_t, t + 1e-9) - 1, 0,
                        len(self.hyp) - 1))
        k = int(np.clip(np.searchsorted(self.dir_t, t + 1e-9) - 1, 0,
                        max(len(self.directives) - 1, 0))) if self.directives else -1
        d = self.directives[k] if k >= 0 else None
        mode = self.mode[i]
        if mode == "PROBE" and d is not None:
            if str(d.get("reason", "")).startswith("touch_step_maintenance"):
                mode = "TOUCH"
            elif d.get("station") in self.disambig:
                mode = "LANDMARK_PROBE"
        return {"i": i, "j": j, "mode": mode, "hyp": self.hyp[j],
                "turn_pt": self.turn_pts[j], "directive": d}


# ── panels ──────────────────────────────────────────────────────────────────────


def _left_panel(scene, tel, w, h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon as MplPoly

    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    # video panels are PORTRAIT (600x720) while the scene is landscape: give
    # the map the upper block and use the leftover band for the legend and
    # observation counters; the standalone still overrides this rect
    ax = fig.add_axes([0.03, 0.30, 0.94, 0.62] if h > w
                      else [0.02, 0.02, 0.96, 0.90])
    for body in scene.plan.bodies:
        ax.add_patch(MplPoly(np.asarray(body.verts), closed=True,
                             facecolor="#c9c9c9", edgecolor="#636363", lw=0.9))
    for e in scene.plan.edges:
        a, b = np.asarray(e.a), np.asarray(e.b)
        ax.plot([a[0], b[0]], [a[1], b[1]], "--", color="#9ecae1", lw=1.8,
                zorder=2, label="_")
    g = np.asarray(scene.goal_map_xy)
    ax.add_patch(Circle(g, scene.goal_radius_m, facecolor="none",
                        edgecolor="#31a354", lw=2.0, zorder=3))
    xs = [v[0] for b in scene.plan.bodies for v in np.asarray(b.verts)]
    ys = [v[1] for b in scene.plan.bodies for v in np.asarray(b.verts)]
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.9)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.7)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if w >= h:                       # the still keeps its title; the video
        ax.set_title("overhead: planned route / raw + smoothed path",
                     fontsize=9, pad=4)   # panel's band legend replaces it
    art = {
        "raw": ax.plot([], [], "-", color="#b0b0b0", lw=0.7, alpha=0.8,
                       zorder=4)[0],
        "smooth": ax.plot([], [], "-", color="#1a1a1a", lw=2.2, zorder=5)[0],
        "robot": ax.plot([], [], "o", color="#1a1a1a", ms=8, zorder=8)[0],
        "est": ax.plot([], [], "s", mfc="none", mec="#756bb1", ms=10, mew=2.0,
                       zorder=8)[0],
        "est_line": ax.plot([], [], "--", color="#756bb1", lw=1.4, alpha=0.9,
                            zorder=7)[0],
        "flash": [ax.add_patch(Circle((0, 0), 0.01, facecolor="none",
                                      edgecolor="#e34a33", lw=2.5, alpha=0.0,
                                      zorder=9)) for _ in range(3)],
        "modes": [ax.plot([], [], "o", mfc="#9e9ac8", mec="#54278f", mew=1.2,
                          ms=10, alpha=0.7, zorder=6)[0] for _ in range(4)],
        "flip_old": ax.plot([], [], "X", color="#d62728", ms=13, mec="k",
                            mew=0.6, zorder=10)[0],
        "flip_new": ax.plot([], [], "D", mfc="none", mec="#31a354", ms=12,
                            mew=2.2, zorder=10)[0],
        "flip_seg": ax.plot([], [], "-", color="#ffbf00", lw=6, alpha=0.7,
                            zorder=6)[0],
        "flip_txt": ax.text(0.5, 1.03, "", transform=ax.transAxes, fontsize=10,
                            ha="center", color="#d62728", fontweight="bold"),
        "count": fig.text(0.5, 0.205, "", fontsize=11, ha="center",
                          color="#333333") if h > w else None,
        "ax": ax, "touch_art": [],
    }
    if h > w:
        fig.text(0.5, 0.155, "green dot = contact    blue ring = swept free    "
                 "grey x = invalid attempt", fontsize=8.5, ha="center",
                 color="#555555")
        fig.text(0.5, 0.115, "red X > amber > green diamond = planned turn "
                 "moved by touch evidence", fontsize=8.5, ha="center",
                 color="#555555")
        fig.text(0.5, 0.075, "grey trail = raw path    black = smoothed "
                 "(visualization only)", fontsize=8.5, ha="center",
                 color="#555555")
        fig.text(0.5, 0.035, "purple = PF modes (area ~ mass, square = "
                 "leading)    black dot = TRUE (evaluation only)",
                 fontsize=8.0, ha="center", color="#756bb1")
    else:
        fig.text(0.03, 0.03, "grey = raw   black = smoothed (visualization "
                 "only)", fontsize=7, color="#666666")
    return fig, art


def _draw_left(fig, art, tel, t):
    ax = art["ax"]
    st = tel.at(t)
    i = st["i"]
    art["raw"].set_data(tel.path_map[:i + 1:2, 0], tel.path_map[:i + 1:2, 1])
    m = tel.smooth_t <= t
    art["smooth"].set_data(tel.smooth[m, 0], tel.smooth[m, 1])
    pos = tel.path_map[i]
    art["robot"].set_data([pos[0]], [pos[1]])
    if getattr(tel, "modes", None) is not None and len(tel.err_t):
        # ACTUAL belief geometry: the leading mode where the filter puts it,
        # and every live mode at its own map xy.  The previous version placed
        # the "believed pose" at TRUTH plus a scalar along-route error, so the
        # marker could not disagree with truth in any direction that error did
        # not have -- and it inherited that error's frame bug.
        j = int(np.clip(np.searchsorted(tel.err_t, t + 1e-9) - 1, 0,
                        len(tel.err_t) - 1))
        bel = tel.bel_map[j]
        art["est"].set_data([bel[0]], [bel[1]])
        art["est_line"].set_data([pos[0], bel[0]], [pos[1], bel[1]])
        ms = tel.modes[j]
        for k, hm in enumerate(art["modes"]):
            if k < len(ms):
                hm.set_data([ms[k]["xy"][0]], [ms[k]["xy"][1]])
                hm.set_markersize(6.0 + 14.0 * float(ms[k]["mass"]))
                hm.set_alpha(0.30 + 0.55 * float(ms[k]["mass"]))
            else:
                hm.set_data([], [])
    elif len(tel.err_t) and art.get("count") is not None:
        err = float(np.interp(t, tel.err_t, tel.err_v))
        hcur = tel.at(t)["hyp"]
        leg = tel.scene.legs[min(hcur["leg"], len(tel.scene.legs) - 1)]
        u = np.asarray(leg["u"], float)
        est = pos + err * u
        art["est"].set_data([est[0]], [est[1]])
        art["est_line"].set_data([pos[0], est[0]], [pos[1], est[1]])

    # persistent touch markers (numbered valid, grey x invalid)
    done = [p for p in tel.presses if p["t"] <= t]
    while len(art["touch_art"]) < len(done):
        p = done[len(art["touch_art"])]
        if p["valid"]:
            fc = "#31a354" if p["kind"] == "contact" else "none"
            ec = "#1a6e33" if p["kind"] == "contact" else "#2b8cbe"
            h1 = ax.plot([p["xy"][0]], [p["xy"][1]], "o", mfc=fc, mec=ec,
                         ms=9, mew=1.6, zorder=7)[0]
            jit = ((p["num"] % 3) - 1) * 11.0
            off = -16.0 * np.asarray(p["facing"]) + jit * np.asarray(p["travel"])
            h2 = ax.annotate(str(p["num"]), p["xy"],
                             textcoords="offset points",
                             xytext=(float(off[0]), float(off[1])),
                             fontsize=8, fontweight="bold", ha="center",
                             color="#1a6e33" if p["kind"] == "contact"
                             else "#2b8cbe", zorder=11)
        else:
            h1 = ax.plot([p["xy"][0]], [p["xy"][1]], "x", color="#909090",
                         ms=7, mew=1.6, zorder=7)[0]
            h2 = None
        art["touch_art"].append((h1, h2))

    # expanding flash rings on recent touches
    recent = [p for p in done if t - p["t"] < FLASH_S][-3:]
    for ring, p in zip(art["flash"], recent + [None] * 3):
        if p is None:
            ring.set_alpha(0.0)
            continue
        f = (t - p["t"]) / FLASH_S
        ring.center = tuple(p["xy"])
        ring.set_radius(0.06 + 0.30 * f)
        ring.set_alpha(max(0.0, 0.9 * (1 - f)))
        ring.set_edgecolor("#e34a33" if p["valid"] else "#909090")

    # planned-turn flip: old + new + changed segment, held FLIP_SHOW_S
    live = [f for f in tel.flips if 0.0 <= t - f["t"] < FLIP_SHOW_S]
    if live:
        f = live[-1]
        art["flip_old"].set_data([f["old"][0]], [f["old"][1]])
        art["flip_new"].set_data([f["new"][0]], [f["new"][1]])
        art["flip_seg"].set_data([f["old"][0], f["new"][0]],
                                 [f["old"][1], f["new"][1]])
        art["flip_txt"].set_text("planned turn moved "
                                 f"{np.linalg.norm(f['new'] - f['old']):.2f} m")
    if art.get("count") is not None:
        nv = sum(1 for p in tel.presses if p["valid"] and p["t"] <= t)
        ni = sum(1 for p in tel.presses if not p["valid"] and p["t"] <= t)
        art["count"].set_text(f"valid observations {nv}/{tel.n_valid}     "
                              f"invalid attempts {ni}/{tel.n_invalid}")
    if not live:
        # keep the CURRENT planned turn as a quiet marker
        art["flip_old"].set_data([], [])
        art["flip_seg"].set_data([], [])
        art["flip_txt"].set_text("")
        pt = tel.at(t)["turn_pt"]
        if pt is not None:
            art["flip_new"].set_data([pt[0]], [pt[1]])
        else:
            art["flip_new"].set_data([], [])

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    ww, hh = fig.canvas.get_width_height()
    return buf.reshape(hh, ww, 4)[:, :, :3].copy()


def _right_panel(w, h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    art = {
        "mode_box": ax.add_patch(__import__("matplotlib").patches.FancyBboxPatch(
            (0.08, 0.84), 0.84, 0.11, boxstyle="round,pad=0.012",
            facecolor="#2b8cbe", edgecolor="none")),
        "mode": ax.text(0.5, 0.895, "WALK", fontsize=26, fontweight="bold",
                        ha="center", va="center", color="white"),
        "bel_lab": ax.text(0.08, 0.76, "BELIEF", fontsize=10, color="#777777",
                           fontweight="bold"),
        "bar_lead": ax.add_patch(__import__("matplotlib").patches.Rectangle(
            (0.30, 0.685), 0.5, 0.045, facecolor="#1a1a1a")),
        "bar_minor": ax.add_patch(__import__("matplotlib").patches.Rectangle(
            (0.30, 0.625), 0.1, 0.045, facecolor="#e34a33")),
        "lead_txt": ax.text(0.08, 0.707, "", fontsize=11, ha="left",
                            va="center"),
        "minor_txt": ax.text(0.08, 0.647, "", fontsize=11, ha="left",
                             va="center", color="#e34a33"),
        "sigma": ax.text(0.08, 0.555, "", fontsize=13),
        "err": ax.text(0.08, 0.505, "", fontsize=12.5, color="#756bb1"),
        "err2": ax.text(0.08, 0.466, "", fontsize=9.5, color="#999999"),
        "turn_lab": ax.text(0.08, 0.408, "PLANNED TURN", fontsize=10,
                            color="#777777", fontweight="bold"),
        "turn": ax.text(0.08, 0.335, "", fontsize=13),
        "contra": ax.text(0.5, 0.26, "", fontsize=13.5, ha="center",
                          color="#d62728", fontweight="bold", wrap=True),
        "ax": ax,
    }
    ax.text(0.5, 0.03, "contact-rich demo - offline replay of recorded states",
            fontsize=7, ha="center", color="#999999")
    return fig, art


def _draw_right(fig, art, tel, t):
    st = tel.at(t)
    mode = st["mode"]
    art["mode"].set_text(mode)
    art["mode_box"].set_facecolor(MODE_COLOR.get(mode, "#636363"))
    h = st["hyp"]
    masses = sorted(h["masses"], reverse=True)
    lead = masses[0]
    minor = 1.0 - lead
    art["bar_lead"].set_width(max(0.62 * lead, 0.01))
    art["bar_minor"].set_width(max(0.62 * minor, 0.005))
    second = masses[1] if len(masses) > 1 else 0.0
    art["lead_txt"].set_text(f"mode 1  {lead:.2f}")
    art["minor_txt"].set_text(f"mode 2  {second:.2f}"
                              + (f"   (+{len(masses) - 2} more)"
                                 if len(masses) > 2 else ""))
    art["sigma"].set_text(f"spread σ = {h['s_sigma']:.2f} m")
    if getattr(tel, "err_xy", None) is not None and len(tel.err_t):
        # POSE ERROR, not route progress.  The old line showed an along-route
        # scalar built from a double-mapped truth sample and read up to 12 m on
        # an episode whose belief stayed within 0.5 m -- see
        # `scripts/audit_belief_viz.py`.
        j = int(np.clip(np.searchsorted(tel.err_t, t + 1e-9) - 1, 0,
                        len(tel.err_t) - 1))
        art["err"].set_text("pose error |belief \u2212 TRUE| = "
                            f"{float(tel.err_xy[j]):.2f} m")
        art["err2"].set_text(f"map-x component {float(tel.err_v[j]):+.2f} m"
                             "   (evaluation only)")
    elif len(tel.err_t):
        e = float(np.interp(t, tel.err_t, tel.err_v))
        art["err"].set_text(f"believed vs TRUE (along-route): {e:+.2f} m")
    pt = st["turn_pt"]
    if pt is not None:
        flipped = tel.flips and t >= tel.flips[0]["t"]
        note = "TRUE corner" if flipped else "0.55 m EARLY (wrong mode)"
        art["turn"].set_text(f"leg {h['leg']}: turn at "
                             f"({pt[0]:.2f}, {pt[1]:.2f})\n{note}")
    else:
        art["turn"].set_text("final leg: ARRIVE at goal")
    contra = any(p["contradicts"] and 0 <= t - p["t"] < 2.5 for p in tel.presses)
    art["contra"].set_text("TOUCH CONTRADICTS\nWRONG HYPOTHESIS" if contra else "")
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    ww, hh = fig.canvas.get_width_height()
    return buf.reshape(hh, ww, 4)[:, :, :3].copy()


def _still(tel, scene, out_png: Path):
    """The overhead route view: raw + smoothed + planned + numbered touches."""
    xs = [v[0] for b in scene.plan.bodies for v in np.asarray(b.verts)]
    ys = [v[1] for b in scene.plan.bodies for v in np.asarray(b.verts)]
    w = (max(xs) - min(xs) + 1.4)
    h = (max(ys) - min(ys) + 1.6)
    fig, art = _left_panel(scene, tel, 980, int(980 * h / w) + 70)
    _draw_left(fig, art, tel, float(tel.t[-1]) + 1.0)
    art["ax"].set_title(
        "planned route (dashed) / raw realized path (grey) / smoothed "
        "visualization trace (black)\n10 Hz resample + 2.5 s centered moving "
        f"average / {tel.n_valid} valid touches numbered, "
        f"{tel.n_invalid} invalid attempts (grey x)",
        fontsize=8, pad=5)
    fig.savefig(out_png, dpi=115, bbox_inches="tight")
    print(f"still -> {out_png}", flush=True)


def render(episode_dir: Path, scene_name: str, out: Path, fps: float = 25.0,
           stride: int = 1, still_only: bool = False) -> None:
    from .scenes import build_scene
    ep = json.loads((episode_dir / "episode.json").read_text())
    scene = build_scene(scene_name)
    tel = Telemetry2(ep, scene)
    _still(tel, scene, episode_dir / "overhead_routes.png")
    if still_only:
        return

    from .backends import BackendConfig
    from .mujoco_backend import MujocoBackend, MujocoConfig
    rec = np.load(episode_dir / "states.npz")
    ts, qs = rec["t"], rec["qpos"]
    be = MujocoBackend(scene, BackendConfig(), np.random.default_rng(0),
                       MujocoConfig(seed=11 + int(ep.get("seed", 0)),
                                    video_width=720, video_height=720,
                                    camera_distance=1.9,
                                    camera_elevation=-16.0),
                       render=True, video_path=None)
    lfig, lart = _left_panel(scene, tel, 600, 720)
    rfig, rart = _right_panel(600, 720)
    cam = be.env._offline_renderer._cam if hasattr(be.env, "_offline_renderer") \
        else None
    az = None
    enc = None
    try:
        import torch
        for n in range(0, len(ts), stride):
            t = float(ts[n])
            q = torch.from_numpy(qs[n]).to(be.device)
            be.env.sim.data.qpos[0] = q
            be.env.sim.forward()
            if cam is not None:
                # look over the robot's shoulder toward the wall it faces:
                # azimuth = world yaw of facing, slewed ~1 s to avoid snaps
                i = tel.at(t)["i"]
                target = math.degrees(tel.yaw[i])
                if az is None:
                    az = target
                err = (target - az + 180.0) % 360.0 - 180.0
                az += np.clip(err, -4.0, 4.0)
                cam.azimuth = az
            frame = be.env.render()
            rgb = np.ascontiguousarray(np.asarray(frame, np.uint8)[:, :, :3])
            left = _draw_left(lfig, lart, tel, t)
            right = _draw_right(rfig, rart, tel, t)
            hh = min(left.shape[0], rgb.shape[0], right.shape[0])
            comp = np.hstack([left[:hh], rgb[:hh], right[:hh]])
            comp = comp[:comp.shape[0] // 2 * 2, :comp.shape[1] // 2 * 2]
            if enc is None:
                H, W = comp.shape[:2]
                enc = subprocess.Popen(
                    ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
                     "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                     "-r", str(fps / stride), "-i", "-", "-an",
                     "-c:v", "libx264", "-preset", "medium", "-crf", "21",
                     "-pix_fmt", "yuv420p", str(out)],
                    stdin=subprocess.PIPE)
            enc.stdin.write(comp.tobytes())
    finally:
        if enc is not None:
            enc.stdin.close()
            enc.wait()
        be.close()
    print(f"v2 replayed {len(ts) // stride} frames -> {out}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--still-only", action="store_true")
    args = ap.parse_args(argv)
    render(args.episode, args.scene, args.out, fps=args.fps,
           stride=args.stride, still_only=args.still_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
