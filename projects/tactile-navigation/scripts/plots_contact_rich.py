"""Deliverable figures for the contact-rich demo: labeled floorplan,
belief/uncertainty timeline, and the contact map.  Reads episode.json only."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon as MplPoly

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from execution.scenes import build_scene

d = sys.argv[1]
ep = json.load(open(f"{d}/episode.json"))
scene = build_scene(ep.get("scene", "contact_rich_gap"))
tr = ep["tick_trace"]
hyp = ep["hypothesis_trace"]

# world->map: fit rigid transform from truth_map_xy samples if present, else
# assume identity offset from the first tick (map frame == world frame here
# only up to the start pose; the recorded truth_map_xy is authoritative)
txy = np.asarray([r["xy"] for r in tr])
tmap = np.asarray(ep.get("truth_map_xy", []))
if len(tmap) >= 2 and len(tmap) == len(ep.get("truth_modes", [])) and False:
    pass
# tick_trace xy is world; truth_map_xy (sampled per _sample_truth) aligns 1:1
# with hypothesis_trace timestamps
ht = np.asarray([h["t"] for h in hyp])
if len(tmap) == len(hyp) and len(tmap) >= 3:
    tw = np.asarray([txy[np.searchsorted(np.asarray([r["t"] for r in tr]),
                                         t + 1e-9) - 1] for t in ht])
    A = np.c_[tw, np.ones(len(tw))]
    sol_x, *_ = np.linalg.lstsq(A, tmap[:, 0], rcond=None)
    sol_y, *_ = np.linalg.lstsq(A, tmap[:, 1], rcond=None)
    def w2m(p):
        v = np.r_[np.asarray(p, float), 1.0]
        return np.array([v @ sol_x, v @ sol_y])
else:
    def w2m(p):
        return np.asarray(p, float)

path = np.asarray([w2m(p) for p in txy[::4]])

def floorplan(ax, label_walls=True):
    for body in scene.plan.bodies:
        ax.add_patch(MplPoly(np.asarray(body.verts), closed=True,
                             facecolor="#c6c6c6", edgecolor="#636363", lw=0.9))
        if label_walls:
            c = np.asarray(body.verts).mean(axis=0)
            ax.annotate(body.name, c, fontsize=6.5, ha="center", va="center",
                        color="#404040")
    for e in scene.plan.edges:
        a, b = np.asarray(e.a), np.asarray(e.b)
        ax.plot([a[0], b[0]], [a[1], b[1]], "--", color="#9ecae1", lw=1.6)
    g = np.asarray(scene.goal_map_xy)
    ax.add_patch(Circle(g, scene.goal_radius_m, facecolor="none",
                        edgecolor="#31a354", lw=1.8))
    ax.annotate("GOAL", g + [0.06, 0.06], fontsize=8, color="#31a354",
                fontweight="bold")
    ax.set_aspect("equal")

# ── figure 1: the labeled floorplan ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=130)
floorplan(ax)
term0 = float(scene.legs[0]["terminus"])
ax.annotate("start", (0, 0), textcoords="offset points", xytext=(-4, -14),
            fontsize=8, fontweight="bold")
ax.plot([0], [0], "s", color="#252525", ms=7)
ax.annotate("0.40 m GAP\n(the landmark)", (0.90, 0.44), fontsize=7.5,
            color="#c51b8a", ha="center", fontweight="bold")
ax.annotate("", xy=(0.70, 0.40), xytext=(0.55, 0.62),
            arrowprops=dict(arrowstyle="->", color="#c51b8a", lw=1.2))
ax.annotate("", xy=(1.10, 0.40), xytext=(1.25, 0.62),
            arrowprops=dict(arrowstyle="->", color="#c51b8a", lw=1.2))
wrong = term0 - 0.55 + scene.standoff_m
ax.plot([wrong], [0.0], "X", color="#d62728", ms=10, mec="k", mew=0.5)
ax.annotate("wrong branch:\nturn here under the\nwrong registration\n(wall still alongside)",
            (wrong, -0.06), fontsize=7, color="#d62728", ha="center", va="top")
turn1 = term0 + scene.standoff_m
ax.plot([turn1], [0.0], "D", mfc="none", mec="#31a354", ms=9, mew=1.6)
ax.annotate("turn 1 (true)", (turn1 + 0.05, -0.14), fontsize=7.5, color="#31a354")
o2 = np.asarray(scene.legs[2]["origin"])
ax.annotate("turn 2", o2 + [0.06, 0.10], fontsize=7.5, color="#31a354")
ax.annotate("target area", np.asarray(scene.goal_map_xy) + [-0.55, 0.28],
            fontsize=8, color="#31a354")
for nm in ("GAPW", "GAPE"):
    st = scene.stations[nm]
    ax.plot([st.xy[0]], [st.xy[1]], "*", color="#c51b8a", ms=11, mec="k", mew=0.4)
ax.set_title("contact_rich_gap: floorplan, route, landmark, branch geometry",
             fontsize=9)
ax.set_xlim(-2.9, 4.6); ax.set_ylim(-1.1, 3.6)
fig.tight_layout(); fig.savefig(f"{d}/floorplan_labeled.png"); plt.close(fig)

# ── figure 2: belief timeline (masses, sigma, force, modes) ─────────────────────
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9.5, 7.2), dpi=130,
                                 sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1.4, 1.4]})
tm = [h["t"] for h in hyp]
lead, minor = [], []
for h in hyp:
    m = sorted(h["masses"], reverse=True)
    lead.append(m[0]); minor.append(1.0 - m[0])
a1.plot(tm, lead, "-", color="#252525", lw=1.6, label="leading-mode mass")
a1.plot(tm, minor, "-", color="#e34a33", lw=1.2, label="minor mass (wrong-mode risk)")
a1.axhline(0.10, color="#e34a33", ls=":", lw=1.0)
for p in ep["probes"]:
    if p["station"] in ("GAPW", "GAPE") and p["kind"] in ("contact", "swept_free"):
        a1.axvline(p["t"], color="#c51b8a", lw=0.7, alpha=0.5)
a1.legend(fontsize=7, loc="center right"); a1.set_ylabel("mass", fontsize=8)
a1.set_title("hypothesis masses -- magenta lines are landmark (gap) presses",
             fontsize=9)
a2.plot(tm, [h["s_sigma"] for h in hyp], "-", color="#756bb1", lw=1.4)
a2.axhline(0.12, color="#756bb1", ls="--", lw=1.0)
a2.set_ylabel("mixture sigma (m)", fontsize=8)
a2.set_title("along-route MIXTURE spread (a surviving ghost inflates it; the commit gates use the tracked cluster) vs the 0.12 m gate", fontsize=9)
tt = [r["t"] for r in tr]
ff = [max(r.get("force_l", 0), r.get("force_r", 0)) for r in tr]
a3.plot(tt, ff, "-", color="#252525", lw=0.7)
a3.axhline(25, color="crimson", ls="--", lw=1.0)
a3.axhline(12, color="darkorange", ls=":", lw=1.0)
a3.set_ylabel("hand force (N)", fontsize=8); a3.set_xlabel("sim time (s)", fontsize=8)
peak = ep.get("peak_force_n", 0.0)
a3.set_title(f"all-hand force -- episode peak {peak:.1f} N (12 N guard / 25 N limit)",
             fontsize=9)
for a in (a1, a2, a3):
    a.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(f"{d}/belief_timeline.png"); plt.close(fig)

# ── figure 3: contact map (trajectory + every valid touch) ──────────────────────
fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=130)
floorplan(ax, label_walls=False)
ax.plot(path[:, 0], path[:, 1], "-", color="#252525", lw=1.5, label="trajectory")
mk = {"contact": ("o", "#31a354", "contact"),
      "swept_free": ("o", "none", "swept free"),
      "invalid": ("x", "#969696", "invalid")}
for kind, (m, c, lab) in mk.items():
    pts = []
    for p in ep["probes"]:
        if p["kind"] != kind:
            continue
        st = scene.stations.get(p["station"])
        if st is not None:
            pts.append(st.xy + float(p["offset_m"]) * st.travel)
    if pts:
        pts = np.asarray(pts)
        if c == "none":
            ax.scatter(pts[:, 0], pts[:, 1], s=34, facecolors="none",
                       edgecolors="#2b8cbe", linewidths=1.3, label=lab, zorder=6)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], s=30 if m == "o" else 16, c=c,
                       marker=m, edgecolors="k" if m == "o" else None,
                       linewidths=0.4, label=lab, zorder=6)
ax.legend(fontsize=7.5, loc="upper left")
nv = len([p for p in ep["probes"] if p["kind"] in ("contact", "swept_free")])
ax.set_title(f"realized trajectory and all {nv} valid touches "
             f"({ep.get('n_probes')} presses total)", fontsize=9)
ax.set_xlim(-2.9, 4.6); ax.set_ylim(-1.1, 3.6)
fig.tight_layout(); fig.savefig(f"{d}/contact_map.png"); plt.close(fig)
print(f"figures -> {d}/floorplan_labeled.png, belief_timeline.png, contact_map.png")
