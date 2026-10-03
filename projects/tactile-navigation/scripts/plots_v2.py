"""Route/contact/belief figures for a V2 episode (episode.json only)."""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon as MplPoly

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lane_graph.scene2 import build_v2_scene

d, map_name = sys.argv[1], sys.argv[2]
ep = json.load(open(f"{d}/episode.json"))
scene = build_v2_scene(map_name)
tr = ep["tick_trace"]
base0 = np.asarray(tr[0]["xy"], float)
yaw0 = float(tr[0]["yaw"])
v = np.array([math.cos(yaw0), math.sin(yaw0)])
u = np.array([v[1], -v[0]])
w2m = lambda p: np.array([(np.asarray(p, float) - base0) @ u,
                          (np.asarray(p, float) - base0) @ v])
path = np.asarray([w2m(r["xy"]) for r in tr[::4]])

fig, ax = plt.subplots(figsize=(8.6, 6.8), dpi=130)
for b in scene.plan.bodies:
    ax.add_patch(MplPoly(np.asarray(b.verts), closed=True,
                         facecolor="#c9c9c9", edgecolor="#5a5a5a", lw=0.8))
g = np.asarray(scene.goal_map_xy)
ax.add_patch(Circle(g, scene.goal_radius_m, facecolor="none",
                    edgecolor="#31a354", lw=2.0))
ax.annotate("GOAL", g + [0.06, 0.26], color="#31a354", fontweight="bold",
            fontsize=9, ha="center")
ax.plot(path[:, 0], path[:, 1], "-", color="#9c9c9c", lw=0.7,
        label="raw realized path")
tu = np.arange(tr[0]["t"], tr[-1]["t"], 0.1)
tt = np.asarray([r["t"] for r in tr])
sm = np.c_[[np.convolve(np.pad(np.interp(tu, tt, np.asarray(
    [w2m(r["xy"])[c] for r in tr])), 12, mode="edge"),
    np.ones(25) / 25, "valid")[:len(tu)] for c in range(2)]].T
ax.plot(sm[:, 0], sm[:, 1], "-", color="#1a1a1a", lw=2.0,
        label="smoothed (viz only)")
ax.plot([0], [0], "s", color="#111111", ms=9, label="start")
for p in ep["probes"]:
    pm = w2m(p["truth_xy"])
    ok = p["kind"] in ("contact", "swept_free")
    ax.plot([pm[0]], [pm[1]], "o" if ok else "x",
            color="#31a354" if p["kind"] == "contact"
            else ("#2b8cbe" if ok else "#909090"),
            ms=9 if ok else 7, mec="k" if ok else None, mew=0.5 if ok else 1.4)
ax.set_aspect("equal")
nv = sum(1 for p in ep["probes"] if p["kind"] in ("contact", "swept_free"))
ax.set_title(f"{map_name}: first V2 MVP episode -- {ep['outcome']}, "
             f"goal {ep['goal_dist_m']:.2f} m, {nv} valid touches "
             f"({ep['n_probes']} presses), {ep['path_len_m']:.1f} m, "
             f"{ep['sim_time_s']:.0f} s", fontsize=9)
ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout()
fig.savefig(f"{d}/route_contacts.png", bbox_inches="tight")
plt.close(fig)

fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9.5, 7), dpi=130, sharex=True)
hs = [h for h in ep["hypothesis_trace"] if h.get("truth_xy")]
tm = [h["t"] for h in hs]
lead = [max(h["masses"] or [1.0]) for h in hs]
a1.plot(tm, lead, "-", color="#1a1a1a", lw=1.5, label="leading-mode mass")
a1.plot(tm, [1 - x for x in lead], "-", color="#e34a33", lw=1.1,
        label="minor mass")
for p in ep["probes"]:
    a1.axvline(p["t"], color="#c51b8a", lw=0.7, alpha=0.5)
a1.legend(fontsize=7); a1.set_ylabel("mass", fontsize=8)
a1.set_title("hypothesis masses (magenta = presses)", fontsize=9)
err = [float(np.linalg.norm(w2m(h["truth_xy"]) - np.asarray(h["bel_xy"])))
       for h in hs]
a2.plot(tm, err, "-", color="#756bb1", lw=1.4)
a2.set_ylabel("bel err (m)", fontsize=8)
a2.set_title("believed vs TRUE position error (map frame, viz only)",
             fontsize=9)
ttf = [r["t"] for r in tr]
ff = [max(r.get("force_l", 0), r.get("force_r", 0)) for r in tr]
a3.plot(ttf, ff, "-", color="#1a1a1a", lw=0.6)
a3.axhline(25, color="crimson", ls="--", lw=1.0)
a3.axhline(12, color="darkorange", ls=":", lw=1.0)
a3.set_ylabel("hand N", fontsize=8); a3.set_xlabel("sim time (s)", fontsize=8)
a3.set_title(f"all-hand force -- peak {ep['peak_force_n']:.1f} N", fontsize=9)
fig.tight_layout()
fig.savefig(f"{d}/belief_timeline.png", bbox_inches="tight")
plt.close(fig)
print(f"figures -> {d}/route_contacts.png, belief_timeline.png")
