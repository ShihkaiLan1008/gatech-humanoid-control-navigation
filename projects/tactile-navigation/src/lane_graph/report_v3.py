"""Per-demo figure + metrics table for a V3 contact-rich episode."""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon as MplPoly

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lane_graph.scene3 import build_v3_scene


def metrics(ep: dict) -> dict:
    pr = ep["probes"]
    win = [p for p in pr if p.get("in_window")]
    forces = [p["peak_force_n"] for p in pr if p["kind"] == "contact"]
    hyp = ep["hypothesis_trace"]
    return {
        "outcome": ep["outcome"], "goal_dist_m": ep["goal_dist_m"],
        "pos_err_m": ep["global_pos_err_m"], "path_m": ep["path_len_m"],
        "sim_s": ep["sim_time_s"], "steps_tf": ep["steps_touch_follow"],
        "steps_touched": ep["steps_with_valid_touch"],
        "coverage": ep["touch_coverage"], "valid_obs": ep["n_valid_obs"],
        "in_window": len(win), "presses": ep["n_probes"],
        "peak_force_n": ep["peak_force_n"],
        "contact_force_range": (min(forces) if forces else 0.0,
                                max(forces) if forces else 0.0),
        "turns": ep["turns"], "wrong_turn": ep["wrong_turn"],
        "force_aborts": ep.get("force_aborts", 0),
        "belief_start": hyp[0]["masses"] if hyp else [],
        "belief_end": hyp[-1]["masses"] if hyp else [],
        "sigma_start": round(hyp[0]["s_sigma"], 3) if hyp else 0,
        "sigma_end": round(hyp[-1]["s_sigma"], 3) if hyp else 0,
    }


def figure(d: str, map_name: str) -> None:
    ep = json.load(open(f"{d}/episode.json"))
    scene = build_v3_scene(map_name)
    tr = ep["tick_trace"]
    base0 = np.asarray(tr[0]["xy"], float)
    yaw0 = float(tr[0]["yaw"])
    v = np.array([math.cos(yaw0), math.sin(yaw0)])
    u = np.array([v[1], -v[0]])
    w2m = lambda p: np.array([(np.asarray(p, float) - base0) @ u,
                              (np.asarray(p, float) - base0) @ v])
    m = metrics(ep)

    fig, (ax, a2) = plt.subplots(
        1, 2, figsize=(15.5, 6.6), dpi=125,
        gridspec_kw={"width_ratios": [1.35, 1.0]})
    for b in scene.plan.bodies:
        ax.add_patch(MplPoly(np.asarray(b.verts), closed=True,
                             facecolor="#c9c9c9", edgecolor="#5a5a5a", lw=0.7))
    g = np.asarray(scene.goal_map_xy)
    ax.add_patch(Circle(g, scene.goal_radius_m, facecolor="none",
                        edgecolor="#31a354", lw=2.0))
    ax.annotate("GOAL", g + [0, scene.goal_radius_m + 0.12], color="#31a354",
                fontweight="bold", fontsize=9, ha="center")
    path = np.asarray([w2m(r["xy"]) for r in tr[::4]])
    ax.plot(path[:, 0], path[:, 1], "-", color="#9c9c9c", lw=0.7,
            label="raw path")
    tt = np.asarray([r["t"] for r in tr])
    tu = np.arange(tt[0], tt[-1], 0.1)
    sm = np.c_[[np.convolve(np.pad(np.interp(
        tu, tt, np.asarray([w2m(r["xy"])[c] for r in tr])), 12, mode="edge"),
        np.ones(25) / 25, "valid")[:len(tu)] for c in range(2)]].T
    ax.plot(sm[:, 0], sm[:, 1], "-", color="#1a1a1a", lw=1.9,
            label="smoothed (viz)")
    ax.plot([0], [0], "s", color="#111", ms=9, label="start")
    for p in ep["probes"]:
        pm = w2m(p["truth_xy"])
        if p.get("in_window"):
            ax.plot([pm[0]], [pm[1]], "o", color="#e34a33", ms=9, mec="k",
                    mew=0.5, zorder=6)
        elif p["kind"] == "contact":
            ax.plot([pm[0]], [pm[1]], "o", color="#31a354", ms=7, mec="k",
                    mew=0.4, zorder=6)
        elif p["kind"] == "swept_free":
            ax.plot([pm[0]], [pm[1]], "o", mfc="none", mec="#2b8cbe", ms=7,
                    mew=1.2, zorder=6)
        else:
            ax.plot([pm[0]], [pm[1]], "x", color="#999", ms=6, zorder=6)
    ax.plot([], [], "o", color="#e34a33", ms=9, label="touch, 10-25 N")
    ax.plot([], [], "o", mfc="none", mec="#2b8cbe", ms=7, label="swept free")
    ax.set_aspect("equal")
    ax.legend(fontsize=7.5, loc="best")
    ax.set_title(f"{map_name}: {m['outcome']}, goal {m['goal_dist_m']:.3f} m | "
                 f"{m['steps_touched']}/{m['steps_tf']} increments touched "
                 f"({m['coverage']*100:.0f}%), {m['in_window']} in 10-25 N",
                 fontsize=9)

    hyp = [h for h in ep["hypothesis_trace"] if h.get("truth_xy")]
    th = [h["t"] for h in hyp]
    lead = [max(h["masses"] or [1.0]) for h in hyp]
    a2.plot(th, lead, "-", color="#1a1a1a", lw=1.5, label="leading mass")
    a2.plot(th, [1 - x for x in lead], "-", color="#e34a33", lw=1.1,
            label="minor mass")
    # hypothesis_trace truth_xy is ALREADY map-frame (the executive converts
    # it); only tick_trace xy and probe truth_xy are world-frame
    err = [float(np.linalg.norm(np.asarray(h["truth_xy"])
                                - np.asarray(h["bel_xy"]))) for h in hyp]
    a2.plot(th, err, "-", color="#756bb1", lw=1.4, label="belief error (m)")
    for p in ep["probes"]:
        if p.get("in_window"):
            a2.axvline(p["t"], color="#e34a33", lw=0.5, alpha=0.35)
    a2.set_xlabel("sim time (s)", fontsize=8)
    a2.legend(fontsize=7.5)
    a2.set_title(f"belief: masses {m['belief_start']} -> {m['belief_end']}, "
                 f"sigma {m['sigma_start']} -> {m['sigma_end']} "
                 f"(red = in-window touches)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{d}/route_contacts_belief.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(m, indent=1))
    print(f"figure -> {d}/route_contacts_belief.png")


if __name__ == "__main__":
    figure(sys.argv[1], sys.argv[2])
