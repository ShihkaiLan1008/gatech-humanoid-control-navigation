"""Synchronized belief/trajectory figures for one Batch-3A episode.

Everything except the map shares the episode clock, so a probe, the hypothesis
mass it moved, the branch gate it opened and the physical turn that followed can
be read off one x axis.  That is the only way to SEE the claim "realized touch
changed the route plan" rather than take it on trust.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from global_nav.plots import GRID, INK, INK_2, SERIES, SURFACE, WALL, WALL_EDGE, _style

MODE_COLOR = {"LOCOMOTION": SERIES[0], "STABILIZE": SERIES[2], "PROBE": SERIES[3],
              "TURN": SERIES[1], "start": INK_2, "END": INK_2}


def _draw_scene(ax, scene, delta: float = 0.0) -> None:
    from matplotlib.patches import Polygon as MPoly
    for k, w in enumerate(scene.walls):
        poly = w.polygon(f"w{k}")
        verts = poly.verts + np.array([delta, 0.0])
        ax.add_patch(MPoly(verts, closed=True, facecolor=WALL, edgecolor=WALL_EDGE,
                           linewidth=0.8, zorder=1))
    for k, leg in enumerate(scene.legs):
        a = np.asarray(leg["origin"], float)
        b = a + (float(leg["terminus"]) + scene.standoff_m) * np.asarray(leg["u"], float)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK_2, linewidth=1.0,
                linestyle="--", alpha=0.5, zorder=2)
    for st in scene.stations.values():
        ax.plot(*st.xy, marker="s", markersize=3.0,
                color=SERIES[4] if st.purpose == "disambiguator" else GRID,
                markeredgecolor=SURFACE, markeredgewidth=0.5, zorder=3)
    ax.set_aspect("equal")


def plot_episode(out: Path, result: dict, scene) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = result["hypothesis_trace"]
    t = np.array([r["t"] for r in tr])
    s_hat = np.array([r["s_hat"] for r in tr])
    s_sig = np.array([r["s_sigma"] for r in tr])
    legs = np.array([r["leg"] for r in tr])
    top = np.array([max(r["masses"]) if r["masses"] else 1.0 for r in tr])
    minor = 1.0 - top

    probes = result.get("probes", [])
    p_t = np.array([p["t"] for p in probes]) if probes else np.zeros(0)
    turn_t = [r["t"] for r in tr if r["mode"] == "TURN"]

    fig = plt.figure(figsize=(11.5, 8.4), facecolor=SURFACE)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.15, 1.0], hspace=0.55, wspace=0.22)

    # ── the map ─────────────────────────────────────────────────────────────
    axm = fig.add_subplot(gs[:, 0])
    _draw_scene(axm, scene)
    truth = result.get("truth_map_xy")
    if truth:
        p = np.asarray(truth, float).reshape(-1, 2)
        ax_modes = [r["mode"] for r in tr][:len(p)]
        axm.plot(p[:, 0], p[:, 1], color=INK, linewidth=1.4, alpha=0.35, zorder=5)
        for mode, color in MODE_COLOR.items():
            sel = [i for i, m in enumerate(ax_modes) if m == mode and i < len(p)]
            if sel:
                axm.plot(p[sel, 0], p[sel, 1], linestyle="", marker="o", markersize=4.0,
                         color=color, label=mode, zorder=6)
        axm.plot(*p[0], marker="o", markersize=8, color=INK, zorder=8)
    gx, gy = scene.goal_map_xy
    axm.add_patch(plt.Circle((gx, gy), scene.goal_radius_m, facecolor="none",
                             edgecolor=SERIES[2], linewidth=1.8, zorder=7))
    _style(axm, "map x — along the first wall (m)", "map y — toward the wall (m)")
    axm.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, -0.09))
    axm.set_title(f"{result['scene']} / {result['policy']} / seed {result['seed']}\n"
                  f"{result['outcome']}, {result['n_route_decisions']}"
                  f"/{result['route_decisions_required']} route decisions, "
                  f"{result['n_probes']} probes, {result['peak_force_n']:.1f} N peak",
                  fontsize=9.5, color=INK, loc="left")

    def mark(ax):
        for x in p_t:
            ax.axvline(x, color=SERIES[3], linewidth=0.7, alpha=0.55, zorder=0)
        for x in turn_t:
            ax.axvline(x, color=SERIES[1], linewidth=1.2, alpha=0.8, zorder=0)

    # ── along-leg belief ────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 1])
    _style(ax0, "", "s along leg (m)")
    ax0.fill_between(t, s_hat - s_sig, s_hat + s_sig, color=SERIES[0], alpha=0.18,
                     linewidth=0)
    ax0.plot(t, s_hat, color=SERIES[0], linewidth=1.8)
    for k, leg in enumerate(scene.legs[:-1]):
        sel = legs == k
        if sel.any():
            ax0.plot(t[sel], np.full(sel.sum(), float(leg["terminus"]) + scene.standoff_m),
                     color=INK_2, linewidth=1.0, linestyle="--")
    mark(ax0)
    ax0.set_title("believed position along the current leg (dashed: the mapped corner)",
                  fontsize=8.5, color=INK, loc="left")

    # ── the branch gate ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 1])
    _style(ax1, "", "minor mass")
    ax1.plot(t, minor, color=SERIES[4], linewidth=1.8)
    ax1.axhline(0.10, color=INK_2, linewidth=1.0, linestyle="--")
    ax1.annotate("branch gate δ = 0.10", (t[-1] if len(t) else 0, 0.10),
                 xytext=(-4, 4), textcoords="offset points", ha="right",
                 fontsize=7.5, color=INK_2)
    ax1.set_ylim(-0.04, max(0.6, float(minor.max()) * 1.2 if len(minor) else 0.6))
    mark(ax1)
    ax1.set_title("rival-hypothesis mass — the gate the turn waits on",
                  fontsize=8.5, color=INK, loc="left")

    # ── sigma vs the commit tolerance ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[2, 1])
    _style(ax2, "", "σ_s (m)")
    ax2.plot(t, s_sig, color=SERIES[0], linewidth=1.8)
    ax2.axhline(0.12, color=INK_2, linewidth=1.0, linestyle="--")
    ax2.annotate("commit tolerance", (t[-1] if len(t) else 0, 0.12), xytext=(-4, 4),
                 textcoords="offset points", ha="right", fontsize=7.5, color=INK_2)
    mark(ax2)
    ax2.set_title("along-leg spread — touch is what brings it under the tolerance",
                  fontsize=8.5, color=INK, loc="left")

    # ── forces ──────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3, 1])
    _style(ax3, "episode time (s)", "N")
    ticks = result.get("tick_trace") or []
    if ticks:
        tt = np.array([r["t"] for r in ticks])
        ff = np.array([r["probe_force_n"] for r in ticks])
        ax3.plot(tt, ff, color=SERIES[3], linewidth=1.0)
    elif probes:
        ax3.plot(p_t, [p["peak_force_n"] for p in probes], linestyle="", marker="o",
                 markersize=5, color=SERIES[3])
    ax3.axhline(25.0, color=SERIES[1], linewidth=1.2, linestyle="--")
    ax3.annotate("safety limit 25 N", (t[-1] if len(t) else 0, 25.0), xytext=(-4, -10),
                 textcoords="offset points", ha="right", fontsize=7.5, color=SERIES[1])
    ax3.set_ylim(0, 28)
    mark(ax3)
    ax3.set_title(f"probe force — peak {result['peak_force_n']:.2f} N",
                  fontsize=8.5, color=INK, loc="left")

    fig.suptitle("orange lines: probes   |   red lines: physical turns",
                 fontsize=8.5, color=INK_2, x=0.60, y=0.985, ha="center")
    path = out / f"episode_{result['scene']}_{result['policy']}_s{result['seed']}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path
