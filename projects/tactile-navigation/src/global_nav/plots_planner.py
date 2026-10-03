"""Planner benchmark figures.  Reuses the Batch-1 validated palette and styling.

The figure that matters most is the trajectory panel: the same seed, walked by
the no-touch baseline and by the belief-aware planner, wrong turn versus pillar
detour, on the same map.  Everything else quantifies what that picture shows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .plots import INK, INK_2, SERIES, SURFACE, _draw_map, _style
from .stressmaps import build_scenario

POLICY_ORDER = ("shortest_astar", "fixed_interval", "max_wall", "max_landmark",
                "belief_aware", "oracle")
POLICY_LABEL = {
    "shortest_astar": "no-touch\nshortest", "fixed_interval": "fixed\ninterval",
    "max_wall": "max wall\nadjacent", "max_landmark": "max\nlandmark",
    "belief_aware": "belief\naware", "oracle": "oracle",
}


def _cell(results: dict, scn: str, pol: str) -> dict:
    return results["cells"][f"{scn}/{pol}"]


# ── figure 1: success / wrong-turn / excess bars ─────────────────────────────────


def plot_headline(out: Path, results: dict, scenario: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measures = (("success", "navigation success rate", None),
                ("wrong_turn", "wrong-turn rate", None),
                ("excess_m", "excess distance over oracle (m)", None),
                ("n_probes", "probes per episode", None))
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.6), facecolor=SURFACE)
    x = np.arange(len(POLICY_ORDER))
    for ax, (key, title, _) in zip(axes, measures):
        _style(ax)
        mu = [_cell(results, scenario, p)[key]["mean"] for p in POLICY_ORDER]
        sem = [_cell(results, scenario, p)[key]["sem"] for p in POLICY_ORDER]
        bars = ax.bar(x, mu, 0.62, color=[SERIES[4] if p == "belief_aware" else SERIES[0]
                                          for p in POLICY_ORDER],
                      edgecolor=SURFACE, linewidth=1.4, zorder=3)
        ax.errorbar(x, mu, yerr=sem, fmt="none", ecolor=INK_2, elinewidth=1.0,
                    capsize=2.5, zorder=4)
        for b, v, e in zip(bars, mu, sem):        # direct labels (relief rule)
            ax.annotate(f"{v:.2f}" if key in ("success", "wrong_turn") else f"{v:.1f}",
                        (b.get_x() + b.get_width() / 2, v + e), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7.5,
                        color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_LABEL[p] for p in POLICY_ORDER], fontsize=7,
                           color=INK)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        if key in ("success", "wrong_turn"):
            ax.set_ylim(0, 1.12)
    fig.suptitle(f"{scenario} — {results['cells'][f'{scenario}/oracle']['n']} seeds, "
                 f"{results['n_particles']} particles (error bars: 1 s.e.m.; "
                 "highlighted bar: the belief-aware planner)",
                 fontsize=10.5, color=INK, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = out / f"headline_{scenario}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ── figure 2: trajectories, wrong turn vs detour ─────────────────────────────────


def plot_trajectories(out: Path, results: dict, scenario: str,
                      policies=("shortest_astar", "belief_aware"), max_panels: int = 4) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = build_scenario(scenario)
    traces = results.get("traces", {})
    panels = []
    for pol in policies:
        eps = results["episodes"].get(f"{scenario}/{pol}", [])
        trs = traces.get(f"{scenario}/{pol}", [])
        # one wrong/failed episode and one successful one, where available
        for want_success in (False, True):
            for ep_meta, tr in zip(eps, trs):
                if ep_meta["success"] == want_success:
                    panels.append((pol, ep_meta, tr))
                    break
    panels = panels[:max_panels]

    x0, x1, y0, y1 = sc.plan.bounds()
    n = max(len(panels), 1)
    fig, axes = plt.subplots(n, 1, figsize=(8.6, 0.6 + (y1 - y0) / (x1 - x0) * 8.0 * n),
                             facecolor=SURFACE, squeeze=False)
    for ax, (pol, meta, tr) in zip(axes[:, 0], panels):
        _draw_map(ax, sc.plan, reg_true=np.asarray(sc.reg_true), show_route=False)
        xy = np.array([row["truth_xy"] for row in tr])
        ax.plot(xy[:, 0], xy[:, 1], color=SERIES[4] if pol == "belief_aware" else SERIES[0],
                linewidth=2.0, zorder=6)
        ax.plot(*xy[0], marker="o", markersize=6, color=INK,
                markeredgecolor=SURFACE, zorder=7)
        probes = [row["truth_xy"] for row in tr if row["event"].startswith("probe")]
        if probes:
            p = np.array(probes)
            ax.plot(p[:, 0], p[:, 1], linestyle="", marker="x", markersize=7,
                    markeredgewidth=2.0, color=SERIES[3], zorder=8)
        gx, gy = sc.goal_xy
        circ = plt.Circle((gx, gy), sc.goal_radius_m, facecolor="none",
                          edgecolor=SERIES[2], linewidth=1.6, zorder=5)
        ax.add_patch(circ)
        _style(ax, "x (m)", "y (m)")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title(
            f"{pol} — {meta['outcome']}"
            f"{', WRONG TURN' if meta['wrong_turn'] else ''}   |   "
            f"cost {meta['cost_m']:.1f} m (oracle {meta['oracle_ref_m']:.1f})   |   "
            f"{meta['n_probes']} probes (×)",
            fontsize=9, color=INK, loc="left")
    fig.suptitle(f"{scenario} — trajectories: what acting on the belief does",
                 fontsize=11, color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = out / f"trajectories_{scenario}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ── figure 3: wrong-turn risk over the run ───────────────────────────────────────


def plot_risk_trace(out: Path, results: dict, scenario: str = "twin_junctions") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces = results.get("traces", {})
    fig, ax = plt.subplots(figsize=(8.6, 3.8), facecolor=SURFACE)
    _style(ax, "cost spent (m)", "minor-hypothesis mass = risk if committing now")
    shown = {"belief_aware": 0, "shortest_astar": 0}
    for pol, color in (("shortest_astar", SERIES[0]), ("belief_aware", SERIES[4])):
        for tr in traces.get(f"{scenario}/{pol}", [])[:3]:
            cost = np.array([row["cost_m"] for row in tr])
            risk = np.array([1.0 - max(row["mode_masses"]) if row["mode_masses"] else 0.0
                             for row in tr])
            label = pol if shown[pol] == 0 else None
            ax.plot(cost, risk, color=color, linewidth=1.8,
                    alpha=0.9 if pol == "belief_aware" else 0.55, label=label, zorder=4)
            shown[pol] += 1
            if pol == "belief_aware":
                probes = [(row["cost_m"], 1.0 - max(row["mode_masses"]))
                          for row in tr if row["event"].startswith("probe")]
                if probes:
                    p = np.array(probes)
                    ax.plot(p[:, 0], p[:, 1], linestyle="", marker="x", markersize=7,
                            markeredgewidth=2.0, color=SERIES[3], zorder=6,
                            label="probe" if shown[pol] == 1 else None)
    ax.axhline(0.10, color=INK_2, linewidth=1.0, linestyle="--")
    ax.annotate("branch gate δ = 0.10", (ax.get_xlim()[1], 0.10), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=8, color=INK_2)
    ax.set_ylim(-0.04, 0.62)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper right")
    ax.set_title("the probe is where the risk dies — no-touch carries it into the junction",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    path = out / f"risk_{scenario}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ── figure 4: registration gate ──────────────────────────────────────────────────


def plot_reg_gate(out: Path, sweep: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = sweep["points"]
    true = np.array([p["true_offset_m"] for p in pts])
    fig, ax = plt.subplots(figsize=(6.8, 4.0), facecolor=SURFACE)
    _style(ax, "true registration offset ‖r‖ (m)", "final error of the estimate (m)")
    ax.plot(true, [p["do_nothing_err_m"] for p in pts], color=INK_2, linewidth=1.2,
            linestyle="--", zorder=3)
    ax.annotate("assume zero", (true[-1], pts[-1]["do_nothing_err_m"]),
                xytext=(-4, 5), textcoords="offset points", ha="right", fontsize=8,
                color=INK_2)
    for key, color, label in (("raw_err_m", SERIES[0], "raw estimate"),
                              ("gated_err_m", SERIES[4], "gated estimate")):
        mu = np.array([p[key] for p in pts])
        sem = np.array([p[key.replace("_err_m", "_sem")] for p in pts])
        ax.fill_between(true, mu - sem, mu + sem, color=color, alpha=0.15, linewidth=0)
        ax.plot(true, mu, color=color, linewidth=2.0, marker="o", markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.1, label=label, zorder=5)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper left")
    ax.set_title("the gate never does worse than assuming zero;\n"
                 "above the floor it passes the partial recovery 0.45 m touch can earn",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    path = out / "registration_gate.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ── figure 5: relocalization timeline ────────────────────────────────────────────


def plot_reloc_timeline(out: Path, results: dict, scenario: str = "gauntlet_slip") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces = results.get("traces", {}).get(f"{scenario}/belief_aware", [])
    eps = results["episodes"].get(f"{scenario}/belief_aware", [])
    pick = None
    for meta, tr in zip(eps, traces):
        if meta["n_reloc_recovered"] > 0 and meta["success"]:
            pick = tr
            break
    if pick is None:
        for meta, tr in zip(eps, traces):
            if meta["n_reloc_triggered"] > 0:
                pick = tr
                break
    if pick is None:
        return out / "reloc_timeline_SKIPPED.png"

    cost = np.array([row["cost_m"] for row in pick])
    err = np.array([row["nav_err"] for row in pick])
    state = [row["belief_state"] for row in pick]
    color_of = {"NORMAL": SERIES[2], "SUSPECT_LOST": SERIES[3], "RELOCALIZE": SERIES[4]}

    fig, ax = plt.subplots(figsize=(8.6, 3.6), facecolor=SURFACE)
    _style(ax, "cost spent (m)", "navigation-reference error (m)")
    for i in range(len(cost) - 1):
        ax.plot(cost[i:i + 2], err[i:i + 2], color=color_of[state[i]], linewidth=2.2,
                zorder=4)
    for name, color in color_of.items():
        ax.plot([], [], color=color, linewidth=2.2, label=name)
    slip_i = int(np.argmax(err > 0.6)) if np.any(err > 0.6) else 0
    ax.axvline(cost[slip_i], color=INK_2, linewidth=1.0, linestyle=":")
    ax.annotate("slip becomes visible", (cost[slip_i], ax.get_ylim()[1] * 0.95),
                xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK_2)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper left",
              ncol=3)
    ax.set_title("a 0.9 m un-odometered slip: rank-1 blindness delays detection, "
                 "RELOCALIZE re-seeds, recovery closes the error",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    path = out / "reloc_timeline.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def make_all_figures(out: Path, results: dict) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    scenarios = sorted({k.split("/")[0] for k in results["cells"]})
    for scn in scenarios:
        paths.append(plot_headline(out, results, scn))
        if "traces" in results:
            paths.append(plot_trajectories(out, results, scn))
    if "traces" in results and "twin_junctions" in scenarios:
        paths.append(plot_risk_trace(out, results))
    if "registration_gate" in results:
        paths.append(plot_reg_gate(out, results["registration_gate"]))
    if "traces" in results and "gauntlet_slip" in scenarios:
        paths.append(plot_reloc_timeline(out, results))
    return paths
