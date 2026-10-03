"""Benchmark figures.

The one figure that matters is the map: a bimodal particle cloud is precisely the
thing a mean and a covariance cannot show, and the whole claim of this module is
that such clouds are carried rather than collapsed.  Every time-series panel shares
the arclength axis with it, so a station, the belief step it produced, and the mode
count that followed can be read off one x.

Palette: the validated categorical default (blue / orange / aqua / yellow /
magenta).  Three of those five sit below 3:1 against the light surface, so the
relief rule applies and every one of them is directly labelled rather than being
identified by colour alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# validated categorical slots -- fixed order, never cycled
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
WALL = "#d8d7d2"
WALL_EDGE = "#b4b3ad"
TRUTH = "#0b0b0b"


def _style(ax, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=8.5)


def _weight_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    # sequential: ONE hue, light -> dark, monotone lightness
    return LinearSegmentedColormap.from_list("w", ["#cfe0f6", "#7aa8e0", "#2a78d6", "#173f72"])


def _draw_map(ax, plan, *, reg_true=None, show_route: bool = True) -> None:
    from matplotlib.patches import Polygon as MPoly
    for body in plan.bodies:
        ax.add_patch(MPoly(body.verts, closed=True, facecolor=WALL, edgecolor=WALL_EDGE,
                           linewidth=0.7, zorder=1))
    if show_route:
        for e in plan.edges:
            pts = np.vstack([e.a, e.b])
            on_route = e.index in plan.route
            ax.plot(pts[:, 0], pts[:, 1], color=INK_2 if on_route else SERIES[1],
                    linewidth=1.0, linestyle="--" if on_route else ":",
                    alpha=0.55, zorder=2)
    for f in plan.features:
        ax.plot(*f.position, marker="D", markersize=3.4, color=SERIES[3],
                markeredgecolor=SURFACE, markeredgewidth=0.7, zorder=6)
    if reg_true is not None and float(np.linalg.norm(reg_true)) > 1e-9:
        for body in plan.translated(reg_true).bodies:
            ax.add_patch(MPoly(body.verts, closed=True, facecolor="none",
                               edgecolor=SERIES[1], linewidth=0.7, linestyle="--",
                               alpha=0.75, zorder=3))
    ax.set_aspect("equal")


def _cov_ellipse(ax, mean, cov, n_sigma: float = 2.0, color: str = SERIES[0]):
    from matplotlib.patches import Ellipse
    vals, vecs = np.linalg.eigh(np.asarray(cov, float)[:2, :2])
    vals = np.maximum(vals, 1e-12)
    angle = float(np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1])))
    ax.add_patch(Ellipse(mean[:2], *(2 * n_sigma * np.sqrt(vals[::-1])), angle=angle,
                         facecolor="none", edgecolor=color, linewidth=1.6, zorder=8))


def _grid(n_panels: int, xspan: float, yspan: float, *, ncols: int = 2,
          panel_w: float = 5.4, extra_h: float = 1.0):
    """Panel geometry that respects the map's aspect ratio.

    A map axis with ``set_aspect('equal')`` in a panel of the wrong shape does not
    stretch -- it pads, and the figure comes out mostly blank margin.
    """
    ncols = min(ncols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    panel_h = panel_w * float(yspan) / max(float(xspan), 1e-9)
    return nrows, ncols, (panel_w * ncols, panel_h * nrows + extra_h)


def _draw_belief(ax, plan, row, *, cmap=None) -> None:
    cmap = cmap or _weight_cmap()
    p = row.get("particles")
    if p:
        xy, w = np.asarray(p["xy"], float), np.asarray(p["w"], float)
        order = np.argsort(w)
        ax.scatter(xy[order, 0], xy[order, 1], c=w[order], cmap=cmap, s=9.0,
                   linewidths=0, zorder=4, alpha=0.85,
                   vmin=0.0, vmax=max(float(w.max()), 1e-9))
    mean = np.asarray(row["pose_mean"], float)
    _cov_ellipse(ax, mean, np.asarray(row["pose_cov"], float))
    ax.plot(*mean[:2], marker="o", markersize=5.5, color=SERIES[0],
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=9)
    ax.plot(*row["truth_xy"], marker="*", markersize=11, color=TRUTH,
            markeredgecolor=SURFACE, markeredgewidth=0.9, zorder=10)


# ── figure 1: where the belief thinks it is ──────────────────────────────────────


def plot_map(out_dir: Path, result: dict, plan, n_panels: int = 4) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in result["trace"] if r.get("particles")]
    if not rows:
        rows = result["trace"]
    pick = np.linspace(0, len(rows) - 1, min(n_panels, len(rows))).astype(int)
    reg_true = np.asarray(result.get("reg_true", (0.0, 0.0)), float)
    x0, x1, y0, y1 = plan.bounds()
    nrows, ncols, figsize = _grid(len(pick), x1 - x0, y1 - y0, ncols=1, panel_w=9.0,
                                  extra_h=1.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE, squeeze=False)
    flat = axes.ravel()
    for ax, i in zip(flat, pick):
        row = rows[i]
        _draw_map(ax, plan, reg_true=reg_true)
        _draw_belief(ax, plan, row)
        _style(ax, "x (m)", "y (m)")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title(
            f"s = {row['s_m']:.2f} m   |   error {row['position_error_m']:.3f} m   "
            f"|   {row['n_modes']} mode(s)   |   ESS {row['ess']:.0f}   "
            f"|   P(wrong turn) {row['wrong_turn_probability']:.2f}",
            fontsize=9, color=INK, loc="left")
    for ax in flat[len(pick):]:
        ax.set_visible(False)
    handles = [
        flat[0].plot([], [], marker="*", linestyle="", color=TRUTH, label="true pose")[0],
        flat[0].plot([], [], marker="o", linestyle="", color=SERIES[0],
                     label="estimate (2σ ellipse)")[0],
        flat[0].plot([], [], marker="D", linestyle="", color=SERIES[3],
                     label="map feature")[0],
        flat[0].plot([], [], linestyle="--", color=SERIES[1],
                     label="true wall (map ⊕ registration offset)")[0],
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8, frameon=False, ncol=4,
               labelcolor=INK_2, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(f"{result['scenario']} — belief over the floorplan",
                 fontsize=11, color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = out_dir / f"map_{result['scenario']}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ── figure 2: every requested output, on one clock ───────────────────────────────


def plot_timeseries(out_dir: Path, result: dict, plan) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = result["trace"]
    s = np.array([r["s_m"] for r in tr])
    err = np.array([r["position_error_m"] for r in tr])
    std = np.array([np.linalg.norm(r["pos_std_m"]) for r in tr])
    reg = np.array([r.get("registration_error_m", np.nan) for r in tr])
    wrong = np.array([r["wrong_turn_probability"] for r in tr])
    ess = np.array([r["ess"] for r in tr])
    modes = np.array([r["n_modes"] for r in tr])
    edge_p = np.array([r["edge_probabilities"] for r in tr])
    stations = np.array([r["s_m"] for r in tr if r["station"] is not None])

    fig, axes = plt.subplots(5, 1, figsize=(9.0, 10.0), sharex=True, facecolor=SURFACE)
    for ax in axes:
        _style(ax)
        for x in stations:
            ax.axvline(x, color=GRID, linewidth=0.8, zorder=0)

    ax = axes[0]
    ax.fill_between(s, 0, std, color=SERIES[0], alpha=0.16, linewidth=0,
                    label="belief 1σ (position)")
    ax.plot(s, err, color=SERIES[0], linewidth=2.0, label="true position error")
    ax.set_ylabel("m", color=INK_2, fontsize=8.5)
    ax.set_title("position error vs what the belief claims to know   "
                 "(band above line = honest; line above band = overconfident)",
                 fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="center right")

    ax = axes[1]
    ax.plot(s, reg, color=SERIES[1], linewidth=2.0)
    ax.axhline(float(np.linalg.norm(result.get("reg_true", (0, 0)))), color=INK_2,
               linewidth=1.0, linestyle="--")
    ax.annotate("‖true offset‖ — the error you get by not estimating at all",
                xy=(s[-1], float(np.linalg.norm(result.get("reg_true", (0, 0))))),
                xytext=(-6, 5), textcoords="offset points", ha="right",
                fontsize=7.5, color=INK_2)
    ax.set_ylabel("m", color=INK_2, fontsize=8.5)
    ax.set_title("registration error", fontsize=9, color=INK, loc="left")

    ax = axes[2]
    ax.plot(s, wrong, color=SERIES[4], linewidth=2.0)
    # fixed floor of 0.3 on purpose: auto-scaling a genuinely-zero series magnifies
    # its noise into what looks like a signal
    ax.set_ylim(-0.03, max(0.3, float(wrong.max()) * 1.25))
    ax.set_ylabel("probability", color=INK_2, fontsize=8.5)
    ax.annotate(f"peak {wrong.max():.3f}", (s[int(np.argmax(wrong))], float(wrong.max())),
                xytext=(0, 8), textcoords="offset points", ha="center",
                fontsize=8, color=SERIES[4])
    ax.set_title("wrong-turn probability — mass whose edge history left the nominal route",
                 fontsize=9, color=INK, loc="left")

    ax = axes[3]
    bottom = np.zeros_like(s)
    for i, e in enumerate(plan.edges):
        top = bottom + edge_p[:, i]
        ax.fill_between(s, bottom, top, color=SERIES[i % len(SERIES)], linewidth=0.8,
                        edgecolor=SURFACE, alpha=0.9)          # 2px surface gap between fills
        mid = int(np.argmax(edge_p[:, i]))
        if edge_p[mid, i] > 0.25:                              # selective direct labels
            ax.annotate(e.name, (s[mid], bottom[mid] + edge_p[mid, i] / 2),
                        ha="center", va="center", fontsize=7.5, color=INK)
        bottom = top
    ax.set_ylim(0, 1)
    ax.set_ylabel("probability", color=INK_2, fontsize=8.5)
    ax.set_title("route-edge posterior", fontsize=9, color=INK, loc="left")

    ax = axes[4]
    ax.plot(s, ess, color=SERIES[2], linewidth=2.0, label="effective sample size")
    ax.set_ylabel("particles", color=INK_2, fontsize=8.5)
    ax.set_xlabel("arclength along the nominal route, s (m)", color=INK_2, fontsize=8.5)
    twin_marks = np.flatnonzero(modes > 1)
    if twin_marks.size:
        ax.plot(s[twin_marks], np.full(twin_marks.size, float(ess.max()) * 0.06),
                marker="^", linestyle="", markersize=6, color=SERIES[4],
                label="belief is multi-modal")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_2, loc="lower right")
    ax.set_title("particle health", fontsize=9, color=INK, loc="left")

    fig.suptitle(f"{result['scenario']} — {result['question']}", fontsize=10.5,
                 color=INK, x=0.012, ha="left", wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = out_dir / f"timeseries_{result['scenario']}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ── figure 3: the ablation ───────────────────────────────────────────────────────


def plot_ablation(out_dir: Path, ablations: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(ablations["table"].keys())
    measures = (("final_position_error_m", "final position error"),
                ("mean_position_error_m", "mean position error over the run"))
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.2), facecolor=SURFACE)
    _style(ax, ylabel="position error (m)")
    for j, (key, label) in enumerate(measures):
        mu = np.array([ablations["table"][n][key]["mean"] for n in names])
        # s.e.m., not s.d.: the question this chart is asked is whether the channels
        # DIFFER, and the spread across seeds is not the uncertainty on that
        sd = np.array([ablations["table"][n][key].get("sem", 0.0) for n in names])
        bars = ax.bar(x + (j - 0.5) * width, mu, width * 0.92, color=SERIES[j],
                      edgecolor=SURFACE, linewidth=1.5, label=label, zorder=3)
        ax.errorbar(x + (j - 0.5) * width, mu, yerr=sd, fmt="none", ecolor=INK_2,
                    elinewidth=1.0, capsize=3, zorder=4)
        for b, v, e in zip(bars, mu, sd):                       # direct labels (relief rule)
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v + e),
                        xytext=(0, 5), textcoords="offset points", ha="center",
                        fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8.5, color=INK)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2)
    ax.set_title(f"measurement-channel ablation — {len(ablations['seeds'])} seeds, "
                 f"{ablations['n_particles']} particles (error bars: 1 s.e.m.)\n"
                 "either tactile channel beats dead reckoning ~3x, and the map prior alone "
                 "buys nothing;\nthe tactile bars overlap unpaired — only the per-seed "
                 "PAIRED comparison separates them (see report)",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    path = out_dir / "ablation.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ── figure 4: is the registration offset observable at all? ──────────────────────


def plot_registration(out_dir: Path, sweep: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    true = np.array([p["true_offset_m"] for p in sweep["points"]])
    res = np.array([p["residual_error_m"] for p in sweep["points"]])
    sd = np.array([p["residual_std_m"] for p in sweep["points"]])
    fig, ax = plt.subplots(figsize=(6.8, 4.2), facecolor=SURFACE)
    _style(ax, "true registration offset ‖r‖ (m)", "residual error after the run (m)")
    ax.plot(true, true, color=INK_2, linewidth=1.2, linestyle="--", zorder=2)
    ax.annotate("do nothing (leave the offset at zero)", (true[-1], true[-1]),
                xytext=(-6, 4), textcoords="offset points", ha="right",
                fontsize=8, color=INK_2)
    ax.fill_between(true, res - sd, res + sd, color=SERIES[0], alpha=0.16, linewidth=0)
    ax.plot(true, res, color=SERIES[0], linewidth=2.0, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5)
    floor = float(res.min())
    ax.axhline(floor, color=SERIES[1], linewidth=1.2, linestyle=":")
    ax.annotate(f"observability floor ≈ {floor:.02f} m", (true[-1], floor),
                xytext=(-4, -12), textcoords="offset points", ha="right",
                fontsize=8.5, color=SERIES[1])
    ax.set_ylim(bottom=0.0)
    ax.set_title("the belief only recovers an offset bigger than its own floor\n"
                 f"({len(sweep['seeds'])} seeds, {sweep['n_particles']} particles)",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    path = out_dir / "registration_observability.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ── figure 5: the thing a covariance cannot show ─────────────────────────────────


def plot_ambiguity(out_dir: Path, result: dict, plan) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in result["trace"] if r.get("particles") and r["station"] is not None]
    multi = [r for r in rows if r["n_modes"] > 1]
    single = [r for r in rows if r["n_modes"] == 1]
    chosen = (multi[:3] + single[:1])[:4] if multi else rows[:4]
    chosen.sort(key=lambda r: r["s_m"])

    xlim, ylim = (1.2, 7.6), (-1.2, 4.2)
    nrows, ncols, figsize = _grid(len(chosen), xlim[1] - xlim[0], ylim[1] - ylim[0],
                                  ncols=2, panel_w=5.0, extra_h=1.1)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE, squeeze=False)
    flat = axes.ravel()
    for ax, row in zip(flat, chosen):
        _draw_map(ax, plan, reg_true=None)   # the offset is not what this figure is about
        _draw_belief(ax, plan, row)
        shown = [m for m in row["modes"] if m["weight"] >= 0.05]
        for k, m in enumerate(shown):
            # stagger: neighbouring modes can be closer together than their labels
            ax.annotate(f"{m['weight']:.0%}", m["xy"], xytext=(0, 26 + 20 * (k % 2)),
                        textcoords="offset points", ha="center", fontsize=10, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.8),
                        bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE,
                                  edgecolor=GRID, linewidth=0.9))
        _style(ax, "x (m)", "y (m)")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"s = {row['s_m']:.2f} m   |   {row['n_modes']} mode(s)   "
                     f"|   mean {row['position_error_m']:.2f} m from truth",
                     fontsize=9, color=INK, loc="left")
    for ax in flat[len(chosen):]:
        ax.set_visible(False)
    fig.suptitle(f"{result['scenario']} — two identical corners, and what resolves them\n"
                 "the 2σ ellipse spans both modes: where the belief is bimodal, "
                 "the mean is a place no particle believes in",
                 fontsize=10, color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.2)
    path = out_dir / f"ambiguity_{result['scenario']}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
