"""Top-down figures: the geometry validation (Phase 2) and the route (Phase 4).

ON "OVERLAY THE SKETCH".  A geometric overlay would be a lie.  The drawing is
schematic, and its own pixels say by how much: the 59.69 cm chord is drawn
113 px wide, the 675.64 cm run beside it 173 px -- 0.53 vs 3.91 cm/px, a factor
of 7.4 along one wall.  No affine (or even similarity-per-region) transform can
put the two pictures on top of each other without inventing distortion.  What
the comparison figure does instead is put the sketch and the reconstruction
side by side and stamp the SAME feature IDs on both, at the pixel positions
those features actually occupy in the drawing -- so the correspondence can be
checked feature by feature, which is the thing an overlay was wanted for.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .geometry import (CHOSEN, DemoMap, build_demo_map, clearance, remeasure)
from .route import Route, build_reference_route

WALL_FC = "#8a7f70"
CURVE_FC = "#9c6b8c"
BOX_FC = "#b08654"
DIM_C = "#1f6fd0"
START_C = "#8fc98f"
GOAL_C = "#e79a92"
ROUTE_C = "#d94436"

# Where each reconstructed feature sits in measured_map.jpg, in image pixels.
# Read off the black line work directly (see docs/demo_maps/RECONSTRUCTION.md);
# used ONLY to label the sketch panel, never to derive a coordinate.
SKETCH_PX = {
    "C1": (192, 640), "C2": (492, 660),
    "W1": (110, 558), "W2": (335, 590), "W3": (660, 590),
    "W4": (400, 752), "W5": (753, 460), "W6": (898, 470),
    "W7": (600, 322), "W8": (930, 322), "W9": (1105, 322),
    "W11": (800, 20), "W10": (461, 180), "W12": (1134, 180),
    "DOOR": (1026, 300), "START": (90, 598), "GOAL": (1020, 350),
}


def _wall_patches(ax, m: DemoMap, ids=True, fs=8):
    from matplotlib.patches import Polygon

    for w in m.walls:
        a, b = (np.asarray(p, float) for p in w.centreline)
        u = b - a
        u = u / max(float(np.linalg.norm(u)), 1e-12)
        n = np.array([-u[1], u[0]]) * 0.5 * w.thickness
        ax.add_patch(Polygon([a - n, b - n, b + n, a + n], closed=True,
                             fc=WALL_FC, ec="#3d372f", lw=0.7, zorder=3))
        if ids:
            c = 0.5 * (a + b) - np.asarray(w.inward, float) * 0.34
            ax.text(c[0], c[1], w.wid, fontsize=fs, color="#3d372f",
                    ha="center", va="center", zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))


def _curve_patches(ax, m: DemoMap, ids=True, fs=8):
    from matplotlib.patches import Circle

    for c in m.curves:
        ax.add_patch(Circle(c.center, c.radius, fc=CURVE_FC, ec="#4a2f42",
                            lw=0.7, zorder=4))
        if ids:
            ax.text(c.center[0], c.center[1] - c.radius - 0.30, c.cid,
                    fontsize=fs, color="#4a2f42", ha="center", va="top",
                    zorder=6, fontweight="bold")


def _box_patches(ax, m: DemoMap, ids=False, fs=6):
    from matplotlib.patches import Circle, Polygon

    for b in getattr(m, "boxes", ()):
        ax.add_patch(Polygon(b.poly(), closed=True, fc=BOX_FC, ec="#5c452a",
                             lw=0.6, alpha=0.9, zorder=4))
        if ids:
            ax.text(b.center[0], b.center[1], b.bid, fontsize=fs, color="#3b2c1b",
                    ha="center", va="center", zorder=6)
    for p in getattr(m, "pillars", ()):
        ax.add_patch(Circle(p.center, p.radius, fc=BOX_FC, ec="#5c452a",
                            lw=0.6, alpha=0.9, zorder=4))
        if ids:
            ax.text(p.center[0], p.center[1], p.pid, fontsize=fs,
                    color="#3b2c1b", ha="center", va="center", zorder=6)
    for h in getattr(m, "hulls", ()):
        ax.add_patch(Polygon(h.poly(), closed=True, fc=BOX_FC, ec="#5c452a",
                             lw=0.6, alpha=0.9, zorder=4))
        if ids:
            ax.text(h.center[0], h.center[1], h.hid, fontsize=fs,
                    color="#3b2c1b", ha="center", va="center", zorder=6)


def _regions(ax, m: DemoMap, fs=8):
    from matplotlib.patches import Circle

    ax.add_patch(Circle(m.start.center, m.start.radius, fc=START_C, ec="#4a7a4a",
                        lw=1.0, alpha=0.85, zorder=5))
    ax.add_patch(Circle(m.goal.center, m.goal.radius, fc=GOAL_C, ec="#a85a52",
                        lw=1.0, alpha=0.85, zorder=5))
    ax.text(m.start.center[0], m.start.center[1] - m.start.radius - 0.22, "START",
            fontsize=fs, color="#2f5c2f", ha="center", va="top", fontweight="bold")
    ax.text(m.goal.center[0], m.goal.center[1] + m.goal.radius + 0.22, "GOAL",
            fontsize=fs, color="#8a3a33", ha="center", va="bottom", fontweight="bold")
    for o in m.openings:
        ax.plot([o.p0[0], o.p1[0]], [o.p0[1], o.p1[1]], color="#a85a52",
                lw=2.4, ls=(0, (2, 1.6)), zorder=6)


def _dim(ax, p0, p1, label, off=0.0, side="above", fs=7.5, color=DIM_C,
         rotate=True, text_off=0.0):
    """One dimension arrow with its label, offset perpendicular to p0->p1.

    ``rotate=False`` keeps the text upright and pushes it ``text_off`` clear of
    the arrow -- what short vertical dimensions need to stay readable."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    u = p1 - p0
    L = float(np.linalg.norm(u))
    u = u / max(L, 1e-12)
    perp = np.array([-u[1], u[0]])
    n = perp * (off if side == "above" else -off)
    a, b = p0 + n, p1 + n
    ax.annotate("", xy=b, xytext=a, zorder=8,
                arrowprops=dict(arrowstyle="<|-|>", color=color, lw=0.9,
                                shrinkA=0, shrinkB=0, mutation_scale=8))
    for q, r in ((p0, a), (p1, b)):
        ax.plot([q[0], r[0]], [q[1], r[1]], color=color, lw=0.5, alpha=0.6, zorder=8)
    mid = 0.5 * (a + b) + perp * text_off
    if rotate:
        rot = math.degrees(math.atan2(u[1], u[0]))
        rot = rot + 180 if rot > 90 or rot < -90 else rot
    else:
        rot = 0.0
    ax.text(mid[0], mid[1], label, fontsize=fs, color=color, ha="center",
            va="center", rotation=rot, rotation_mode="anchor", zorder=9,
            bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.6))


def _axes_glyph(ax, at=(-1.9, -2.6), L=0.9):
    ax.annotate("", xy=(at[0] + L, at[1]), xytext=at,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.1))
    ax.annotate("", xy=(at[0], at[1] + L), xytext=at,
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.1))
    ax.text(at[0] + L + 0.12, at[1], "+x", fontsize=8, va="center")
    ax.text(at[0], at[1] + L + 0.12, "+y", fontsize=8, ha="center")
    ax.plot(*at, "ko", ms=3)
    ax.text(at[0] - 0.12, at[1] - 0.12, "map frame", fontsize=6.5, ha="right",
            va="top", color="#555")


def _frame(ax, m: DemoMap, pad=(2.0, 1.6)):
    ax.plot(0, 0, marker="o", ms=5, mfc="none", mec="k", mew=1.2, zorder=10)
    # the two maps have different datums; saying the wrong one is worse than
    # saying nothing, so the label follows the map
    origin = ("origin (0, 0)\nC1 west chord end,\non the long wall face"
              if m.curves else
              "origin (0, 0)\ncorner where the two longest\nscanned walls meet")
    ax.annotate(origin, xy=(0, 0), xytext=(0.35, 1.35), fontsize=6.8, color="k",
                ha="left", va="bottom", zorder=10,
                arrowprops=dict(arrowstyle="-", color="k", lw=0.7))
    ax.set_aspect("equal")
    ax.set_xlabel("x  [m]", fontsize=8)
    ax.set_ylabel("y  [m]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.3, color="#cccccc", alpha=0.7)
    ax.set_axisbelow(True)
    # bounds from the primitives, so this works for the tape map and the
    # scan-derived map without either having to declare named corners
    pts = np.array([p for w in m.walls for p in (w.p0, w.p1)], float)
    ax.set_xlim(pts[:, 0].min() - pad[0], pts[:, 0].max() + pad[0])
    ax.set_ylim(pts[:, 1].min() - pad[1], pts[:, 1].max() + pad[1])


def _dimension_set(ax, m: DemoMap):
    kp = m.key_points
    y0 = 0.0
    # the long wall's four-segment chain, drawn clear of the wall
    for x0, x1, lab in (
        (kp["x_c1_w"], kp["x_c1_e"], "59.69"),
        (kp["x_c1_e"], kp["x_c2_w"], "675.64"),
        (kp["x_c2_w"], kp["x_c2_e"], "73.66"),
        (kp["x_c2_e"], kp["x_corner"], "226.06"),
    ):
        _dim(ax, (x0, y0), (x1, y0), lab, off=0.95, side="above")
    # curved feature depths, upright and clear of the circle
    for c, lab in ((m.curve("C1"), "29.21"), (m.curve("C2"), "46.99")):
        x = c.center[0]
        _dim(ax, (x, 0.0), (x, c.center[1] - c.radius), lab, off=0.0,
             rotate=False, text_off=-0.62)
    _dim(ax, (kp["x_corner"], kp["y_bottom"] - 0.85),
         (kp["x_corr_e"], kp["y_bottom"] - 0.85), "241.3", off=0.0)
    _dim(ax, (kp["x_corner"] - 0.75, 0.0), (kp["x_corner"] - 0.75, kp["y_bottom"]),
         "162.56", off=0.0)
    _dim(ax, (kp["x_upper_e"] + 1.35, kp["y_bottom"]),
         (kp["x_upper_e"] + 1.35, kp["y_upper_s"]), "1054.1", off=0.0)
    _dim(ax, (kp["x_corr_e"], kp["y_upper_s"]), (kp["x_door_w"], kp["y_upper_s"]),
         "146.3", off=-0.95, side="above")
    _dim(ax, (kp["x_door_w"], kp["y_upper_s"]), (kp["x_door_e"], kp["y_upper_s"]),
         "162.56", off=1.05, side="above")
    ax.text(kp["x_west"] - 2.3, kp["y_upper_n"] + 0.35,
            "blue = the drawing's labelled dimensions, in cm,\n"
            "measured back OFF the reconstruction",
            fontsize=7, color=DIM_C, va="top")


def _sketch_panel(ax, sketch: Path, m: DemoMap):
    from matplotlib.offsetbox import AnchoredText
    import matplotlib.image as mpimg

    ax.imshow(mpimg.imread(str(sketch)))
    ax.set_xticks([]); ax.set_yticks([])
    for key, (px, py) in SKETCH_PX.items():
        col = {"C1": "#4a2f42", "C2": "#4a2f42", "DOOR": "#a85a52",
               "START": "#2f5c2f", "GOAL": "#8a3a33"}.get(key, "#3d372f")
        ax.text(px, py, key, fontsize=7, color=col, ha="center", va="center",
                fontweight="bold",
                bbox=dict(fc="white", ec=col, lw=0.5, alpha=0.85, pad=0.9))
    ax.add_artist(AnchoredText(
        "measured_map.jpg with the reconstruction's IDs stamped on.\n"
        "NOT to scale: 59.69 cm is drawn 113 px, 675.64 cm is drawn 173 px\n"
        "(0.53 vs 3.91 cm/px) -- which is why no overlay transform exists.",
        loc="lower right", prop=dict(size=7), frameon=True, pad=0.3))


def geometry_figure(m: DemoMap | None = None, sketch: Path | None = None,
                    out: Path | None = None):
    """Phase 2: reconstruction + dimensions + the sketch, side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = m or build_demo_map()
    have = sketch is not None and Path(sketch).exists()
    fig = plt.figure(figsize=(13.5, 13.0 if have else 8.2))
    gs = fig.add_gridspec(2 if have else 1, 1,
                          height_ratios=[1.0, 1.75] if have else [1.0],
                          hspace=0.10)
    if have:
        _sketch_panel(fig.add_subplot(gs[0]), Path(sketch), m)
    ax = fig.add_subplot(gs[-1])
    _wall_patches(ax, m)
    _curve_patches(ax, m)
    _box_patches(ax, m)
    _regions(ax, m)
    _dimension_set(ax, m)
    _axes_glyph(ax)
    _frame(ax, m)
    ax.set_title("Reconstructed demo room -- metric, from the labelled "
                 "dimensions only", fontsize=11, pad=8)
    err = max(abs(r["labelled"] - r["recon"]) for r in remeasure(m))
    fig.suptitle("Phase 2 -- geometry validation      "
                 f"all 12 labelled dimensions recovered to {err:.1e} m",
                 fontsize=12, y=0.985)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=145, bbox_inches="tight")
        plt.close(fig)
    return out


def route_figure(m: DemoMap | None = None, r: Route | None = None,
                 out: Path | None = None):
    """Phase 4: the same map with the sparse reference route on it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    m = m or build_demo_map()
    r = r or build_reference_route(m)
    fig, ax = plt.subplots(figsize=(13.5, 9.4))
    _wall_patches(ax, m, fs=7)
    _curve_patches(ax, m, fs=7)
    _box_patches(ax, m)
    _regions(ax, m)

    for s in r.segs:
        p = s.sample(0.02)
        ax.plot(p[:, 0], p[:, 1], color=ROUTE_C, lw=3.0,
                ls="--" if s.kind in ("approach", "transit", "goal_approach") else "-",
                solid_capstyle="round", zorder=7, alpha=0.95)
        # direction arrows along the segment
        n = max(1, int(s.length // 1.1))
        for t in np.linspace(0.18, 0.86, n):
            i = int(t * (len(p) - 1))
            j = min(i + 2, len(p) - 1)
            if j > i:
                ax.annotate("", xy=p[j], xytext=p[i], zorder=8,
                            arrowprops=dict(arrowstyle="-|>", color=ROUTE_C,
                                            lw=0.1, mutation_scale=15))
        mid = p[len(p) // 2]
        nrm = np.array([-(p[-1] - p[0])[1], (p[-1] - p[0])[0]])
        nrm = nrm / max(float(np.linalg.norm(nrm)), 1e-9)
        ax.text(*(mid + 0.42 * nrm), s.sid, fontsize=8, color=ROUTE_C,
                ha="center", va="center", fontweight="bold", zorder=9,
                bbox=dict(fc="white", ec=ROUTE_C, lw=0.6, alpha=0.9, pad=1.0))

    for wid, p in r.waypoints():
        ax.plot(p[0], p[1], "o", ms=5, mfc="white", mec=ROUTE_C, mew=1.4, zorder=10)
        ax.text(p[0], p[1] - 0.34, wid, fontsize=7, color="#7a2318", ha="center",
                va="top", zorder=10)

    # only when the route actually switches hands -- a generic traverse across a
    # scanned room follows nothing, so there is no switch to point at
    sw = next((b for a, b in zip(r.segs, r.segs[1:])
               if a.hand and b.hand and a.hand != b.hand), None)
    if sw is None:
        sw = next((s for s in r.segs if s.kind == "transit"), None)
    if sw is not None and any(s.hand for s in r.segs):
        p = sw.sample(0.05)
        ax.annotate("hand switches\nLEFT -> RIGHT here", xy=p[len(p) // 2],
                    xytext=(sw.p0[0] - 1.5, sw.p0[1] - 2.6), fontsize=7.5,
                    color="#7a2318", ha="center",
                    arrowprops=dict(arrowstyle="-|>", color="#7a2318", lw=0.8))
    _axes_glyph(ax)
    _frame(ax, m)
    followed = [s for s in r.segs
                if s.kind in ("wall_follow", "curve_follow", "corner_turn")]
    ax.legend(handles=([
        Line2D([], [], color=ROUTE_C, lw=3,
               label="followed at a constant 0.56 m (wall / curve / corner)")]
        if followed else []) + [
        Line2D([], [], color=ROUTE_C, lw=3, ls="--",
               label=("approach, free transit, goal approach" if followed
                      else "free traverse (nothing followed by hand)")),
        Line2D([], [], color="none", label=f"{len(r.segs)} segments, {r.length:.1f} m"),
        Line2D([], [], color="none",
               label=f"min clearance {clearance(r.polyline(0.02), m).min():.2f} m"),
    ] + ([Line2D([], [], color=BOX_FC, lw=6,
                 label=f"{len(m.boxes)} scanned clutter boxes")] if m.boxes else [
    ]) + [
    ], loc="upper left", fontsize=7.5, framealpha=0.92)
    if m.curves:
        end = "doorway" if m.openings else "end room"
        title = ("Phase 4 -- sparse reference route: START -> long wall -> C1 "
                 f"-> C2 -> corridor transit -> east wall -> corner -> {end}")
    else:
        title = ("Reference path across the scanned room: shortest clear "
                 "traverse, START -> GOAL")
    ax.set_title(title, fontsize=11, pad=8)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=145, bbox_inches="tight")
        plt.close(fig)
    return out
