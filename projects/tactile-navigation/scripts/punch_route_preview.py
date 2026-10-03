#!/usr/bin/env python3
"""Plot task route geometry before simulation (not an EIG action rollout)."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from execution.punch_tasks import TASKS
from execution.punch_eig_policy import EigRoutePolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", choices=TASKS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--standoff", type=float, default=0.4)
    parser.add_argument("--full-route", action="store_true")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task = TASKS[args.room](standoff_m=args.standoff)
    if args.full_route:
        task.enable_full_route()
    builder = EigRoutePolicy.__new__(EigRoutePolicy)
    builder.standoff_m = args.standoff
    segments = builder._route(task.feature_sequence())
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    for body in task.scene.plan.bodies:
        v = np.asarray(body.verts)
        ax.fill(v[:, 0], v[:, 1], color="0.8", ec="0.4", zorder=1)
    previous = np.zeros(2)
    points = [previous.copy()]
    for i, segment in enumerate(segments):
        start_s, _ = segment.coords(previous)
        ss = np.linspace(start_s, segment.s_end, 100)
        xy = np.array([segment.lane(s, args.standoff) for s in ss])
        color = plt.get_cmap("tab10")(i)
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2.5,
                label=f"{i + 1}: {segment.label}")
        ax.annotate("", xy=xy[60], xytext=xy[40],
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
        if i:
            ax.plot([previous[0], xy[0, 0]], [previous[1], xy[0, 1]],
                    color="0.4", ls=":", lw=1)
        previous = xy[-1]
        points.extend(xy)
        ax.text(*xy[65], f" {i + 1}", fontsize=12, color=color)
    ax.scatter(0, 0, marker="*", s=150, color="green", zorder=5, label="start")
    ax.scatter(*previous, marker="X", s=80, color="red", zorder=5, label="end")
    points = np.asarray(points)
    lo, hi = points.min(axis=0) - 0.8, points.max(axis=0) + 0.8
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel="map x [m]",
           ylabel="map y [m]", title=f"{args.room}: configured route / face order\n"
           "Reference offset only; EIG chooses actual steps online")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
