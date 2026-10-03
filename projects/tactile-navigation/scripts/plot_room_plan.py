"""Top-down plot of a punch task's 2D map -- the `Floorplan` the particle
filter scores contacts against and the EIG raycasts, which is the same
geometry MuJoCo is built from.

    python scripts/plot_room_plan.py room830
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("room", default="room830", nargs="?")
    ap.add_argument("--out", type=Path, default=Path("out/punch/figures"))
    ap.add_argument("--run", type=Path, default=None,
                    help="overlay this run's true path and punch contacts")
    a = ap.parse_args(argv)

    from execution.punch_tasks import TASKS
    task = TASKS[a.room]()
    plan = task.scene.plan

    fig, ax = plt.subplots(figsize=(9.0, 8.0))
    for b in plan.bodies:
        v = np.asarray(b.verts, float)
        ax.add_patch(Polygon(v, closed=True, facecolor="#c9ccd2",
                             edgecolor="#7d848e", lw=1.2, zorder=1))
    if a.run is not None:
        import json
        doc = json.loads(a.run.read_text())
        tr = np.asarray([r["truth_xy"] for r in doc["nav_trace"]], float)
        if len(tr):
            ax.plot(tr[:, 0], tr[:, 1], color="#2f6fd0", lw=2.2, zorder=3,
                    label="true path")
        hit = np.asarray([t["truth_xy"] for t in doc["touches"]
                          if t["contact"]], float)
        if len(hit):
            ax.plot(hit[:, 0], hit[:, 1], "o", ms=5, color="#2bd67b",
                    mec="white", mew=0.7, zorder=4, label="punch landed")
        ax.legend(frameon=False, fontsize=12, loc="best")

    # the start pose the task hands the policy
    yaw = float(task.face_yaw_map)
    ax.plot([0.0], [0.0], marker="o", ms=11, color="#e8452c", zorder=5)
    ax.arrow(0.0, 0.0, 0.45 * math.cos(yaw), 0.45 * math.sin(yaw),
             width=0.03, color="#e8452c", zorder=5, length_includes_head=True)

    ax.set_aspect("equal")
    ax.set_xlabel("x  (m, map frame)", fontsize=14)
    ax.set_ylabel("y  (m, map frame)", fontsize=14)
    ax.set_title(f"{a.room}  -  2D map ({len(plan.bodies)} convex bodies)",
                 fontsize=15, pad=10)
    ax.grid(True, color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / f"map2d_{a.room}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(out)
    for i, b in enumerate(plan.bodies):
        v = np.asarray(b.verts, float)
        print(f"  [{i:2d}] {b.name:<10s} {b.signature:<8s} "
              f"n={len(v)} x[{v[:,0].min():6.2f},{v[:,0].max():6.2f}] "
              f"y[{v[:,1].min():6.2f},{v[:,1].max():6.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
