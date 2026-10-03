"""Plot recorded ground-truth and belief trajectories for multiple seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title", required=True)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, path in enumerate(args.runs):
        doc = json.loads(path.read_text())
        nav = doc["nav_trace"]
        truth = np.asarray([row["truth_xy"] for row in nav], float)
        belief = np.asarray([row["bel_xy"] for row in nav], float)
        color = colors[i % len(colors)]
        seed = doc["seed"]
        ax.plot(truth[:, 0], truth[:, 1], color=color, lw=2.2,
                label=f"seed {seed} ground truth ({doc['outcome']})")
        ax.plot(belief[:, 0], belief[:, 1], color=color, lw=1.5, ls="--",
                label=f"seed {seed} belief")
        ax.scatter(*truth[0], color=color, marker="o", s=28, zorder=4)
        ax.scatter(*truth[-1], color=color, marker="x", s=45, zorder=4)

    ax.set_title(args.title)
    ax.set_xlabel("map x [m]")
    ax.set_ylabel("map y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, loc="best")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
