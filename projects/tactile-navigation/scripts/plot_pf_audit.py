"""Figures for the PF audit: touch -> particle update -> belief correction.

    PYTHONPATH=src python3 scripts/plot_pf_audit.py out/pf_audit/v3_r2_kin

Three panels, one clock:

1. belief error against the truth, with every tactile update marked, so a touch
   that corrected the belief is visually distinguishable from one that did not;
2. the decisive margin -- how many nats the TRUTH pose earned minus what the
   filter's own leading mode earned on the same evidence.  Bars near zero are
   updates that carried no information about where the robot was;
3. ESS and the distance from the truth to the nearest particle, which is what
   separates "the filter lost the truth" from "the filter never had it".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

INK, GRID = "#1a1a1a", "#d9d9d9"
GOOD, BAD, FLAT = "#2b8cbe", "#cb4b16", "#999999"


def plot(root: Path) -> Path:
    rec = json.loads((root / "pf_updates.json").read_text())
    if not rec:
        raise SystemExit(f"no updates recorded in {root}")
    t = np.array([r["t_s"] for r in rec])
    eb = np.array([r["pos_err_before_m"] for r in rec])
    ea = np.array([r["pos_err_after_m"] for r in rec])
    mg = np.array([r["ll_margin_truth_minus_leading"] for r in rec])
    ess = np.array([r["ess_after"] for r in rec])
    nn = np.array([r["truth_nn_dist_before_m"] for r in rec])
    n = np.array([r["n_particles"] for r in rec])

    fig, ax = plt.subplots(3, 1, figsize=(10, 8.4), sharex=True,
                           gridspec_kw={"height_ratios": [1.25, 1.0, 1.0]})
    for a in ax:
        a.grid(True, color=GRID, lw=0.6, alpha=0.7)
        a.set_axisbelow(True)

    # 1. did the touch correct the belief?
    for i in range(len(rec)):
        c = GOOD if ea[i] < eb[i] - 1e-9 else (BAD if ea[i] > eb[i] + 1e-9 else FLAT)
        ax[0].plot([t[i], t[i]], [eb[i], ea[i]], "-", color=c, lw=2.4, zorder=3)
        ax[0].plot(t[i], ea[i], "o", color=c, ms=5, zorder=4)
    ax[0].plot(t, eb, ":", color=INK, lw=1.0, alpha=0.6, zorder=2)
    ax[0].set_ylabel("belief error (m)")
    ax[0].set_title("touch → particle update → belief correction "
                    "(blue = corrected, orange = worsened, grey = no change)",
                    fontsize=10)

    # 2. was there information to correct WITH?
    col = [FLAT if abs(m) < 1.0 else (GOOD if m > 0 else BAD) for m in mg]
    ax[1].bar(t, mg, width=max(0.6, 0.012 * (t[-1] - t[0] + 1)), color=col)
    ax[1].axhline(0, color=INK, lw=0.8)
    ax[1].axhspan(-1.0, 1.0, color=FLAT, alpha=0.18, zorder=0)
    ax[1].set_ylabel("ll(truth) − ll(leading)\n(nats)")
    ax[1].set_title("shaded band = the evidence does not separate the truth from "
                    "the filter's own answer", fontsize=9)

    # 3. could the filter have represented the truth at all?
    ax[2].plot(t, ess / np.maximum(n, 1), "-o", color=INK, ms=4, label="ESS / N")
    a2 = ax[2].twinx()
    a2.plot(t, nn, "-s", color=BAD, ms=4, label="truth → nearest particle (m)")
    a2.set_ylabel("truth → nearest particle (m)", color=BAD)
    a2.tick_params(axis="y", colors=BAD)
    ax[2].set_ylabel("ESS / N")
    ax[2].set_xlabel("sim time (s)")
    ax[2].set_ylim(0, 1.05)
    ax[2].legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    out = root / "pf_audit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for d in sys.argv[1:]:
        print("wrote", plot(Path(d)))
