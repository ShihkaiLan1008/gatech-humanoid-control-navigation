"""A/B/C comparison of wall-touch behaviours on one segment, seed and speed.

    python3 scripts/wall_touch_compare.py out/hw/abc --out out/hw/abc/compare.png

A  old TOUCH_FOLLOW cadence   (`wall_touch_old_cadence.py`)
B  flat-arm wall-parallel tap (`wall_parallel_demo.py --arm-style flat`)
C  contact-rich wall-parallel (`wall_parallel_demo.py`)

Base pose comes from each run's recorded `states_*.npz`, so the trajectory panel
is the same measurement for all three regardless of how the arm was driven.
Force comes from the per-tick record.  Hand-to-wall distance exists per tick only
for B and C; A is a press primitive and reports one window per press, which is
drawn as a shaded band instead.  No simulator runs here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from execution.wall_lane import WallSegment                      # noqa: E402

ARMS = (("A", "_A", "old TOUCH_FOLLOW cadence"),
        ("B", "_B", "flat-arm wall-parallel tap"),
        ("C", "_C", "contact-rich wall-parallel"))


def _map_frame(q):
    """The backend's map frame, reconstructed from the first recorded state."""
    qw, qx, qy, qz = q[0, 3], q[0, 4], q[0, 5], q[0, 6]
    yaw0 = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    v = np.array([math.cos(yaw0), math.sin(yaw0)])
    return q[0, :2].copy(), np.array([v[1], -v[0]]), v, yaw0


def load(d: Path, tag: str):
    js = sorted(d.glob(f"wp_*{tag}.json"))
    st = sorted(d.glob(f"states_*{tag}.npz"))
    if not js or not st:
        return None
    ep = json.loads(js[0].read_text())
    rec = np.load(st[0])
    q, t = rec["qpos"], rec["t"]
    base0, u, v, yaw0 = _map_frame(q)
    xy = np.c_[(q[:, :2] - base0) @ u, (q[:, :2] - base0) @ v]
    qw, qx, qy, qz = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    yaw_w = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    yaw_map = np.unwrap(yaw_w - yaw0 + math.pi / 2)
    sd = ep["runs"][0]["segment"] if "runs" in ep else ep["segment"]
    seg = WallSegment(p0=np.asarray(sd["p0"], float),
                      tangent=np.asarray(sd["tangent"], float),
                      normal=np.asarray(sd["normal"], float),
                      length=float(sd["length_m"]))
    run = ep["runs"][0] if "runs" in ep else ep
    # the WALL-PARALLEL / TOUCHING phase only: every run also contains the
    # approach walk from the spawn, which is not the behaviour under comparison
    ticks = run.get("ticks", [])
    if ticks:
        lo, hi = ticks[0]["t"], ticks[-1]["t"]
    else:
        pr = [k["t"] for k in run.get("trace", []) if k["mode"] == "PROBE"]
        lo, hi = (min(pr) - 8.0, max(pr) + 8.0) if pr else (t[0], t[-1])
    window = (t >= lo) & (t <= hi)
    if ticks:
        # B and C log the controller's OWN measurement every tick -- exact, and
        # free of the ~1 deg map-frame error a qpos reconstruction carries over
        # an 8 m segment
        return {"ep": ep, "run": run, "seg": seg, "exact": True,
                "t": np.array([k["t"] for k in ticks]),
                "d": np.array([k["standoff_m"] for k in ticks]),
                "s": np.array([k["s_m"] for k in ticks]),
                "yaw": np.array([k["yaw_err_rad"] for k in ticks]) + seg.yaw,
                "window": np.ones(len(ticks), bool)}
    return {"ep": ep, "run": run, "seg": seg, "exact": False,
            "t": t, "yaw": yaw_map, "window": window,
            "d": np.array([seg.standoff_of(p) for p in xy]),
            "s": np.array([seg.arc_length_of(p) for p in xy])}


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    out = a.out or (a.dir / "compare.png")

    arms = [(k, lab, load(a.dir, tag)) for k, tag, lab in ARMS]
    arms = [(k, lab, d) for k, lab, d in arms if d is not None]
    if not arms:
        print(f"nothing to compare under {a.dir}")
        return 1

    fig, ax = plt.subplots(4, len(arms), figsize=(6.0 * len(arms), 13.0),
                           squeeze=False)
    for col, (key, label, D) in enumerate(arms):
        seg, t, d = D["seg"], D["t"], D["d"]
        run, ticks = D["run"], D["run"].get("ticks", [])
        inside = D["window"]
        # contact times: per-tick for B/C, per-press window for A
        if ticks:
            tt = np.array([k["t"] for k in ticks])
            ff = np.array([k["force_n"] for k in ticks])
            gg = np.array([k["ee_gap_m"] for k in ticks])
            c_t = tt[np.array([k["contact"] for k in ticks])]
        else:
            tr = run.get("trace", [])
            tt = np.array([k["t"] for k in tr])
            ff = np.array([max(k["force_l"], k["force_r"]) for k in tr])
            gg = None
            c_t = tt[ff >= 2.0] if len(tt) else np.array([])

        # ── 1. top-down ────────────────────────────────────────────────────
        # TOP-DOWN in the segment's own frame: along-wall vs distance-from-wall,
        # so the wall is the axis and the lane is a horizontal line
        A0 = ax[0][col]
        A0.axhspan(-0.06, 0.0, color="k", alpha=0.85, zorder=1)
        A0.plot(D["s"][inside], d[inside], "-", lw=1.6, color="#2b6cb0",
                label="base", zorder=3)
        for ct in c_t:
            i = int(np.argmin(np.abs(t - ct)))
            A0.plot([D["s"][i]], [d[i]], "o", ms=5, color="#c53030", zorder=4)
        A0.axhline(0.28, ls="--", lw=1.0, color="#718096", label="0.28 m lane",
                   zorder=2)
        A0.set_ylim(-0.06, 0.62)
        A0.set_title(f"{key} — {label}", fontsize=11, weight="bold")
        A0.set_xlabel("along wall [m]"); A0.set_ylabel("distance from wall [m]")
        A0.legend(fontsize=7, loc="upper right")

        # ── 2. base-wall distance ──────────────────────────────────────────
        A1 = ax[1][col]
        A1.plot(t[inside], d[inside], "-", lw=1.3, color="#2b6cb0")
        for ct in c_t:
            A1.axvline(ct, color="#c53030", lw=0.6, alpha=0.5)
        A1.axhline(0.28, color="#718096", ls="--", lw=1.0)
        dd = d[inside]
        A1.set_title(f"standoff  mean {dd.mean():.3f}  sd {dd.std():.3f} m",
                     fontsize=9)
        A1.set_xlabel("t [s]"); A1.set_ylabel("base-wall [m]")
        A1.set_ylim(0.15, 0.60)

        # ── 3. hand-wall distance ──────────────────────────────────────────
        A2 = ax[2][col]
        if gg is not None:
            A2.plot(tt, gg, "-", lw=1.1, color="#2f855a")
            A2.axhline(0.0, color="#718096", lw=0.8)
            A2.set_ylim(-0.05, 0.30)
            sw = [x.get("hand_excursion_m", 0.0) for x in run.get("taps", [])]
            A2.set_title("hand-wall gap   swing "
                         f"{(max(sw) if sw else 0.0):.3f} m", fontsize=9)
        else:
            for x in run.get("taps", []):
                A2.axvspan(0, 0, alpha=0)      # keep the axis alive
            A2.text(0.5, 0.5, "press primitive:\nno per-tick hand pose\n"
                              f"{len(run.get('taps', []))} presses, "
                              f"{np.median([x['duration_s'] for x in run['taps']]):.1f} s median",
                    ha="center", va="center", transform=A2.transAxes, fontsize=9)
            A2.set_title("hand-wall gap (not recorded)", fontsize=9)
        A2.set_xlabel("t [s]"); A2.set_ylabel("hand-wall [m]")

        # ── 4. force ───────────────────────────────────────────────────────
        A3 = ax[3][col]
        A3.plot(tt, ff, "-", lw=1.0, color="#744210")
        A3.axhline(25.0, color="#c53030", ls="--", lw=1.2, label="25 N limit")
        A3.axhspan(10.0, 25.0, color="#38a169", alpha=0.10, label="10-25 N band")
        peaks = [x["peak_force_n"] for x in run.get("taps", [])]
        A3.set_title("force   peaks "
                     + (", ".join(f"{p:.0f}" for p in peaks) if peaks else "-")
                     + " N", fontsize=9)
        A3.set_xlabel("t [s]"); A3.set_ylabel("hand force [N]")
        A3.set_ylim(0, max(30.0, float(ff.max()) * 1.1 if len(ff) else 30.0))
        A3.legend(fontsize=7, loc="upper right")

    fig.suptitle("Wall-touch behaviours, same segment / seed / speed",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    # ── the numeric roll-up the figure is a picture of ─────────────────────
    print(f"\n| arm | heading vs tangent | standoff mean+-sd | per-cycle base "
          f"excursion | contacts | peak N | visible swing |")
    print("|---|---|---|---|---|---|---|")
    for key, label, D in arms:
        seg, run = D["seg"], D["run"]
        inside = D["window"]
        dd, yy = D["d"][inside], D["yaw"][inside]
        yerr = np.degrees(np.abs((yy - seg.yaw + np.pi) % (2 * np.pi) - np.pi))
        note = "" if D["exact"] else " (from qpos)"
        taps = run.get("taps", [])
        exc = [x.get("cycle_excursion_m") for x in taps
               if x.get("cycle_excursion_m") is not None]
        hit = [x for x in taps if x.get("contacted") or x.get("status") == "hit"
               or x.get("status") == "over_limit"]
        pk = [x["peak_force_n"] for x in hit] or [0.0]
        sw = [x.get("hand_excursion_m") for x in taps
              if x.get("hand_excursion_m")]
        print(f"| {key} | {yerr.mean():.1f} deg{note} | {dd.mean():.3f} +- "
              f"{dd.std():.3f} | "
              f"{(f'{np.median(exc):.3f} m' if exc else 'n/a (one lane)')} | "
              f"{len(hit)}/{len(taps)} | {min(pk):.1f}-{max(pk):.1f} | "
              f"{(f'{max(sw):.3f} m' if sw else 'n/a')} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
