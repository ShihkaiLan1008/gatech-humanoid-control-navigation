"""Tabulate a wall-parallel sweep against the acceptance criteria.

    python3 scripts/wall_parallel_report.py out/hw/wpx

Reads every `wp_*.json` a sweep wrote and prints (1) the per-run lane table,
(2) the per-tap table, and (3) the acceptance roll-up.  No simulator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

BAND_N = (10.0, 25.0)


def rows(out_dir: Path):
    for f in sorted(out_dir.glob("wp_*.json")):
        d = json.loads(f.read_text())
        for r in d["runs"]:
            yield f.name, d, r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--band", type=float, nargs=2, default=list(BAND_N))
    a = ap.parse_args(argv)
    lo, hi = a.band

    runs = list(rows(a.out))
    if not runs:
        print(f"no wp_*.json under {a.out}")
        return 1

    print("| scene | seed | seg | hand | travelled | standoff mean+-sd | "
          "range | heading | taps hit/sched | peak N | status |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    all_taps, aborts, falls, over = [], 0, 0, 0
    for _name, d, r in runs:
        hits = [t for t in r["taps"] if t["status"] == "hit"]
        sched = len(r["stations_m"])
        all_taps += r["taps"]
        aborts += int(bool(r.get("guard")))
        falls += int(r["status"] == "fell")
        print(f"| {d['scene']} | {d['seed']} | {r['segment_index']} | "
              f"{r['hand']} | {r['travelled_m']:.2f} m | "
              f"{r['standoff_mean_m']:.3f} +- {r['standoff_std_m']:.3f} | "
              f"[{r['standoff_min_m']:.3f}, {r['standoff_max_m']:.3f}] | "
              f"{math.degrees(r['yaw_err_mean_rad']):.1f} deg | "
              f"{len(hits)}/{sched} | {r['peak_force_n']:.1f} | {r['status']} |")

    peaks = [t["peak_force_n"] for t in all_taps if t["status"] != "miss"]
    inband = [p for p in peaks if lo <= p <= hi]
    over = [p for p in peaks if p > hi]
    dur = [t.get("contact_s", 0.0) for t in all_taps if t["status"] != "miss"]
    outw = [t["outward_contact_m"] for t in all_taps
            if t.get("outward_contact_m") is not None]
    pairs = [t for t in all_taps if t["pairs"]]
    misses = [t for t in all_taps if t["status"] == "miss"]
    skipped = sum(len(r.get("skipped", [])) for _n, _d, r in runs)
    travelled = sum(r["travelled_m"] for _n, _d, r in runs)
    sd = [r["standoff_std_m"] for _n, _d, r in runs]

    print(f"\n{len(runs)} runs, {travelled:.1f} m travelled, "
          f"{len(all_taps)} taps ({len(all_taps) - len(misses)} contacted, "
          f"{len(misses)} bounded misses, {skipped} stations skipped off-lane)")
    print(f"  lane      standoff sd per run {min(sd):.3f}-{max(sd):.3f} m")
    if peaks:
        print(f"  force     peak {min(peaks):.1f}-{max(peaks):.1f} N, "
              f"median {np.median(peaks):.1f};  in {lo:.0f}-{hi:.0f} N: "
              f"{len(inband)}/{len(peaks)};  over {hi:.0f} N: {len(over)}")
    if dur:
        print(f"  duration  contact {min(dur):.2f}-{max(dur):.2f} s, "
              f"median {np.median(dur):.2f} s")
    if outw:
        print(f"  base      displacement from first contact "
              f"{min(outw):+.3f} to {max(outw):+.3f} m, "
              f"mean {np.mean(outw):+.3f}, |mean| {abs(np.mean(outw)):.3f}")
    print(f"  provenance {len(pairs)}/{len(all_taps)} taps carry a named MuJoCo "
          f"contact pair")
    print(f"  safety    force aborts {aborts}, falls {falls}, "
          f"over-limit taps {len(over)}")

    print("\nacceptance")
    checks = [
        ("travels >= 1-2 m parallel to the wall",
         all(r["travelled_m"] >= 1.5 for _n, _d, r in runs)),
        ("standoff quantitatively stable (sd <= 0.03 m every run)",
         all(s <= 0.03 for s in sd)),
        ("multiple real lateral contacts",
         len(all_taps) - len(misses) >= 2 * len(runs)),
        (f"every contact in {lo:.0f}-{hi:.0f} N, never above {hi:.0f}",
         bool(peaks) and not over and len(inband) == len(peaks)),
        ("contact brief (<= 0.25 s)", bool(dur) and max(dur) <= 0.25),
        ("no deliberate retreat after a touch (|mean outward| <= 0.01 m)",
         bool(outw) and abs(float(np.mean(outw))) <= 0.01),
        ("no falls, no unsafe aborts", aborts == 0 and falls == 0),
        ("works on multiple segments without scene-specific rules",
         len({(d["scene"], r["segment_index"]) for _n, d, r in runs}) >= 3),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
