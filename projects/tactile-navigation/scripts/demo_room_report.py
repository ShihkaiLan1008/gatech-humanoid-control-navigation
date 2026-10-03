"""Validation roll-up for a demo-room run: trajectory, belief, forces.

    PYTHONPATH=src python3 scripts/demo_room_report.py \\
        out/demo_map/run/demo_run_s0_full.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    doc = json.loads(Path(argv[0]).read_text())
    nav = doc["nav_trace"]
    touches = doc["touches"]
    phys = doc.get("physical", {})

    print(f"outcome            {doc['outcome']}")
    print(f"goal dist          true {doc['goal_dist_true_m']:.3f} m   "
          f"believed {doc['goal_dist_believed_m']:.3f} m")
    print(f"sim / wall         {nav[-1]['t']:.0f} s / {doc['wall_clock_s']:.0f} s")

    # -- trajectory / belief -------------------------------------------------
    errs = np.array([x["err_m"] for x in nav])
    sigs = np.array([x["sigma_m"] for x in nav])
    print("\nbelief vs truth")
    print(f"  |err|  start {errs[0]:.3f}  median {np.median(errs):.3f}  "
          f"p90 {np.percentile(errs, 90):.3f}  max {errs.max():.3f}  "
          f"final {errs[-1]:.3f} m")
    print(f"  sigma  start {sigs[0]:.3f}  min {sigs.min():.3f}  "
          f"max {sigs.max():.3f}  final {sigs[-1]:.3f} m")
    n_modes = [len(x.get("modes", [])) for x in nav]
    print(f"  modes  max {max(n_modes)}  final {n_modes[-1]}")
    # sigma drop across touch events
    drops = []
    for i, x in enumerate(nav[:-1]):
        if "station" in x["event"] and i + 1 < len(nav):
            drops.append(sigs[i + 1] - sigs[i])
    if drops:
        print(f"  sigma step at stations: median {np.median(drops):+.3f} m "
              f"(n={len(drops)})")

    # -- forces / contacts ---------------------------------------------------
    hits = [t for t in touches if t.get("contacted")]
    in_band = [t for t in hits if 10.0 <= (t["peak_force_n"] or 0) <= 25.0]
    over = [t for t in touches if (t.get("peak_force_n") or 0) > 25.0]
    print("\ntouches")
    print(f"  scheduled {len(touches)}  contacted {len(hits)}  "
          f"in 10-25 N {len(in_band)}  over 25 N {len(over)}")
    if hits:
        pk = np.array([t["peak_force_n"] for t in hits])
        print(f"  peak force  min {pk.min():.2f}  median {np.median(pk):.2f}  "
              f"max {pk.max():.2f} N")
        out = np.array([abs(t.get("outward_m") or 0.0) for t in hits])
        print(f"  |NET wall-normal per cycle|  median {np.median(out):.3f}  "
              f"max {out.max():.3f} m")
    for t in touches:
        print(f"    {t['label']:8s} s={t['station_m']:5.2f}  "
              f"{t['status']:10s} {(t.get('peak_force_n') or 0):6.2f} N  "
              f"in={t.get('turn_in', '')}/{t.get('turn_out', '')}  "
              f"{t.get('reason', '')}")

    print("\nsafety")
    print(f"  force aborts {phys.get('force_aborts')}   "
          f"peak hand force {phys.get('peak_force_n'):.2f} N   "
          f"over-limit time {phys.get('over_limit_s')} s")
    print(f"  replans/recoveries: {len(doc.get('replans', []))}")
    for r in doc.get("replans", []):
        print(f"    t={r['t']:.0f}s {r.get('label')}: {r.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
