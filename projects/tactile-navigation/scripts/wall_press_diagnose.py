"""Read one turn-to-press cycle's geometry log and say what disagreed.

`press()` reports `sweep_stopped_short_of_mapped_face` whenever the measured
hand ends up behind the kiss plane with no force on it.  That single string
covers three different faults -- the wrong wall was followed, the right wall
was followed but the base is not facing it, or the arm simply never tracked
its command -- and they need different fixes.  This separates them from the
per-tick record `--press-trace` writes.

    python scripts/wall_press_diagnose.py out/hw/turn/wp_r5_twinbay_s0_diagA.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _f(v, n=3):
    return "n/a" if v is None else f"{float(v):+.{n}f}"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    doc = json.loads(Path(argv[0]).read_text())
    trace = doc.get("press_trace") or []
    for run in doc.get("runs", []):
        for i, tap in enumerate(run.get("taps", [])):
            if "geometry" not in tap:
                continue
            print(f"\n=== press {i} @ station {tap['station_m']:.2f} m "
                  f"(segment {run.get('segment_index')}) ===")
            print(f"  turn_in  {tap.get('turn_in')}   "
                  f"turn_out {tap.get('turn_out')}")
            print(f"  reason   {tap.get('reason')!r}  peak "
                  f"{tap.get('peak_force_n', 0):.2f} N  "
                  f"press {tap.get('press_s', 0):.1f} s")
            print("  --- followed-wall slot at each stage ---")
            for g in tap["geometry"]:
                print(f"    {g['tag']:18s} slot {g['wall_slot']}  side "
                      f"{g['follow_side']:+.0f}  n {_f(g['wall_normal_w'][0],2)},"
                      f"{_f(g['wall_normal_w'][1], 2)}  "
                      f"base->face {_f(g['base_to_face_m'])} m  "
                      f"facing.n {_f(g['facing_dot'], 2)}  "
                      f"kiss {_f(g['kiss_x_m'])} m")
            g0 = tap["geometry"][0]
            print(f"  frozen base   xy {_f(g0['base_xy_w'][0])}, "
                  f"{_f(g0['base_xy_w'][1])}  yaw "
                  f"{math.degrees(g0['base_yaw_w']):+.1f} deg")
            print(f"  wall plane    centre {_f(g0['wall_centre_w'][0])}, "
                  f"{_f(g0['wall_centre_w'][1])}  yaw "
                  f"{math.degrees(g0['wall_yaw_w']):+.1f} deg  half-len "
                  f"{g0['wall_half_len_m']:.2f}  half-thick "
                  f"{g0['wall_half_thickness_m']:.3f} m")

    ticks = [g for g in trace if g["tag"] == "press"]
    if not ticks:
        print("\n(no per-tick press trace; re-run with --press-trace)")
        return 0
    print(f"\n--- per-tick press ({len(ticks)} ticks) ---")
    print("      t   slot  kiss   cmd_x  ee_x  ee->face  cmd_y  ee_y   "
          "F_ax  F_tot  pairs")
    for g in ticks[:: max(1, len(ticks) // 40)]:
        print(f"  {g['t']:7.2f}  {g['wall_slot']:3d} {_f(g['kiss_x_m'])} "
              f"{_f(g['cmd_ee_b'][0])} {_f(g['meas_ee_b'][0])} "
              f"{_f(g['ee_to_face_m'])}  {_f(g['cmd_ee_b'][1])} "
              f"{_f(g['meas_ee_b'][1])} {g['force_axis_n']:6.2f} "
              f"{g['force_total_n']:6.2f}  "
              f"{[p['sensor'] for p in g['pairs']]}")

    last = ticks[-1]
    kiss = last["kiss_x_m"]
    ee_x = last["meas_ee_b"][0]
    print("\n--- verdict ---")
    slots = {g["wall_slot"] for g in ticks}
    print(f"  followed slot(s) during the press: {sorted(slots)}")
    print(f"  final  kiss {kiss:+.3f}  measured ee_x {ee_x:+.3f}  "
          f"swept past kiss {ee_x - kiss:+.3f} m")
    print(f"  final  true base->face {_f(last['base_to_face_m'])} m, "
          f"hand->face {_f(last['ee_to_face_m'])} m, "
          f"facing.normal {_f(last['facing_dot'], 2)}")
    if abs(last["base_to_face_m"] - kiss) > 0.08:
        print("  => the kiss plane and the TRUE followed face disagree: the "
              "wall geometry the press solves against is not the one in front "
              "of the hand.")
    elif last["ee_to_face_m"] > 0.02:
        print("  => geometry agrees; the ARM never reached the face (it stops "
              f"{last['ee_to_face_m']:+.3f} m short with "
              f"{last['force_axis_n']:.2f} N on it).")
    else:
        print("  => the hand reached the face; the fault is in force reading "
              "or in the stop rule, not the geometry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
