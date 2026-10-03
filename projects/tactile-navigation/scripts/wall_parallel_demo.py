"""WALL_PARALLEL_INTERMITTENT_TOUCH -- constant-standoff walking, brief taps.

    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
      tactile_controller/.venv/bin/python scripts/wall_parallel_demo.py \
        --scene r5_twinbay --seed 0 --standoff 0.30 --spacing 0.8 --out out/hw/wp

`--no-taps` runs the base half alone (Subtask 1: is the lane held?).  Segment
choice, travel direction, hand and tap stations all come from the map geometry
(`execution.wall_lane`); nothing here knows a scene name.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from execution.wall_lane import tap_stations, wall_segments   # noqa: E402
from lane_graph.scene3 import build_v3_scene                  # noqa: E402


FLAT_ARM = dict(wp_carry_x_m=0.34, wp_carry_y_m=0.12, wp_carry_z_m=0.15,
                wp_reach_rate_m_s=0.12, wp_retract_rate_m_s=0.12,
                wp_release_rate_m_s=0.60, wp_release_m=0.08,
                wp_prep_lead_m=0.55, wp_gap_lead_m=0.55, wp_reach_y_m=0.40,
                wp_tap_hold_rate_m_s=9.0, wp_tap_timeout_s=3.0)
"""The pre-16 arm, as a config rather than as dead code: carried AT the contact
pose so only `y` ever moved, no staged reach (`gap_lead` = `prep_lead`), no
pre-extension cap, and no gait-phase hold (a threshold no sway reaches).  The
A/B/C comparison then runs ONE code path and the only difference between B and
C is the arm."""


def backend(scene, seed: int, standoff_m: float, render: bool = False,
            yaw_stiffness: float | None = None, arm_style: str = "reach",
            pin_wall: bool = True, press_rate: float | None = None):
    from execution.backends import BackendConfig
    from execution.mujoco_backend import MujocoConfig
    from lane_graph.mj_backend2 import MujocoBackendV2
    from lane_graph.run_v3 import PRESS_TARGET_N

    class _V3(MujocoBackendV2):
        def _press_target_n(self):
            return PRESS_TARGET_N

    return _V3(scene, BackendConfig(probe_force_n=PRESS_TARGET_N),
               np.random.default_rng(seed),
               MujocoConfig(seed=11 + seed, episode_budget_s=2000.0,
                            press_overlap_m=0.20,
                            wp_yaw_stiffness=yaw_stiffness,
                            wp_pin_wall=pin_wall,
                            **({} if press_rate is None else
                               {"wp_press_speed_m_s": press_rate}),
                            **(FLAT_ARM if arm_style == "flat" else {}),
                            # the FSM's own notion of a legal wall edge, so the
                            # band it samples brackets the lane we ask for
                            wall_standoff_range=(max(0.10, standoff_m - 0.08),
                                                 standoff_m + 0.08)),
               render=render, video_path=None)


def _corridor_clear(a, b, bodies, *, half_width_m: float = 0.24,
                    step_m: float = 0.1) -> bool:
    """Is the STRAIGHT walk from a to b clear?

    `goto_map` is a goal attractor, not a route planner: it walks the
    displacement it is given.  Two of five maps put their longest wall across
    the room from the spawn, and the straight approach walked into an
    intervening wall (0.00-0.03 m travelled, force abort).  Choosing a segment
    the approach can actually reach is map geometry, not a scene rule -- and
    routing to the others is the global planner's job, which this work does not
    touch.  The default half-width is the base's own, not the 0.28 m LANE_CLEAR
    the lane uses: a transit only has to fit, while a lane also has to be a
    place the robot can stand and reach from.  0.20 m was measured too generous
    -- an r4 approach with 0.212 m of clearance aborted on contact -- so the
    default is 0.24."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.linalg.norm(b - a) / step_m) + 1)
    for p in np.linspace(a, b, n):
        if min((float(o.signed_distance(p.reshape(1, 2))[0][0]) for o in bodies),
               default=np.inf) < half_width_m:
            return False
    return True


def candidates(scene, standoff_m: float, spacing_m: float, *, min_length_m: float,
               min_taps: int, start_xy=None):
    bodies = scene.plan.bodies
    start = np.asarray(scene.start_map_xy if start_xy is None else start_xy, float)
    out = []
    for seg in wall_segments(bodies, standoff_m=standoff_m,
                             min_length_m=min_length_m, both_directions=True):
        st = tap_stations(seg, spacing_m, bodies, standoff_m=standoff_m)
        if len(st) < min_taps:
            continue
        lane0 = seg.lane_point(0.0, standoff_m)
        out.append((seg, st, _corridor_clear(start, lane0, bodies)))
    out.sort(key=lambda p: (not p[2], -round(p[0].length, 2),
                            float(np.linalg.norm(p[0].lane_point(0.0, standoff_m)
                                                 - start))))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="r5_twinbay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--standoff", type=float, default=0.30)
    ap.add_argument("--spacing", type=float, default=0.6)
    ap.add_argument("--arm-style", choices=("reach", "flat"), default="reach")
    ap.add_argument("--touch-style",
                    choices=("lateral", "turn_press", "crab_press"),
                    default="lateral")
    ap.add_argument("--segment", type=int, default=0)
    ap.add_argument("--segments", type=int, nargs="+", default=None,
                    help="run several segments back to back in one episode")
    ap.add_argument("--min-length", type=float, default=1.8)
    ap.add_argument("--min-taps", type=int, default=2)
    ap.add_argument("--no-taps", action="store_true")
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--command-standoff", type=float, default=None)
    ap.add_argument("--yaw-stiffness", type=float, default=None)
    ap.add_argument("--tap-force", type=float, default=None)
    ap.add_argument("--max-taps", type=int, default=None)
    ap.add_argument("--s-to", type=float, default=None)
    ap.add_argument("--press-rate", type=float, default=None,
                    help="override wp_press_speed_m_s (m/s)")
    ap.add_argument("--no-pin", action="store_true",
                    help="do not pin the followed wall before the press")
    ap.add_argument("--press-trace", action="store_true",
                    help="log per-tick press geometry (press_snapshot)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    scene = build_v3_scene(a.scene)
    cands = candidates(scene, a.standoff, a.spacing,
                       min_length_m=a.min_length, min_taps=a.min_taps)
    if not cands:
        print(f"{a.scene}: no segment with >= {a.min_taps} taps")
        return 1
    for k, (seg, st, reach) in enumerate(cands):
        p0 = seg.lane_point(0.0, a.standoff)
        p1 = seg.lane_point(seg.length, a.standoff)
        print(f"[{k}] len {seg.length:5.2f} m  lane {np.round(p0, 2)} -> "
              f"{np.round(p1, 2)}  yaw {math.degrees(seg.yaw):+7.1f} deg  "
              f"{'left' if seg.hand_side > 0 else 'right'} hand  "
              f"{len(st)} taps  {'' if reach else '(no straight approach)'}")
    if a.list:
        return 0

    picks = a.segments if a.segments is not None else [a.segment]
    picks = [p for p in picks if 0 <= p < len(cands)]
    if not picks:
        print("no valid segment index")
        return 1

    t_wall = time.time()
    be = backend(scene, a.seed, a.standoff, yaw_stiffness=a.yaw_stiffness,
                 arm_style=a.arm_style, pin_wall=not a.no_pin,
                 press_rate=a.press_rate)
    if a.press_trace:
        be._press_trace = []
    out = {"scene": a.scene, "seed": a.seed, "standoff_m": a.standoff,
           "spacing_m": a.spacing, "taps_enabled": not a.no_taps,
           "arm_style": a.arm_style, "touch_style": a.touch_style,
           "segment_indices": picks, "runs": []}
    try:
        for k in picks:
            seg, st, reach = cands[k]
            # a LATERAL lane needs no bias inversion; `standoff_bias_m`
            # (+0.078 m) was measured facing the wall, not beside it
            cmd_off = (a.command_standoff if a.command_standoff is not None
                       else a.standoff
                       - (be.mj.wp_crab_standoff_bias_m
                          if a.touch_style == "crab_press"
                          else be.mj.wp_turn_standoff_bias_m
                          if a.touch_style == "turn_press"
                          else be.mj.wp_standoff_bias_m))
            s_to = seg.length if a.s_to is None else min(seg.length, a.s_to)
            # approach: stand on the TARGET lane at s = 0.  The biased command
            # is an actuation detail of the walk; approaching to it put the
            # base inside the parked hands' reach and tripped the guard.
            lane0 = seg.lane_point(0.0, a.standoff)
            here = be.world_point_to_map(be.truth_pose()[0])
            goto = be.goto_map(lane0, seg.yaw, here,
                               be.world_yaw_to_map(be.truth_pose()[1]))
            turn = be.turn_to_map(seg.yaw, be.world_yaw_to_map(be.truth_pose()[1]))
            # the turn drifts the base off the lane -- measured 0.46 m of
            # standoff after a 180 deg turn onto a 0.28 m lane -- so close it
            # again now the heading is right, before any tap is scheduled
            goto = be.goto_map(lane0, seg.yaw,
                               be.world_point_to_map(be.truth_pose()[0]),
                               be.world_yaw_to_map(be.truth_pose()[1]),
                               tol_m=0.07)
            arrive = be.world_point_to_map(be.truth_pose()[0])
            print(f"\n# segment {k}: goto {goto}, turn {turn}, "
                  f"standoff at start {seg.standoff_of(arrive):+.3f} m, "
                  f"s {seg.arc_length_of(arrive):+.3f} m")

            r = be.wall_parallel_touch(
                seg, a.standoff, a.spacing,
                stations=([] if a.no_taps else st), s_to=s_to,
                speed_scale=a.speed, command_standoff_m=cmd_off,
                tap_force_n=a.tap_force, touch_style=a.touch_style,
                max_taps=a.max_taps)
            r.update({"segment_index": k, "goto": goto, "turn": turn,
                      "straight_approach": bool(reach),
                      "standoff_at_start_m": seg.standoff_of(arrive)})
            out["runs"].append(r)

            print(f"  {r['status']}  travelled {r['travelled_m']:+.2f} m in "
                  f"{r['duration_s']:.1f} s")
            print(f"  standoff  mean {r['standoff_mean_m']:.3f}  sd "
                  f"{r['standoff_std_m']:.3f}  range [{r['standoff_min_m']:.3f}, "
                  f"{r['standoff_max_m']:.3f}] m   (target {a.standoff:.2f}, "
                  f"commanded {cmd_off:.3f})")
            print(f"  heading   mean {math.degrees(r['yaw_err_mean_rad']):.2f} deg"
                  f"  max {math.degrees(r['yaw_err_max_rad']):.2f} deg")
            print(f"  peak hand force {r['peak_force_n']:.2f} N   taps "
                  f"{len(r['taps'])}")
            for i, t in enumerate(r["taps"]):
                if "turn_in" in t:
                    print(f"    press {i} @ s={t['station_m']:.2f}  "
                          f"{t['status']:10s} peak {t['peak_force_n']:6.2f} N  "
                          f"press {t.get('press_s', 0):.1f}s  standoff "
                          f"{t['standoff_before_m']:.3f} -> "
                          f"{t.get('standoff_faced_m', 0):.3f} (turn) -> "
                          f"{t.get('standoff_pressed_m', 0):.3f} (press) -> "
                          f"{t.get('standoff_after_m', 0):.3f}  "
                          f"NET {t.get('outward_m', 0):+.3f} m  "
                          f"turns {t.get('turn_in')}/{t.get('turn_out')}  "
                          f"pairs {t['pairs']}")
                    continue
                print(f"    tap {i} @ s={t['station_m']:.2f}  {t['status']:10s} "
                      f"peak {t['peak_force_n']:6.2f} N  contact "
                      f"{t.get('contact_s', 0.0):.2f} s/{t['contact_ticks']}tk  standoff "
                      f"{t['standoff_before_m']:.3f} -> "
                      f"{t.get('standoff_after_m', float('nan')):.3f} "
                      f"(out {t.get('outward_m', float('nan')):+.3f})  fwd "
                      f"{t.get('forward_m', float('nan')):+.3f} m  "
                      f"reach {t.get('reach_s', 0.0):.2f}s "
                      f"retract {t.get('retract_s', 0.0):.2f}s "
                      f"swing {t.get('hand_excursion_m', 0.0):.3f}m  "
                      f"pairs {t['pairs']}")
            for sk in r.get("skipped", []):
                print(f"    SKIPPED station {sk['station_m']:.2f}: lane err "
                      f"{sk['lane_err_m']:.3f} m, drift "
                      f"{sk['lane_drift_m']:.3f} m, band "
                      f"{sk['lane_band_m']:.3f} m")
    finally:
        summary = be.physical_summary()
        if getattr(be, "state_log", None):
            a.out.mkdir(parents=True, exist_ok=True)
            ts = np.asarray([t for t, _ in be.state_log], dtype=np.float64)
            qs = np.stack([q for _, q in be.state_log]).astype(np.float32)
            np.savez_compressed(a.out / f"states_{a.scene}_s{a.seed}"
                                        f"{a.tag}.npz", t=ts, qpos=qs)
        be.close()

    if a.press_trace:
        out["press_trace"] = be._press_trace
    out["physical"] = summary
    out["wall_clock_s"] = round(time.time() - t_wall, 1)
    a.out.mkdir(parents=True, exist_ok=True)
    tag = a.tag or ("_notaps" if a.no_taps else "")
    (a.out / f"wp_{a.scene}_s{a.seed}{tag}.json").write_text(
        json.dumps(out, indent=1, default=float))
    print(f"\nsafety: peak {summary.get('peak_hand_force_n')} N  aborts "
          f"{summary.get('force_aborts')}  audit clean "
          f"{summary.get('exclusion_audit', {}).get('clean')}")
    print(f"wrote {a.out}  ({out['wall_clock_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
