"""INJECTED parked-hand contact: prove the force guard works, physically.

Two unvalidated guard placements were shipped before this batch; the first was
re-run on hardware and measurably did nothing (force_aborts = 0, peak 43.2 N).
This script is the demonstration the next placement was owed BEFORE being
believed: it deliberately commands the base toward a mapped wall with both
hands parked -- the exact mechanism of all four benchmark violations -- and
then, from the stance the abort leaves, commands a turn against the same wall.

    PASS requires, measured:
      * the guard fires in LOCOMOTION (walk-in) and again in TURN (sweep);
      * every peak force stays below the 25 N safety limit;
      * the base holds still and both hands are parked after each abort;
      * the summary peak equals the exclusion-audit peak;
      * the robot can still walk AWAY afterwards (recovery is possible).

    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
      uv run --project tactile_controller python -m execution.injected_contact_check \
        --out out/batch3a2/safety/injected
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    from .backends import BackendConfig
    from .mujoco_backend import MujocoBackend, MujocoConfig
    from .scenes import build_scene

    scene = build_scene("two_turns")
    mj = MujocoConfig()
    be = MujocoBackend(scene, BackendConfig(), np.random.default_rng(0), mj,
                       render=not args.no_render,
                       video_path=None if args.no_render else args.out / "episode.mp4")

    checks: dict[str, bool] = {}
    log: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks[name] = bool(ok)
        line = f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}"
        log.append(line)
        print(line, flush=True)

    def hands_state():
        forces = {n: be._hand_total_n(h) for n, h in be.hands.items()}
        parked = all(abs(float(h.pose_command_b[0, 0]) - mj.park_x_m) <= 6e-3
                     for h in be.hands.values())
        return forces, parked

    # ── 1. walk INTO the wall (LOCOMOTION) ──────────────────────────────────────
    # wall 0's face is at map y = 0.38; a tucked wrist reaches it from ~0.26 m of
    # standoff, so commanding the base to y = 0.30 walks the parked hand into the
    # face well before the base arrives.
    status = be.goto_map(np.array([0.60, 0.30]), math.pi / 2.0,
                         np.zeros(2), math.pi / 2.0)
    check("locomotion_guard_fired", status == "force_abort",
          f"goto status={status}, events={len(be.force_abort_events)}")
    check("locomotion_event_recorded",
          bool(be.force_abort_events)
          and be.force_abort_events[0]["mode"] == "LOCOMOTION",
          f"events={be.force_abort_events}")
    forces, parked = hands_state()
    check("hands_retracted_after_abort", parked,
          f"pose_command x per hand ~park, forces={ {k: round(v,2) for k,v in forces.items()} }")
    check("forces_drained_after_abort", max(forces.values()) < 5.0,
          f"max residual {max(forces.values()):.2f} N")
    peak_after_walk = be.peak_force_n
    check("peak_below_limit_after_walk_in", peak_after_walk <= mj.force_limit_n,
          f"peak {peak_after_walk:.2f} N vs limit {mj.force_limit_n:.0f} N")

    # the base must be HOLDING after the stop: watch 2 s, judge the last 1 s
    # (the first second may still carry the decay of the safety stop itself);
    # the loop consumes any trip like every production loop does
    v_max, t0, held_events = 0.0, be._t, 0
    while be._t - t0 < 2.0:
        xy0, yaw0 = be.truth_pose()
        be._set_goal(xy0, yaw0, mj.speed_scale_locomotion, hold=True, standing=True)
        be._park_hand(be.hands["left"])
        be._park_hand(be.hands["right"])
        be._step()
        if be._guard_tripped():
            held_events += 1
        if be._t - t0 >= 1.0:
            v_max = max(v_max, float(np.linalg.norm(be.truth_pose()[0] - xy0)) / be.dt)
    check("base_holds_after_abort", v_max <= 0.12 and held_events == 0,
          f"max speed {v_max:.3f} m/s (last 1 s), re-trips during hold={held_events}")

    # ── 2. TURN against the wall ────────────────────────────────────────────────
    # The scenario 3A.1 measured: load accumulating during a TURN.  Approach
    # PARALLEL to the wall -- facing +x map, the parked hands lead along the
    # wall and stay clear -- then command a turn to face it: the hands sweep an
    # arc whose radius exceeds the remaining standoff.  A second head-on plow
    # would instead re-test LOCOMOTION at higher impact speed (measured 39 N
    # substep peak from momentum absorption alone), which is not this test.
    here = be.world_point_to_map(be.truth_pose()[0])
    status = be.goto_map(np.array([here[0] + 0.35, 0.19]), 0.0, here, 0.0)
    check("parallel_approach_stays_clean", status == "arrived",
          f"goto status={status} (hands lead along the wall, nothing to hit)")
    n0 = len(be.force_abort_events)
    status = be.turn_to_map(math.pi / 2.0, 0.0)
    turned_events = be.force_abort_events[n0:]
    check("turn_guard_fired",
          status == "force_abort" and bool(turned_events),
          f"turn status={status}, new events={turned_events}")
    if turned_events:
        check("turn_event_mode", turned_events[0]["mode"] in ("TURN", "STABILIZE"),
              f"mode={turned_events[0]['mode']}")
    check("peak_below_limit_after_turn", be.peak_force_n <= mj.force_limit_n,
          f"peak {be.peak_force_n:.2f} N")

    # ── 3. recovery: walk AWAY cleanly ──────────────────────────────────────────
    n0 = len(be.force_abort_events)
    status = be.goto_map(np.array([0.30, -0.10]), math.pi / 2.0,
                         np.array([0.55, 0.12]), math.pi / 2.0)
    check("recovery_walk_away", status == "arrived"
          and len(be.force_abort_events) == n0,
          f"status={status}, new events={len(be.force_abort_events) - n0}")

    # ── 4. unified metrics ──────────────────────────────────────────────────────
    summary = be.physical_summary()          # raises if summary peak != audit peak
    check("summary_peak_equals_audit_peak",
          abs(summary["peak_force_n"] - summary["audit_peak_force_n"]) <= 1e-6,
          f"{summary['peak_force_n']:.3f} == {summary['audit_peak_force_n']:.3f}")
    check("final_peak_below_limit", summary["peak_force_n"] <= mj.force_limit_n,
          f"{summary['peak_force_n']:.2f} N, force_aborts={summary['force_aborts']}")

    ok = all(checks.values())
    out = {
        "ok": ok, "checks": checks, "log": log,
        "force_abort_events": summary["force_abort_events"],
        "peak_force_n": summary["peak_force_n"],
        "audit_peak_force_n": summary["audit_peak_force_n"],
        "force_limit_n": mj.force_limit_n,
        "guard_abort_force_n": mj.guard_abort_force_n,
        "ticks": summary["ticks"],
        "sim_time_s": be.time_s,
    }
    (args.out / "injected_contact.json").write_text(json.dumps(out, indent=1))
    (args.out / "trace.json").write_text(json.dumps(be.trace))

    try:
        _plot(args.out, be.trace, out)
    except Exception as exc:                  # the figure must never lose the run
        print(f"  (figure skipped: {exc})", flush=True)
    be.close()
    print(f"\nINJECTED CONTACT CHECK: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def _plot(out: Path, trace: list[dict], summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [r["t"] for r in trace]
    fl = [r["force_l"] for r in trace]
    fr = [r["force_r"] for r in trace]
    modes = [r["mode"] for r in trace]

    fig, (ax, axm) = plt.subplots(
        2, 1, figsize=(10, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    ax.plot(t, fl, lw=0.9, label="left hand [N]")
    ax.plot(t, fr, lw=0.9, label="right hand [N]")
    ax.axhline(summary["force_limit_n"], color="crimson", ls="--", lw=1.2,
               label="25 N safety limit")
    ax.axhline(summary["guard_abort_force_n"], color="darkorange", ls=":", lw=1.2,
               label="12 N guard threshold")
    for ev in summary["force_abort_events"]:
        ax.axvline(ev["t"], color="k", lw=0.8, alpha=0.6)
        ax.annotate(f"abort {ev['mode']}\n{ev['force_n']:.1f} N", (ev["t"], ev["force_n"]),
                    textcoords="offset points", xytext=(6, 8), fontsize=8)
    ax.set_ylabel("hand force [N]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Injected parked-hand contact: the guard fires, force stays below the limit")

    mode_names = sorted(set(modes))
    ax_ids = [mode_names.index(m) for m in modes]
    axm.step(t, ax_ids, where="post", lw=0.9)
    axm.set_yticks(range(len(mode_names)), mode_names, fontsize=8)
    axm.set_xlabel("simulated time [s]")
    fig.savefig(out / "injected_contact.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
