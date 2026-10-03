"""Compile the demo room with room2's planner logic, and check it before it runs.

    PYTHONPATH=src python3 scripts/demo_room_ate_plan.py --out out/demo_map

room2's compiler is one sentence -- ATE owns the landmarks, the flat runs keep
the rule-based cadence, arrival is confirmed by touching something that is not
a wall -- and `demo_map.scene_demo.plan_steps_demo_room` is that sentence
applied to the scanned demo room.  This script compiles it and answers, for
every scheduled touch, the only question that matters before spending an hour
of physics on it: WILL THE HAND FIND ANYTHING THERE?

  * a rhythm station is checked with `face_present`, the same rule the
    executor drops stations by;
  * a landmark press is checked by raycasting the press ray and asking which
    BODY it hits.  `face_present` is the wrong test here and the executor
    already knows it: it demands the hit surface normal match the segment's to
    within 0.999 (2.6 deg), while a bulge is drawn as a 20-gon whose facets
    are quantised to 18 deg and a scanned protrusion sits a couple of degrees
    off its own wall.  Both press perfectly well; only the normal check fails.
  * a corner face is checked by enumerating the ATE's own candidates.

`--simulate` goes one step further and runs the landmark VISIT -- the same
`landmark_probes` -> `landmark_next_probe` -> belief-update loop
`DemoRouteExecutor._ate_pillar` runs -- against a kinematic press oracle
instead of physics, so the claim "the EIG decides how many times to press,
and stops on its own" can be checked without MuJoCo.  What it does NOT
exercise is the motion half of the visit: `_reacquire`, `_press_station`, the
force shield and the stance ladder need the simulator.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demo_map.executor import (feasible_stations,
                               DemoExecCfg, landmark_next_probe,     # noqa: E402
                                landmark_probes, landmark_segment)
from demo_map.landmark_ate import AteProbe, LandmarkATE             # noqa: E402
from demo_map.scene_demo import (build_demo_scene, demo_route,      # noqa: E402
                                 plan_steps_demo_room)
from execution.wall_lane import WallSegment, face_present, raycast   # noqa: E402
from tactile_explore.noise import REGISTERED                        # noqa: E402

FAN_M = 0.18        # the hand lands this far along the lane past the station


def _ate_for(step):
    if step["shape"] == "box":
        return LandmarkATE.box(step["label"], step["centre"], step["half_xy"],
                               step["yaw"])
    return LandmarkATE.cylinder(step["label"], step["centre"], step["radius"])


def check(steps, scene, standoff_m: float) -> dict:
    bodies = tuple(scene.plan.bodies)
    names = [b.name for b in bodies]
    rows, problems = [], []
    for st in steps:
        k = st["kind"]
        if k == "follow":
            seg = st["seg"]
            miss = [s for s in st["stations"]
                    if not face_present(seg, float(s), bodies,
                                        standoff_m=standoff_m)]
            rows.append({"step": "follow", "label": st["label"],
                         "stations": len(st["stations"]),
                         "faceless": len(miss)})
            problems += [f"{st['label']} station {s} has no face" for s in miss]
        elif k == "ate_landmark":
            n = np.asarray(st["normal"], float)
            t = np.array([-n[1], n[0]])
            reach = float(st.get("reach_m") or
                          (st["half_xy"][1] if st["shape"] == "box"
                           else st["radius"]))
            apex = np.asarray(st["centre"], float) + reach * n
            lane = apex + standoff_m * n
            d, _nn, hit = raycast(lane, -n, standoff_m + 0.08, bodies)
            hit_name = None if d is None else names[hit]
            ate = _ate_for(st)
            seg, s_station = landmark_segment(st)
            # the SAME candidate set the executor will rank, so a plan that
            # checks out here is a plan whose visit has probes to choose from
            cands = landmark_probes(st, seg, s_station, bodies,
                                    standoff_m=standoff_m, search_span_m=0.45)
            ranked = ate.rank(list(cands), tie_break_xy=st["centre"]) if cands else []
            probe = ranked[0] if ranked else None
            n_press = sum(1 for c in cands if c.kind == "press")
            n_edge = sum(1 for c in cands if c.kind == "edge")
            ok = hit_name == st["label"]
            rows.append({"step": "ate_landmark", "label": st["label"],
                         "shape": st["shape"], "required": st["required"],
                         "press_hits": hit_name,
                         "press_depth_m": None if d is None else round(float(d), 3),
                         "candidates": len(cands), "on_body": n_press,
                         "edge_probes": n_edge,
                         "eig_nats": round(probe.score_nats, 3) if probe else 0.0,
                         "worth_pressing": bool(probe) and ate.worth_pressing(probe),
                         "ok": ok})
            if not ok:
                problems.append(f"{st['label']}: press ray hits {hit_name}")
            if probe is None or not ate.worth_pressing(probe):
                problems.append(f"{st['label']}: EIG below threshold")
            if n_press == 0:
                problems.append(f"{st['label']}: no candidate lands on the body")
        elif k == "ate_corner":
            ate = LandmarkATE.convex_corner(
                st["label"], st["corner"], st["faces"][0]["face_n"],
                st["faces"][1]["face_n"]) if st.get("convex") else \
                LandmarkATE.corner(st["label"], st["corner"],
                                   st["faces"][0]["face_n"],
                                   st["faces"][1]["face_n"])
            for i, fc in enumerate(st["faces"]):
                seg = fc["seg"]
                # the SAME question the executor asks -- the corner's band is
                # discovered, so a validator that re-derives it from a window
                # would certify a plan the run does not take
                cands = [AteProbe(seg.name, s0,
                                  seg.surface_point(s0 + FAN_M),
                                  -np.asarray(seg.normal, float))
                         for s0 in feasible_stations(seg, bodies, standoff_m)]
                r = ate.rank(list(cands), tie_break_xy=st["corner"]) if cands else []
                rows.append({"step": "ate_corner", "label": st["label"],
                             "face": i, "convex": bool(st.get("convex")),
                             "candidates": len(cands),
                             "best_nats": round(r[0].score_nats, 3) if r else 0.0,
                             "ok": bool(r)})
                if not r:
                    problems.append(f"{st['label']} face {i}: no candidates")
    return {"rows": rows, "problems": problems}


def _press_oracle(seg, s_hand, bodies, standoff_m, label, err, rng):
    """Would a press aimed at ``s_hand`` contact, and where would it read?

    KINEMATIC, not physical: the map IS the truth here and ``err`` is the
    robot's pose error, so a hand commanded to the map point ``q`` actually
    arrives at ``q - err``.  What the belief then SEES is the true contact
    composed back through the believed pose, i.e. shifted by ``+err`` -- the
    same duality `landmark_ate` rests on, run forwards.  The registered
    sensor noise (5 mm, 5 deg) is applied on top, so the belief is fed
    exactly the quality of measurement the robot produces.

    Returns ``(point, normal)`` in the belief's frame, or ``None`` for a
    sweep that found nothing."""
    n = np.asarray(seg.normal, float)
    lane_true = seg.lane_point(float(s_hand), standoff_m) - err
    d, nn, hit = raycast(lane_true, -n, standoff_m + 0.08, tuple(bodies))
    if d is None or bodies[hit].name != label:
        return None
    p = lane_true - float(d) * n + err
    p = p + rng.normal(0.0, REGISTERED.point_sigma_m, size=2)
    a = float(rng.normal(0.0, REGISTERED.normal_sigma_rad))
    ca, sa = np.cos(a), np.sin(a)
    nn = np.asarray(nn, float)
    return p, np.array([ca * nn[0] - sa * nn[1], sa * nn[0] + ca * nn[1]])


def simulate_episode(step, bodies, cfg, err, seed: int = 0) -> dict:
    """One landmark visit's DECISION loop, without the motion.

    Runs the same three calls `_ate_pillar` runs -- `landmark_probes`,
    `landmark_next_probe`, `feed_contact`/`feed_miss` -- so what is measured
    here is the shipped policy, not a re-description of it.  Everything the
    simulator owns (walking to the stance, the press itself, the force
    shield, the stance ladder) is replaced by `_press_oracle`, and any claim
    about those belongs to a physical run."""
    ate = _ate_for(step)
    centre = np.asarray(step["centre"], float)
    required = bool(step.get("required", False))
    seg, s_station = landmark_segment(step)
    rng = np.random.default_rng(seed)
    executed, presses, n_exec, fails = [], [], 0, 0
    stop = "budget_exhausted"
    max_tries = int(step.get("max_tries", cfg.gate_extra_presses))
    while n_exec < max(int(cfg.landmark_press_budget), max_tries):
        if fails >= max_tries:
            stop = "max_tries"
            break
        cands = landmark_probes(step, seg, s_station, bodies,
                                standoff_m=cfg.standoff_m,
                                search_span_m=max(0.45, 3.0 * float(
                                    np.linalg.norm(err))),
                                executed=executed)
        best, ranked, why = landmark_next_probe(
            ate, cands, centre,
            contacted=any(e["outcome"] == "contact" for e in presses))
        if best is None:
            stop = why
            break
        sc = [q.score_nats for q in ranked if q.score_nats > 0.0]
        row = {"try": n_exec + 1, "offset_m": round(float(best.s - s_station), 3),
               "kind": best.kind, "acquiring": why == "acquiring",
               "n_candidates": len(ranked), "n_scoring": len(sc),
               "eig_nats": round(float(best.score_nats), 3),
               "nats_spread": round(float(max(sc) - min(sc)), 3)
               if len(sc) > 1 else 0.0,
               "sigma_before": [round(v, 4) for v in
                                ate.offset_estimate()["sigma"]],
               "entropy_before": round(ate.entropy(), 3)}
        executed.append(float(best.s - s_station))
        n_exec += 1
        felt = _press_oracle(seg, best.s + FAN_M, bodies, cfg.standoff_m,
                             step["label"], err, rng)
        if felt is not None:
            ate.feed_contact(*felt)
            row["outcome"] = "contact"
            fails = 0
        else:
            ate.feed_miss(best)
            row["outcome"] = "miss"
            fails += 1
        row["sigma_after"] = [round(v, 4) for v in
                              ate.offset_estimate()["sigma"]]
        row["entropy_after"] = round(ate.entropy(), 3)
        presses.append(row)
    est = ate.offset_estimate()
    return {"label": step["label"], "shape": step["shape"],
            "required": required, "presses": presses, "stop": stop,
            "n_presses": n_exec,
            "n_contacts": sum(1 for e in presses if e["outcome"] == "contact"),
            "offset_error_m": round(float(np.linalg.norm(
                np.asarray(est["mean"][:2]) - centre - err)), 4),
            "sigma_final": [round(v, 4) for v in est["sigma"]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--spacing", type=float, default=None)
    ap.add_argument("--simulate", action="store_true",
                    help="run each landmark VISIT against a kinematic press "
                         "oracle and report how many presses the EIG takes")
    ap.add_argument("--err", type=str, default="0.06,-0.03",
                    help="robot pose error the oracle simulates, 'dx,dy' [m]")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    cfg = DemoExecCfg()
    if a.spacing is not None:
        cfg.press_spacing_m = a.spacing
    scene, route = build_demo_scene(), demo_route()
    steps, report = plan_steps_demo_room(cfg, scene, route)
    res = check(steps, scene, cfg.standoff_m)

    print(f"demo room: {len(steps)} steps, {report['n_rhythm']} rhythm presses, "
          f"{len(report['visited'])} landmark visits "
          f"({report['n_bodies']} non-wall bodies in the room)")
    print(f"goal-confirming landmark: {report['goal_landmark']}")
    if report["legs_without_rhythm"]:
        print(f"legs with no rule-based cadence: {report['legs_without_rhythm']}"
              "  (reconstructed face shorter than the acquire run-in)")
    print()
    for r in res["rows"]:
        print("  " + "  ".join(f"{k}={v}" for k, v in r.items()))
    print()
    if res["problems"]:
        print(f"PROBLEMS ({len(res['problems'])}):")
        for q in res["problems"]:
            print("   -", q)
    else:
        print("every scheduled touch has something to touch.")
    print()
    print("bodies NOT visited:",
          dict(Counter(x["reason"] for x in report["skipped"])))

    episodes = []
    if a.simulate:
        err = np.array([float(v) for v in a.err.split(",")])
        bodies = tuple(scene.plan.bodies)
        print(f"\n── simulated landmark visits (pose error {err.tolist()} m, "
              f"seed {a.seed}) ──")
        for st in steps:
            if st["kind"] != "ate_landmark":
                continue
            ep = simulate_episode(st, bodies, cfg, err, seed=a.seed)
            episodes.append(ep)
            print(f"\n  {ep['label']} ({ep['shape']}"
                  f"{', required' if ep['required'] else ''}): "
                  f"{ep['n_presses']} presses, {ep['n_contacts']} contacts, "
                  f"stop={ep['stop']}, |offset err| {ep['offset_error_m']} m")
            for r in ep["presses"]:
                print(f"     {r['try']}. off={r['offset_m']:+.2f} {r['kind']:6s} "
                      f"cand={r['n_candidates']:2d}/{r['n_scoring']:<2d} "
                      f"EIG={r['eig_nats']:6.3f} "
                      f"spread={r['nats_spread']:5.3f} -> {r['outcome']:7s} "
                      f"sigma {r['sigma_before'][0]:.3f},{r['sigma_before'][1]:.3f}"
                      f" -> {r['sigma_after'][0]:.3f},{r['sigma_after'][1]:.3f}"
                      f"  H {r['entropy_before']:.2f} -> {r['entropy_after']:.2f}")
        by_stop = Counter(e["stop"] for e in episodes)
        print(f"\n  stop reasons: {dict(by_stop)}")
        print(f"  presses per visit: "
              f"{[e['n_presses'] for e in episodes]} "
              f"(budget {cfg.landmark_press_budget})")

    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        doc = {"n_steps": len(steps), "report": report, "check": res,
               "episodes": episodes}
        (a.out / "demo_room_ate_plan.json").write_text(
            json.dumps(doc, indent=1, default=float))
        print(f"\nwrote {a.out / 'demo_room_ate_plan.json'}")
    return 1 if res["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
