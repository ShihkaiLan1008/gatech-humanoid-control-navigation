#!/usr/bin/env python3
"""Does our compiled room2 route still equal the DEPLOYED one?

    tactile_controller/.venv/bin/python scripts/check_deploy_mission.py

`configs/room2_deploy_mission.json` is vendored verbatim from
`wognl0402/locomanip_deploy` @ `tactile_navigation`
(`src/planner/tacnav_bringup/config/room2_deploy_mission.json`) -- the mission
that repo's room2 run actually executed.  Its header calls it "the real route
planner's compiled steps", i.e. the output of THIS planner at the commit that
was deployed.

So it is a fixture with an owner: if our compiler drifts from it, either the
drift is a bug or the mission needs re-exporting -- and the difference should
be a decision, not a surprise.  Two seconds, no simulator.

FRAMES.  The mission's map frame is ours rotated: ours = (-mission_y,
mission_x), so headings differ by +pi/2 and arc lengths `s` are identical.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from demo_map.executor import DemoExecCfg
from demo_map.scene_room2 import build_room2_scene, plan_steps_room2

MISSION = ROOT / "configs" / "room2_deploy_mission.json"
TOL = 1e-3


def to_ours(p):
    return np.array([-float(p[1]), float(p[0])])


def main() -> int:
    m = json.loads(MISSION.read_text())
    # the cfg `--plan deploy` builds -- the mission's own lane and cadence,
    # NOT the library defaults (those are the demo room's, whose scan bulges
    # into its lane and which therefore walks 0.49 with a step-in)
    cfg = DemoExecCfg(standoff_m=float(m["standoff_m"]),
                      press_spacing_m=float(m["spacing_m"]),
                      monotone_route=False)
    scene = build_room2_scene(False)
    steps = plan_steps_room2(cfg, False, scene=scene)
    bad = []

    def eq(name, got, want, tol=TOL):
        ok = abs(float(got) - float(want)) <= tol
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: ours {got} vs deployed {want}")
        if not ok:
            bad.append(name)

    print(f"mission {MISSION.relative_to(ROOT)}  schema {m['schema']}")
    # The deployed standoff is PHYSICAL -- that deployment's map and its walls
    # are the same walls.  Ours are offset by `true_delta_m` on purpose, so
    # what must equal 0.38 is the lane MINUS the offset.
    from demo_map.executor import lane_standoff
    for delta in (0.0, 0.12):
        eq(f"physical standoff at true_delta={delta}",
           round(lane_standoff(delta) - delta, 4), m["standoff_m"])
    eq("spacing_m", cfg.press_spacing_m, m["spacing_m"])

    ours_follow = {s["seg"].name: sorted(round(float(x), 3)
                                         for x in s["stations"])
                   for s in steps if s["kind"] == "follow"}
    for st in m["steps"]:
        if st["kind"] != "follow":
            continue
        want = sorted(round(float(x["s"]), 3) for x in st["stations"])
        name = next((k for k in ours_follow if k.startswith(st["wall"])), None)
        got = ours_follow.get(name, [])
        ok = got == want
        print(f"  [{'ok ' if ok else 'FAIL'}] stations {st['wall']}: {got}")
        if not ok:
            print(f"         deployed: {want}")
            bad.append(f"stations:{st['wall']}")

    ours_goto = {s.get("label"): s for s in steps if s["kind"] == "goto"}
    for st in m["steps"]:
        if st["kind"] != "goto":
            continue
        g = ours_goto.get(st["label"])
        if g is None:
            print(f"  [FAIL] goto {st['label']} missing"); bad.append(st["label"])
            continue
        d = float(np.linalg.norm(np.asarray(g["p"], float) - to_ours(st["p"])))
        dy = abs((float(g["yaw"]) - (float(st["yaw"]) + 0.5 * math.pi)
                  + math.pi) % (2 * math.pi) - math.pi)
        ok = d <= TOL and dy <= 1e-3
        print(f"  [{'ok ' if ok else 'FAIL'}] goto {st['label']}: "
              f"dp {d:.4f} m, dyaw {math.degrees(dy):.2f} deg")
        if not ok:
            bad.append(f"goto:{st['label']}")

    want_p = next(s for s in m["steps"] if s["kind"] == "pillar")
    got_p = next(s for s in steps if s["kind"] == "ate_pillar")
    d = float(np.linalg.norm(np.asarray(got_p["centre"], float)
                             - to_ours(want_p["centre"])))
    eq("pillar radius", got_p["radius"], want_p["radius"])
    print(f"  [{'ok ' if d <= TOL else 'FAIL'}] pillar centre: dp {d:.4f} m")
    if d > TOL:
        bad.append("pillar")

    # `--plan ate` is this repo's own compile and DOES add EIG visits; the
    # literal plan below must not.
    has_ate = any(s["kind"] in ("ate_corner", "ate_pillar") for s in steps)
    print(f"  [{'ok ' if has_ate else 'FAIL'}] --plan ate adds EIG visits "
          f"(the mission ships bare 'corner'/'pillar' waypoints)")
    if not has_ate:
        bad.append("eig")

    # ── and the LITERAL plan: every station and waypoint, round-tripped ────
    from demo_map.deploy_mission import compile_steps
    lit = compile_steps(m, steps)
    want_st, got_st = [], []
    for st in m["steps"]:
        if st["kind"] == "follow":
            want_st.append([round(float(x["s"]), 3) for x in st["stations"]])
        elif st["kind"] == "corner":
            want_st.append([round(float(x["s"]), 3) for x in st["pre"]])
            want_st.append([round(float(x["s"]), 3) for x in st["post"]])
        elif st["kind"] == "pillar":
            want_st.append(None)                      # one press, no arc list
    for st in lit:
        if st["kind"] == "follow":
            got_st.append(None if st["label"] == "goal_post"
                          else [round(float(v), 3) for v in st["stations"]])
    ok = got_st == want_st
    print(f"  [{'ok ' if ok else 'FAIL'}] literal plan stations {got_st}")
    if not ok:
        print(f"         deployed: {want_st}")
        bad.append("literal:stations")

    n_goto = sum(1 for st in lit if st["kind"] == "goto")
    want_goto = (sum(1 for st in m["steps"] if st["kind"] == "goto")
                 + sum(len(st["arc"]) for st in m["steps"]
                       if st["kind"] == "corner"))
    ok = n_goto == want_goto
    print(f"  [{'ok ' if ok else 'FAIL'}] literal plan waypoints: "
          f"{n_goto} (mission gotos + arc = {want_goto})")
    if not ok:
        bad.append("literal:gotos")

    for st in lit:                       # nothing this repo adds may appear
        assert st["kind"] in ("goto", "follow"), st["kind"]
    print("  [ok ] no EIG landmark visits, no ratchet in the literal plan")

    print("\nMISSION CONFORMANCE OK" if not bad
          else f"\nDRIFTED FROM THE DEPLOYED MISSION: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
