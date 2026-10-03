#!/usr/bin/env python3
"""Compile both rooms' routes and check they are sane.  ~2 s, no simulator.

RUN IT WITH THE VENV INTERPRETER -- that is the whole point:

    tactile_controller/.venv/bin/python scripts/smoke_plan.py

The pytest suite runs on the system python (numpy 1.x); the demos run on
tactile_controller/.venv (numpy 2.4).  A `float()` on a size-1 array is a
deprecation on one and a hard TypeError on the other, so a planner change can
pass every test and still raise at startup for the operator.  It did:
`leg_is_free_space` called the VECTORISED `distance_to_aabb` a point at a time.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from demo_map.executor import DemoExecCfg
from demo_map.scene_demo import build_demo_scene, demo_route, plan_steps_demo_room
from demo_map.scene_room2 import build_room2_scene, plan_steps_room2


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    return bool(ok)


def report(label, scene, steps):
    good = True
    gotos = [s for s in steps if s["kind"] == "goto"]
    lms = [s for s in steps if s["kind"] in ("ate_landmark", "ate_pillar")]
    follows = [s for s in steps if s["kind"] == "follow"]
    stations = sum(len(s["stations"]) for s in follows)
    kinds = sorted({b.name.split("_")[0] for b in scene.plan.bodies})
    print(f"\n{label}: {len(steps)} steps, {stations} rhythm presses, "
          f"{len(lms)} landmark visits, bodies {kinds}")

    good &= check("every goto has a gait",
                  all(s.get("gait") in ("crab", "walk") for s in gotos),
                  str([(s.get("label"), s.get("gait")) for s in gotos]))
    good &= check("landmarks are corners and cylinders only",
                  all(s.get("shape", "cylinder") == "cylinder" for s in lms))
    good &= check("no box bodies in the scene", "prot" not in kinds)
    cad = DemoExecCfg().press_spacing_m
    gaps = [g for s in follows if len(s["stations"]) > 1
            for g in np.diff(sorted(s["stations"]))]
    good &= check(f"rhythm cadence is {cad:.2f} m",
                  gaps and abs(np.median(gaps) - cad) < 1e-6,
                  f"median {np.median(gaps):.3f}" if gaps else "no stations")
    # Tight pairs are allowed ONLY where the deployed mission has them: room2's
    # north:east leg ships 1.8/1.95/2.0, so two presses sit 0.05 m apart.  That
    # is the route that was deployed, and `check_deploy_mission.py` is what
    # pins it; here we only refuse pairs TIGHTER than the deployment's.
    good &= check("no station tighter than the deployed minimum (0.05 m)",
                  not gaps or min(gaps) >= 0.05 - 1e-9,
                  f"min gap {min(gaps):.3f}" if gaps else "")
    return good


def main() -> int:
    print(f"numpy {np.__version__}  python {sys.version.split()[0]}")
    ok = True
    scene = build_demo_scene()
    ok &= report("demo_room", scene,
                 plan_steps_demo_room(DemoExecCfg(), scene, demo_route())[0])
    for rev in (False, True):
        sc = build_room2_scene(rev)
        ok &= report(f"room2 (reverse={rev})", sc,
                     plan_steps_room2(DemoExecCfg(), rev, scene=sc))
    print("\nSMOKE OK" if ok else "\nSMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
