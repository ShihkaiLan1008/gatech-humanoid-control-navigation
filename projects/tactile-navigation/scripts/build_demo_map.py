"""Rebuild every demo-map artefact and print the Phase 2 validation table.

    PYTHONPATH=src python3 scripts/build_demo_map.py [--out out/demo_map]

Writes the metric spec, the geometry validation figure, the route figure and a
MuJoCo XML of the room; prints the labelled-vs-reconstructed dimension table and
the per-segment route report.  Everything here is regenerated from
src/demo_map/geometry.py, so this script is also the drift check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from demo_map import build_demo_map, build_reference_route
from demo_map.export import SPEC_PATH, write_yaml
from demo_map.geometry import clearance, remeasure
from demo_map.plots import geometry_figure, route_figure
from demo_map.route import route_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/demo_map"))
    ap.add_argument("--spec", type=Path, default=SPEC_PATH)
    ap.add_argument("--sketch", type=Path,
                    default=Path("docs/demo_maps/measured_map.jpg"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    m = build_demo_map()
    r = build_reference_route(m)

    write_yaml(args.spec, m, r)
    geometry_figure(m, sketch=args.sketch, out=args.out / "geometry_validation.png")
    route_figure(m, r, out=args.out / "reference_route.png")

    from demo_map.mujoco_scene import demo_map_xml
    (args.out / "demo_room.xml").write_text(demo_map_xml(m))

    rows = remeasure(m)
    w = max(len(d["feature"]) for d in rows)
    print(f"\n{'feature':{w}s} {'labelled':>10s} {'recon':>10s} {'abs err':>10s}")
    print("-" * (w + 33))
    for d in rows:
        e = abs(d["labelled"] - d["recon"])
        print(f"{d['feature']:{w}s} {d['labelled']:10.4f} {d['recon']:10.4f} {e:10.2e}")
    print(f"\nworst absolute error: {max(abs(d['labelled'] - d['recon']) for d in rows):.2e} m")

    rep = route_report(m, r)
    print(f"\nroute: {rep['n_segments']} segments, {rep['total_length_m']:.2f} m, "
          f"min clearance {rep['min_clearance_m']:.3f} m, "
          f"continuity gap {rep['max_continuity_gap_m']:.1e} m, "
          f"inside free space: {rep['all_inside_free_space']}")
    print(f"{'id':4s} {'kind':14s} {'follows':9s} {'hand':6s} {'len':>7s} {'min clr':>8s}")
    for s in rep["segments"]:
        print(f"{s['sid']:4s} {s['kind']:14s} {s['follows']:9s} {s['hand']:6s} "
              f"{s['length_m']:7.3f} {s['min_clear_m']:8.3f}")

    print(f"\nstart clearance {float(clearance([m.start.center], m)[0]):.3f} m   "
          f"goal clearance {float(clearance([m.goal.center], m)[0]):.3f} m")
    print(f"\nwrote {args.spec}, {args.out}/geometry_validation.png, "
          f"{args.out}/reference_route.png, {args.out}/demo_room.xml")


if __name__ == "__main__":
    main()
