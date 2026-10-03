"""Figures and report for the scan-derived map, and the tape-vs-scan comparison.

    PYTHONPATH=src python3 scripts/build_scan_map.py

Reads docs/demo_maps/demo_map_scan_walls.json (produced once by
scripts/extract_scan_walls.py, which is the only step that needs trimesh and
the 14 MB mesh) and writes the route figure, the side-by-side comparison and a
MuJoCo XML of the scanned room.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from demo_map import build_demo_map, build_reference_route
from demo_map.geometry import LABELLED, clearance
from demo_map.plots import (_box_patches, _curve_patches, _frame, _regions,
                            _wall_patches, route_figure)
from demo_map.route import route_report
from demo_map.scan_map import build_scan_map, build_scan_route, load_scan_walls


def comparison_figure(tape, scan, tape_r, scan_r, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(15.5, 13.5))
    for ax, (m, r, title) in zip(axes, [
        (tape, tape_r, "TAPE reconstruction -- what the drawing says "
                       "(4 of the upper room's 6 numbers were chosen, not measured)"),
        (scan, scan_r, "SCAN-derived map -- wall faces extracted from the iPhone "
                       "mesh, registered into the same frame")]):
        _wall_patches(ax, m, ids=False)
        _curve_patches(ax, m)
        _box_patches(ax, m)
        _regions(ax, m)
        p = r.polyline(0.02)
        ax.plot(p[:, 0], p[:, 1], color="#d94436", lw=2.4, zorder=7)
        _frame(ax, m)
        ax.set_title(title, fontsize=10)
    lim = [min(a.get_xlim()[0] for a in axes), max(a.get_xlim()[1] for a in axes)]
    ylim = [min(a.get_ylim()[0] for a in axes), max(a.get_ylim()[1] for a in axes)]
    for ax in axes:
        ax.set_xlim(*lim)
        ax.set_ylim(*ylim)
    fig.suptitle("The same room, two sources, one frame", fontsize=12, y=0.94)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _self_framed(doc, args) -> None:
    """A scan of somewhere the tape never measured: no comparison to draw, so
    report the map on its own terms."""
    from demo_map.mujoco_scene import demo_map_xml
    from demo_map.scan_map import build_scan_map, build_scan_traverse

    m = build_scan_map(args.walls)
    r = build_scan_traverse(m)
    reg = doc["registration"]
    print(f"{doc.get('name', args.tag)}: frame derived from the scan itself "
          f"(theta {reg['theta_deg']:+.3f} deg, origin at the corner where its "
          "two longest walls meet)")
    pts = np.array([p for w in m.walls for p in (w.p0, w.p1)], float)
    print(f"  {len(m.walls)} walls, {len(m.curves)} curved features, "
          f"{len(m.boxes)} boxes + {len(m.pillars)} pillars + "
          f"{len(m.hulls)} hull prisms, {len(m.openings)} openings over "
          f"{np.ptp(pts[:, 0]):.2f} x {np.ptp(pts[:, 1]):.2f} m")
    print(f"  band coverage claimed by walls: "
          f"{100 * doc['coverage']['claimed_by_walls']:.1f}%")
    print(f"  start {tuple(round(v, 2) for v in m.start.center)} "
          f"clear {float(clearance([m.start.center], m)[0]):.3f} m   "
          f"goal {tuple(round(v, 2) for v in m.goal.center)} "
          f"clear {float(clearance([m.goal.center], m)[0]):.3f} m")
    if r is not None:
        rep = route_report(m, r)
        print(f"  traverse {rep['total_length_m']:.2f} m in "
              f"{rep['n_segments']} segments, min clearance "
              f"{rep['min_clearance_m']:.3f} m")
        route_figure(m, r, out=args.out / f"{args.tag}_route.png")
    (args.out / f"{args.tag}_room.xml").write_text(
        demo_map_xml(m, visual_mesh=args.visual))
    print(f"\nwrote {args.out}/{args.tag}_route.png, "
          f"{args.out}/{args.tag}_room.xml")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/demo_map"))
    ap.add_argument("--walls", type=Path, default=None,
                    help="scan walls JSON; default is the demo room's")
    ap.add_argument("--tag", default="scan", help="output filename suffix")
    ap.add_argument("--visual", type=Path, default=None,
                    help="registered decimated scan OBJ (from "
                         "extract_scan_walls --visual) to embed as the "
                         "scene's look-only layer")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    doc = load_scan_walls(args.walls) if args.walls else load_scan_walls()
    if doc.get("frame", "tape") != "tape":
        return _self_framed(doc, args)

    doc = load_scan_walls()
    tape, scan = build_demo_map(), build_scan_map()
    tape_r, scan_r = build_reference_route(tape), build_scan_route(scan)

    reg = doc["registration"]
    print(f"registration: theta={reg['theta_deg']:.3f} deg, "
          f"t=({reg['tx']:.4f}, {reg['ty']:.4f}); residual median "
          f"{reg['residual']['median_mm']:.1f} mm, p90 "
          f"{reg['residual']['p90_mm']:.1f} mm over {reg['residual']['n']} points")

    a = scan.key_points
    rows = [
        ("lower corridor clear width", LABELLED["D_LOWER_W"],
         a["long_wall_y"] - a["bottom_y"]),
        ("branch corridor clear width", LABELLED["D_CORRIDOR_W"],
         a["branch_e_x"] - a["branch_w_x"]),
        ("long wall: origin -> branch mouth", 10.3505, a["branch_w_x"]),
        ("bottom wall -> branch corridor end", LABELLED["D_BOTTOM_TO_UPPER"],
         a["branch_end_y"] - a["bottom_y"]),
    ]
    print(f"\n{'feature':38s} {'tape':>9s} {'scan':>9s} {'diff':>10s}")
    print("-" * 70)
    for name, tp, sc in rows:
        print(f"{name:38s} {tp:9.4f} {sc:9.4f} {1000*(sc-tp):+8.0f} mm")

    for label, m, r in (("tape", tape, tape_r), ("scan", scan, scan_r)):
        rep = route_report(m, r)
        print(f"\n{label} map: {len(m.walls)} walls, {len(m.curves)} curves, "
              f"route {rep['total_length_m']:.2f} m in {rep['n_segments']} segments, "
              f"min clearance {rep['min_clear_m'] if 'min_clear_m' in rep else rep['min_clearance_m']:.3f} m, "
              f"inside free space {rep['all_inside_free_space']}")
        print(f"  start {tuple(round(v, 3) for v in m.start.center)} "
              f"clear {float(clearance([m.start.center], m)[0]):.3f} m   "
              f"goal {tuple(round(v, 3) for v in m.goal.center)} "
              f"clear {float(clearance([m.goal.center], m)[0]):.3f} m")

    route_figure(scan, scan_r, out=args.out / "scan_reference_route.png")
    comparison_figure(tape, scan, tape_r, scan_r,
                      args.out / "tape_vs_scan.png")
    from demo_map.mujoco_scene import demo_map_xml
    (args.out / "scan_room.xml").write_text(
        demo_map_xml(scan, visual_mesh=args.visual))
    print(f"\nwrote {args.out}/scan_reference_route.png, {args.out}/tape_vs_scan.png, "
          f"{args.out}/scan_room.xml")


if __name__ == "__main__":
    main()
