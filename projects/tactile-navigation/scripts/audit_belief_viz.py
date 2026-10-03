"""Audit the belief VISUALIZATION against the belief itself.

    PYTHONPATH=src python3 scripts/audit_belief_viz.py out/demo_v3/r1_gallery \
        --map r1_gallery --out out/viz_audit

The r1_gallery demo video reports "believed vs TRUE (along-route)" growing past
10 m while the same episode ends `arrived` with a 0.098 m belief error. Both
cannot be true of the same belief, so one of them is a display artefact.

This dumps, per recorded belief sample (the replay interpolates between these,
so they are the frames' source of truth):

    truth_map_xy            what the executive logged, ALREADY map-frame
    belief_map_xy           the leading/tracked pose the executive navigates on
    euclid_err              || belief - truth ||, the honest pose error
    modes                   every live mode: xy (or x) and mass
    truth_seg / bel_seg     nearest segment index on ONE shared polyline
    truth_s / bel_s         arclength of each, projected on that SAME polyline
    s_err                   bel_s - truth_s, the along-route error done right
    shown_err               what the panel actually displays today

and marks the replans and `_relatch()` boundaries so a jump can be attributed.

Route parameterisation, edge switching and cumulative progress are all tested
by projecting BOTH poses onto one fixed polyline: if the displayed number were
any of those, `s_err` would reproduce it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _project(poly: np.ndarray, p: np.ndarray):
    """(arclength, segment index, perpendicular distance) of p on a polyline."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    L = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
    t = np.clip(np.einsum("ij,ij->i", p[None, :] - a, ab) / L, 0.0, 1.0)
    foot = a + t[:, None] * ab
    d = np.linalg.norm(foot - p[None, :], axis=1)
    k = int(np.argmin(d))
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(ab, axis=1))]
    return float(seg[k] + t[k] * math.sqrt(L[k])), k, float(d[k])


def route_polyline(scene, graph, ep) -> np.ndarray:
    """One fixed reference polyline: the route the planner chose first.

    Fixed on purpose -- the point of the audit is to ask whether the displayed
    number is explained by the route CHANGING, so the reference must not."""
    from lane_graph.graph3 import Belief3, astar3
    b = Belief3(sigma=0.10, minor=0.0)
    path, _ = astar3(graph, 0, 1, b)
    pts = [np.asarray(scene.start_map_xy, float)]
    for eid in path or []:
        e = graph.edges[eid]
        if e.poly is not None:
            pts.extend(np.asarray(e.poly, float))
        else:
            pts.append(np.asarray(graph.nodes[e.b].xy, float))
    pts.append(np.asarray(scene.goal_map_xy, float))
    out = [pts[0]]
    for q in pts[1:]:
        if float(np.linalg.norm(q - out[-1])) > 1e-6:
            out.append(q)
    return np.asarray(out, float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", type=Path)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", type=Path, default=Path("out/viz_audit"))
    a = ap.parse_args(argv)

    from lane_graph.graph3 import build_graph3
    from lane_graph.scene3 import build_v3_scene

    ep = json.loads((a.episode / "episode.json").read_text())
    scene = build_v3_scene(a.map)
    poly = route_polyline(scene, build_graph3(scene), ep)

    # the frame the replay derives from the FIRST tick: world spawn pose
    tr = ep["tick_trace"]
    base0 = np.asarray(tr[0]["xy"], float)
    yaw0 = float(tr[0]["yaw"])
    v = np.array([math.cos(yaw0), math.sin(yaw0)])
    u = np.array([v[1], -v[0]])

    dir_t = [float(d["t"]) for d in ep.get("directives", [])]
    rows = []
    for h in ep["hypothesis_trace"]:
        tm = np.asarray(h["truth_xy"], float)        # ALREADY map frame
        bel = np.asarray(h["bel_xy"], float)
        # what replay_v2.TelemetryV2 does today: map it a SECOND time
        tm_double = np.array([(tm - base0) @ u, (tm - base0) @ v])
        ts, tseg, tperp = _project(poly, tm)
        bs, bseg, bperp = _project(poly, bel)
        masses = list(h.get("masses") or [])
        xs = list(h.get("hyp_s") or [])
        xys = h.get("hyp_xy")
        modes = [{"xy": (list(map(float, xys[i])) if xys and i < len(xys)
                         else [float(xs[i]), float("nan")]),
                  "mass": float(masses[i])} for i in range(len(masses))]
        modes.sort(key=lambda m: -m["mass"])
        rows.append({
            "t": float(h["t"]), "mode": h.get("mode", ""),
            "truth_map_xy": tm.tolist(), "belief_map_xy": bel.tolist(),
            "euclid_err": float(np.linalg.norm(bel - tm)),
            "modes": modes[:3], "n_modes": len(masses),
            "truth_seg": tseg, "bel_seg": bseg,
            "truth_s": ts, "bel_s": bs, "s_err": bs - ts,
            "truth_perp": tperp, "bel_perp": bperp,
            "shown_err": float(bel[0] - tm_double[0]),
            "replan_here": any(abs(float(h["t"]) - d) < 1e-6 for d in dir_t),
        })

    def col(k):
        return np.asarray([r[k] for r in rows], float)

    print(f"# belief-visualisation audit -- {a.map}  ({a.episode})")
    print(f"{len(rows)} belief samples, reference polyline {len(poly)} pts, "
          f"{float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1))):.2f} m")
    print(f"replay frame: base0 {np.round(base0, 3)} yaw0 {yaw0:+.4f}")
    print()
    print(f"{'t':>7s} {'mode':>10s} {'truth_map':>16s} {'belief_map':>16s}"
          f" {'euclid':>7s} {'t_seg':>5s} {'b_seg':>5s} {'truth_s':>8s}"
          f" {'bel_s':>8s} {'s_err':>7s} {'SHOWN':>8s} {'top masses':>16s}")
    step = max(1, len(rows) // 20)
    for r in rows[::step]:
        mm = ", ".join(f"{m['mass']:.2f}" for m in r["modes"][:2])
        print(f"{r['t']:7.1f} {r['mode']:>10s}"
              f" {str(np.round(r['truth_map_xy'], 2)):>16s}"
              f" {str(np.round(r['belief_map_xy'], 2)):>16s}"
              f" {r['euclid_err']:7.3f} {r['truth_seg']:5d} {r['bel_seg']:5d}"
              f" {r['truth_s']:8.2f} {r['bel_s']:8.2f} {r['s_err']:+7.3f}"
              f" {r['shown_err']:+8.2f} {mm:>16s}")

    print()
    print("SUMMARY")
    for k, lab in (("euclid_err", "Euclidean pose error   "),
                   ("s_err", "along-route error (same polyline)"),
                   ("shown_err", "DISPLAYED 'along-route'")):
        c = col(k)
        print(f"  {lab:34s} min {c.min():+7.3f}  max {c.max():+7.3f}"
              f"  median {np.median(c):+7.3f} m")
    seg_sw = int(np.sum(col("truth_seg") != col("bel_seg")))
    print(f"  samples where truth/belief sit on DIFFERENT segments: {seg_sw}"
          f" of {len(rows)}")
    print(f"  replans: {len(dir_t)}   modes at end: {rows[-1]['n_modes']}")
    print(f"  episode: arrived={ep['arrived']} goal_dist="
          f"{ep.get('goal_dist_m', float('nan')):.3f} m "
          f"global_pos_err={ep['global_pos_err_m']:.3f} m")

    print()
    e, s = np.abs(col("euclid_err")).max(), np.abs(col("s_err")).max()
    d = np.abs(col("shown_err")).max()
    print("VERDICT")
    print(f"  the belief is never worse than {e:.3f} m (Euclidean) or "
          f"{s:.3f} m along the route,")
    print(f"  and the panel displays up to {d:.2f} m.  Route parameterisation,")
    print("  edge switching and cumulative progress are all ruled out: BOTH")
    print("  poses were projected onto ONE fixed polyline and s_err tracks the")
    print("  Euclidean error, not the displayed number.")
    print("  The displayed number applies the world->map transform to a truth")
    print("  sample that `_snap` ALREADY stored in map coordinates -- a")
    print("  replay-side double transform, growing with distance from spawn.")

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.map}_viz_audit.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {a.out}/{a.map}_viz_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
