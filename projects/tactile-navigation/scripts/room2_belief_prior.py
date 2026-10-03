#!/usr/bin/env python3
"""Section 3: what covariance does the Room 2 particle filter actually carry?

    scripts/room2_belief_prior.py --room room2 --out out/room2_belief_prior

Reads every run record under `--from` (default `out/`), buckets each logged
belief by route region, and reports the empirical distribution -- median,
25/75, 10/90, per-axis sigmas, and the eigen-structure of the position block.
Writes the pool the collector bootstraps `b_0` from.

Nothing here chooses a covariance.  The pool holds matrices the filter really
carried; `--jitter` only decides how much log-scale spread is added when one
is resampled, and `--jitter 0` is a pure bootstrap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", choices=("room2", "demo_room"), default="room2")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--from", dest="src", type=Path, default=ROOT / "out",
                    help="where to look for run records (default out/)")
    ap.add_argument("--no-hug-corner", action="store_true",
                    help="bucket beliefs against the planner's raw route "
                         "instead of the wall-following one the collector "
                         "uses")
    ap.add_argument("--jitter", type=float, default=None,
                    help="log-scale jitter sigma applied when a covariance is "
                         "resampled (default 0.15; 0 = pure bootstrap)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    from data_collect.belief_prior import (JITTER_LOG_SIGMA, CovariancePool,
                                           describe, error_covariances,
                                           harvest)
    from data_collect.rooms import build_scene, planned_centreline
    from data_collect.route import RouteRef

    out = a.out or (ROOT / "out" / f"{a.room}_belief_prior")
    out.mkdir(parents=True, exist_ok=True)
    scene = build_scene(a.room, a.reverse)
    route = RouteRef.from_polyline(
        planned_centreline(scene, hug_corner=not a.no_hug_corner),
        scene.plan.bodies)
    prefix = "room2" if a.room == "room2" else "demo_room"

    samples = harvest(a.src, route, scene_prefix=prefix)
    if not samples:
        print(f"[prior] no {prefix} belief logs under {a.src}", file=sys.stderr)
        return 1
    err = error_covariances(a.src, route, scene_prefix=prefix)
    rep = describe(samples, error_cov=err)
    pool = CovariancePool.from_samples(
        samples,
        jitter_log_sigma=(JITTER_LOG_SIGMA if a.jitter is None
                          else float(a.jitter)))

    rep["pool"] = pool.as_dict()
    rep["source_root"] = str(a.src)
    rep["room"] = a.room
    (out / "belief_prior.json").write_text(json.dumps(rep, indent=1))
    pool.to_npz(out / "belief_prior_pool.npz")

    print(f"room {a.room}: {rep['n_total']} logged beliefs "
          f"from {rep['by_source']}")
    for region in ("horizontal", "corner", "vertical"):
        r = rep.get(region)
        if not r:
            print(f"  {region:11s} -- none")
            continue
        sx, sy, st = r["sigma_x_m"], r["sigma_y_m"], r["sigma_theta_deg"]
        print(f"  {region:11s} n={r['n']:4d}  cross_terms="
              f"{str(r['cross_terms_available']):5s}")
        print(f"      sigma_x  [m]   p10 {sx['p10']:.3f}  p25 {sx['p25']:.3f}"
              f"  med {sx['median']:.3f}  p75 {sx['p75']:.3f}"
              f"  p90 {sx['p90']:.3f}")
        print(f"      sigma_y  [m]   p10 {sy['p10']:.3f}  p25 {sy['p25']:.3f}"
              f"  med {sy['median']:.3f}  p75 {sy['p75']:.3f}"
              f"  p90 {sy['p90']:.3f}")
        print(f"      sigma_th [deg] p10 {st['p10']:.2f}  p25 {st['p25']:.2f}"
              f"  med {st['median']:.2f}  p75 {st['p75']:.2f}"
              f"  p90 {st['p90']:.2f}")
        e = r["median_pos_eigenvalues"]
        print(f"      median pos eigenvalues {e}  major axis "
              f"{r['median_pos_major_axis_deg']:+.1f} deg")
        if "fallback_error_cov" in r:
            f = r["fallback_error_cov"]
            print(f"      [fallback] belief-vs-GT error sigma "
                  f"x {f['sigma_x_m']:.3f}  y {f['sigma_y_m']:.3f}  "
                  f"th {f['sigma_theta_deg']:.2f} deg")
    print(f"\npool: {pool.as_dict()}")
    print(f"wrote {out}/belief_prior.json, {out}/belief_prior_pool.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
