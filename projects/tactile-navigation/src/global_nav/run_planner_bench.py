"""Run the planner benchmark: sweep, registration-gate sweep, figures, tables.

    python3 -m global_nav.run_planner_bench --out out/global_tacnav_planner

Deterministic given seeds.  ``--quick`` runs 6 seeds at 500 particles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .evalplanner import registration_gate_sweep, run_sweep, save
from .stressmaps import SCENARIOS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out/global_tacnav_planner"))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--particles", type=int, default=800)
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--skip-reg-sweep", action="store_true")
    args = ap.parse_args(argv)

    if args.quick:
        args.seeds, args.particles = 6, 500
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    keep = tuple((s, "belief_aware") for s in args.scenarios) + \
        tuple((s, "shortest_astar") for s in args.scenarios)
    results = run_sweep(seeds=range(args.seeds), n_particles=args.particles,
                        scenarios=args.scenarios, keep_traces_for=keep)

    if not args.skip_reg_sweep:
        print("\nregistration-gate sweep (fixed_interval, dense touch):")
        results["registration_gate"] = registration_gate_sweep(
            seeds=range(max(args.seeds // 2, 6)), n_particles=args.particles)

    save(results, out / "planner_bench.json")

    if not args.no_plots:
        from .plots_planner import make_all_figures
        for p in make_all_figures(out, results):
            print("wrote", p)

    print(f"\nwrote {out}/planner_bench.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
