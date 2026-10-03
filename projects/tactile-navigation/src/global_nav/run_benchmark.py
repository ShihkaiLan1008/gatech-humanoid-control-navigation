"""Run the benchmark and write traces, summaries and figures.

    python3 -m global_nav.run_benchmark --out out/global_tacnav

Everything is deterministic given ``--seed``: the traces regenerate byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .benchmark import (
    ABLATIONS,
    SCENARIOS,
    build_scenarios,
    registration_sweep,
    run_ablations,
    run_bad_association,
    run_scenario,
)
from .floorplan import benchmark_floorplan
from .plots import plot_ablation, plot_ambiguity, plot_map, plot_registration, plot_timeseries


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out/global_tacnav"))
    ap.add_argument("--particles", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="seeds for the multi-seed ablation table")
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--skip-sweeps", action="store_true",
                    help="skip the multi-seed ablation and the registration sweep")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = benchmark_floorplan()
    built = build_scenarios(args.particles, args.seed)
    results, summaries = {}, {}

    for name in args.scenarios:
        scn = built[name]
        if not args.no_plots:                      # particle clouds are only for the figures
            scn = type(scn)(**{**scn.__dict__, "particle_sample": 700})
        res = run_bad_association(scn) if name == "bad_association" else run_scenario(scn)
        results[name] = res
        summaries[name] = res["summary"]
        print(f"[{name:22s}] final_err={res['summary']['final_position_error_m']:.3f} m  "
              f"mean_err={res['summary']['mean_position_error_m']:.3f} m  "
              f"max_modes={res['summary']['max_modes']}  "
              f"min_ess={res['summary']['min_ess']:.0f}  "
              f"P(wrong turn)_max={res['summary']['max_wrong_turn_probability']:.3f}")
        (out / f"trace_{name}.json").write_text(json.dumps(_jsonable(res), indent=1))
        if not args.no_plots:
            plot_map(out, res, plan)
            plot_timeseries(out, res, plan)
            if name.startswith("ambiguous"):
                plot_ambiguity(out, res, plan)

    payload = {"summaries": summaries, "particles": args.particles, "seed": args.seed}
    if not args.skip_sweeps:
        print("\nrunning the multi-seed ablation ...")
        ab = run_ablations(tuple(args.seeds), args.particles)
        payload["ablations"] = ab
        print("running the registration-observability sweep ...")
        sweep = registration_sweep(seeds=tuple(args.seeds),
                                   n_particles=max(args.particles // 2, 400))
        payload["registration_sweep"] = sweep
        if not args.no_plots:
            plot_ablation(out, ab)
            plot_registration(out, sweep)
        print(f"\n{'scenario':20s} {'final err (m)':>18s} {'mean err (m)':>18s} {'reg err (m)':>12s}")
        for name in ABLATIONS:
            row = ab["table"][name]
            f, m = row["final_position_error_m"], row["mean_position_error_m"]
            print(f"{name:20s} {f['mean']:8.3f} ± {f['std']:6.3f} "
                  f"{m['mean']:8.3f} ± {m['std']:6.3f} "
                  f"{row['final_registration_error_m']['mean']:12.3f}")

    # merge rather than overwrite: a --skip-sweeps pass must not silently delete the
    # ablation table a previous full run wrote to the same directory
    path = out / "summary.json"
    if path.exists():
        try:
            payload = {**json.loads(path.read_text()), **payload}
        except json.JSONDecodeError:
            pass
    path.write_text(json.dumps(_jsonable(payload), indent=1))
    print(f"\nwrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
