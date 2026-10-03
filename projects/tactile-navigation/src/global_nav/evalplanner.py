"""The benchmark sweep: every policy on every stressed floorplan, many seeds.

Reports exactly what the batch spec asks for -- navigation success and wrong-turn
rate, path length and excess distance, probes and landmark detours, final pose
and route-belief error, relocalization triggers and recovery, planning time --
plus the registration-gate sweep that closes the Batch-1 §7.1 item.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from .episode import EpisodeConfig, run_episode
from .policies import POLICIES, make_policy
from .stressmaps import SCENARIOS, build_scenario

__all__ = ["run_sweep", "registration_gate_sweep", "summarize_cell"]

METRICS = ("success", "wrong_turn", "cost_m", "excess_m", "n_probes",
           "n_station_visits", "final_pos_err_m", "n_reloc_triggered",
           "n_reloc_recovered", "n_suspect", "plan_time_mean_ms", "plan_time_max_ms")


def summarize_cell(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for key in METRICS:
        v = np.array([float(r[key]) for r in rows])
        out[key] = {"mean": float(v.mean()),
                    "sem": float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else 0.0}
    out["outcomes"] = {}
    for r in rows:
        out["outcomes"][r["outcome"]] = out["outcomes"].get(r["outcome"], 0) + 1
    return out


def run_sweep(seeds=range(20), n_particles: int = 800, scenarios=SCENARIOS,
              policies=tuple(POLICIES), keep_traces_for: tuple = (),
              verbose: bool = True) -> dict:
    results: dict = {"cells": {}, "episodes": {}, "n_particles": n_particles,
                     "seeds": list(map(int, seeds))}
    for scn_name in scenarios:
        sc = build_scenario(scn_name)
        for pol in policies:
            rows = []
            for seed in seeds:
                r = run_episode(sc, make_policy(pol),
                                EpisodeConfig(n_particles=n_particles, seed=int(seed)))
                if (scn_name, pol) in keep_traces_for or not keep_traces_for:
                    pass
                rows.append(r)
            cell = summarize_cell(rows)
            results["cells"][f"{scn_name}/{pol}"] = cell
            results["episodes"][f"{scn_name}/{pol}"] = [
                {k: v for k, v in r.items() if k != "trace"} for r in rows]
            if (scn_name, pol) in keep_traces_for:
                results.setdefault("traces", {})[f"{scn_name}/{pol}"] = [
                    r["trace"] for r in rows]
            if verbose:
                print(f"[{scn_name:16s}] {pol:16s} "
                      f"success={cell['success']['mean']:.2f} "
                      f"wrong={cell['wrong_turn']['mean']:.2f} "
                      f"excess={cell['excess_m']['mean']:5.1f} "
                      f"probes={cell['n_probes']['mean']:4.1f} "
                      f"reloc={cell['n_reloc_triggered']['mean']:.2f}"
                      f"/{cell['n_reloc_recovered']['mean']:.2f} "
                      f"plan={cell['plan_time_mean_ms']['mean']:.1f}ms", flush=True)
    return results


def registration_gate_sweep(magnitudes=(0.0, 0.02, 0.05, 0.10, 0.20, 0.30),
                            seeds=range(10), n_particles: int = 800,
                            verbose: bool = True) -> dict:
    """Does the gate avoid injecting noise below the floor and still recover
    above it?  Run under DENSE touch (fixed_interval on twin_junctions), because
    the gate is an estimator property and the estimator needs evidence; the
    belief-aware planner's minimal probing under-feeds it, which is reported.
    """
    base = build_scenario("twin_junctions")
    points = []
    for mag in magnitudes:
        raws, gateds, zeros = [], [], []
        for seed in seeds:
            reg = (0.8 * float(mag), -0.6 * float(mag))
            sc = replace(base, reg_true=reg)
            r = run_episode(sc, make_policy("fixed_interval"),
                            EpisodeConfig(n_particles=n_particles, seed=int(seed)))
            true = np.asarray(reg, float)
            raws.append(float(np.linalg.norm(np.asarray(r["registration_raw"]) - true)))
            gateds.append(float(np.linalg.norm(np.asarray(r["registration_gated"]) - true)))
            zeros.append(float(np.linalg.norm(true)))
        row = {"true_offset_m": float(mag),
               "raw_err_m": float(np.mean(raws)), "raw_sem": float(np.std(raws, ddof=1) / math.sqrt(len(raws))),
               "gated_err_m": float(np.mean(gateds)), "gated_sem": float(np.std(gateds, ddof=1) / math.sqrt(len(gateds))),
               "do_nothing_err_m": float(np.mean(zeros))}
        points.append(row)
        if verbose:
            print(f"  |r|={mag:.2f}  raw={row['raw_err_m']:.3f}  gated={row['gated_err_m']:.3f}  "
                  f"do_nothing={row['do_nothing_err_m']:.3f}", flush=True)
    return {"points": points, "seeds": list(map(int, seeds)), "n_particles": n_particles,
            "policy": "fixed_interval", "scenario": "twin_junctions"}


def save(results: dict, path: Path) -> None:
    def jsonable(o):
        if isinstance(o, dict):
            return {k: jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [jsonable(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(results), indent=1))
