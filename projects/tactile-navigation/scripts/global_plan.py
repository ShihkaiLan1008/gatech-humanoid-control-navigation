#!/usr/bin/env python3
"""Run the global search on registered maps without starting a controller.

    tactile_controller/.venv/bin/python scripts/global_plan.py \
      --rooms ref1_bend ref2_hall ref3_zsuite ref4_room \
      --bound-pos .18 --bound-yaw-deg 20 \
      --planner-set route_objective=distance_and_rotation \
        grid_resolution_m=.15 max_move_m=.375 --out out/global_plans
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from execution.demo_scenes import DEMO_ROOMS, REFERENCE_MAPS, load_demo_scene
from execution.global_route import filter_consistent_config, plan_global, plot_route, route_record
from execution.punch_belief import LocalizerConfig
from lane_graph.constrained import position_sigma_max
from lane_graph.tradeoff import feasible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rooms", nargs="+", choices=[*DEMO_ROOMS, "all"], default=list(REFERENCE_MAPS))
    parser.add_argument("--bound-pos", type=float, default=.18)
    parser.add_argument("--bound-yaw-deg", type=float, default=20.)
    parser.add_argument("--standoff", type=float, default=.30,
                        help="nominal robot-base to contact-surface distance [m] (default: 0.30)")
    parser.add_argument("--thin", choices=("dense", "balanced", "time"), default="dense")
    parser.add_argument("--start-heading", choices=("route", "scene"), default="route",
                        help="use the execution spawn-heading policy (default: route)")
    parser.add_argument("--planner-set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="UncertaintyConfig overrides; defaults: route_strategy=seed_refine "
                             "seed_fallback=true; route_strategy=belief_astar selects the original search")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rooms = list(DEMO_ROOMS) if "all" in args.rooms else list(dict.fromkeys(args.rooms))
    args.out.mkdir(parents=True, exist_ok=True)
    loc = LocalizerConfig()
    cfg = filter_consistent_config(loc, [f"contact_standoff_m={args.standoff}",
                                         "position_constraint=cross_track", *args.planner_set])
    cfg = cfg.merged(position_threshold_m=args.bound_pos,
                     yaw_threshold_rad=math.radians(args.bound_yaw_deg))
    config = dict(planner=asdict(cfg), localizer=asdict(loc), thin=args.thin,
                  start_heading=args.start_heading)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    summary = []
    for room in rooms:
        setup = load_demo_scene(room, args.standoff)
        directory = args.out / room
        directory.mkdir(exist_ok=True)
        events = []

        def log(event, **values):
            row = dict(event=event, **values)
            events.append(row)
            print(json.dumps(dict(room=room, **row)), flush=True)

        print(f"Planning {room}: {args.bound_pos:g} m / {args.bound_yaw_deg:g} deg, "
              f"{cfg.route_strategy}, {cfg.route_objective}", flush=True)
        started = time.monotonic()
        route = plan_global(setup.scene, setup.face_yaw_map, cfg, args.bound_pos,
                            math.radians(args.bound_yaw_deg),
                            prior_sigma_xy_m=loc.prior_sigma_xy_m,
                            prior_sigma_theta_rad=loc.prior_sigma_theta_rad,
                            thin=args.thin, log=log, align_start=args.start_heading == "route")
        elapsed = time.monotonic() - started
        plan_event = next(e for e in events if e["event"] == "GLOBAL_PLAN")
        checks = {}
        if route is not None:
            xy = np.array([route.grid.cell_xy(c) for c in route.plan.path])
            goal_margin = float(np.linalg.norm(xy[-1] - route.grid.goal_xy)
                                + route.cfg.goal_sigma_multiplier * position_sigma_max(route.trace[-1]["cov"]))
            checks = dict(within_bounds=feasible(route.metrics, args.bound_pos, args.bound_yaw_deg),
                          segments_clear=all(route.grid.segment_clear(a, b) for a, b in zip(xy[:-1], xy[1:])),
                          goal_confidence_fits=goal_margin <= route.grid.goal_radius + 1e-9,
                          goal_confidence_margin_m=goal_margin)
            plot_route(route, setup.scene, directory / "route.png")
        valid = route is not None and all(checks[k] for k in ("within_bounds", "segments_clear", "goal_confidence_fits"))
        record = dict(kind="global_planner_only", room=room, setup=setup.describe(),
                      config=config, events=events, elapsed_s=elapsed, checks=checks,
                      route=None if route is None else route_record(route))
        (directory / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        metrics = {} if route is None else route.metrics
        summary.append(dict(room=room, feasible=valid, reason=plan_event["reason"],
                            requested_strategy=cfg.route_strategy, strategy=plan_event["strategy"],
                            position_bound_m=args.bound_pos, yaw_bound_deg=args.bound_yaw_deg,
                            distance_m=metrics.get("distance_m"), contacts=metrics.get("probe_count"),
                            peak_position_sigma_m=metrics.get("peak_position_sigma_m"),
                            peak_yaw_sigma_deg=metrics.get("peak_yaw_sigma_deg"),
                            explored_states=plan_event["explored"], elapsed_s=elapsed))
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (args.out / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print(json.dumps(summary[-1]), flush=True)
    print(f"Results: {args.out / 'summary.csv'}", flush=True)
    return 0 if all(row["feasible"] for row in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
