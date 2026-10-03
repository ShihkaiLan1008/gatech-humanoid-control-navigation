#!/usr/bin/env python3
"""Run the punch demo over maps/seeds/controllers, or summarize .eval.json files."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from execution.controllers import CONTROLLERS
from execution.evaluation import write_report
from execution.demo_scenes import DEMO_ROOMS
from seed_batch import run_batch, save
from fleet_eval import recover


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="calls punch_demo.py; arguments after -- are forwarded")
    run.add_argument("--rooms", nargs="+", choices=[*DEMO_ROOMS, "all"], default=["room2"])
    run.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=["vforce"])
    run.add_argument("--seeds", nargs="+", type=int, default=[0])
    run.add_argument("--cpus-per-seed", type=int, default=0, help="0: split available CPUs equally")
    run.add_argument("--memory-per-seed-gb", type=float, default=0,
                     help="polled process-group RSS limit (not an address-space cap)")
    run.add_argument("--wall-timeout-s", type=float, default=0)
    run.add_argument("--layout", choices=("standard", "fleet"), default="standard")
    run.add_argument("--resume", action="store_true", help="preserve completed trials; run missing seeds")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("demo_args", nargs=argparse.REMAINDER)
    summarize = commands.add_parser("summarize", help="recompute metrics from canonical records")
    summarize.add_argument("inputs", nargs="+", type=Path)
    summarize.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = []
    if args.command == "summarize":
        for path in args.inputs:
            paths.extend(path.rglob("*.eval.json") if path.is_dir() else [path])
        if not paths:
            parser.error("no .eval.json records found")
    else:
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("seeds must be unique (one process per seed)")
        if min(args.cpus_per_seed, args.memory_per_seed_gb, args.wall_timeout_s) < 0:
            parser.error("resource limits cannot be negative")
        extra = args.demo_args[1:] if args.demo_args[:1] == ["--"] else args.demo_args
        reserved = {"--room", "--seed", "--controller", "--out", "--no-eval"}
        if any(a.partition("=")[0] in reserved for a in extra):
            parser.error("room, seed, controller, output and evaluation recording are managed by this runner")
        if not any(a.partition("=")[0] == "--policy" for a in extra):
            extra = ["--policy", "global", *extra]
        rooms = list(DEMO_ROOMS) if "all" in args.rooms else list(dict.fromkeys(args.rooms))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "tactile_controller/src"),
                                            env.get("PYTHONPATH", "")])
        env.setdefault("MUJOCO_GL", "egl")
        env["PYTHONUNBUFFERED"] = "1"
        args.out = args.out.resolve()
        args.out.mkdir(parents=True, exist_ok=True)
        lock = (args.out / ".queue.lock").open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("this evaluation output is already active")
        settings = dict(rooms=rooms, controllers=args.controllers, seeds=args.seeds,
                        demo_args=extra, layout=args.layout, cpus_per_seed=args.cpus_per_seed,
                        memory_per_seed_gb=args.memory_per_seed_gb, wall_timeout_s=args.wall_timeout_s)
        manifest = args.out / "queue.json"
        if manifest.exists() and (not args.resume or json.loads(manifest.read_text()) != settings):
            parser.error("existing output requires --resume with identical settings")
        save(manifest, settings)
        attempt_file = args.out / "attempts.json"
        attempts = json.loads(attempt_file.read_text()) if attempt_file.exists() else []
        if args.layout == "fleet" and len(set(args.controllers)) != 1:
            parser.error("fleet layout requires one controller per output")
        stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%m_%d_%H")
        stopped = False

        def finished(row):
            directory = Path(row["directory"])
            found = list(directory.glob("*.eval.json"))
            if not found:
                recover(directory, row["room"], row["seed"], row["controller"],
                        row["wall_clock_s"], row.get("stop_reason", "process_failed"))
                found = list(directory.glob("*.eval.json"))
            row["records"] = [str(p) for p in found]
            attempts.append(row)
            paths.extend(found)
            write_report(paths, args.out)
            save(attempt_file, attempts)

        for controller in dict.fromkeys(args.controllers):
            for room in rooms:
                jobs = []
                for seed in args.seeds:
                    if args.layout == "fleet":
                        existing = list((args.out / "runs").glob(f"{room}_original_{seed}_*"))
                        directory = existing[0] if existing else args.out / "runs" / f"{room}_original_{seed}_{stamp}"
                    else:
                        directory = args.out / controller / room / f"seed_{seed}"
                    directory.mkdir(parents=True, exist_ok=True)
                    if list(directory.glob("*.eval.json")):
                        if not args.resume:
                            parser.error(f"run already exists: {directory}; use a new --out")
                        paths.extend(directory.glob("*.eval.json"))
                        continue
                    if (directory / "command.json").exists():
                        parser.error(f"incomplete trial exists: {directory}; preserve it and use a fresh output")
                    cmd = [sys.executable, str(ROOT / "scripts/punch_demo.py"),
                           "--room", room, "--controller", controller, "--seed", str(seed),
                           "--out", str(directory), "--no-video", *extra]
                    jobs.append(dict(command=cmd, directory=directory,
                                     metadata=dict(room=room, controller=controller, seed=seed)))
                if jobs:
                    _, stopped = run_batch(jobs, cwd=ROOT, env=env,
                        cpus_per_seed=args.cpus_per_seed, memory_gb=args.memory_per_seed_gb,
                        timeout_s=args.wall_timeout_s, on_finish=finished)
                if stopped:
                    break
            if stopped:
                break
    report = write_report(paths, args.out)
    print(json.dumps(report["overall"], indent=2))
    print(f"Report: {args.out / 'summary.json'} and {args.out / 'runs.csv'}")
    return int(args.command == "run" and (stopped or any(r.get("returncode", 0) for r in attempts)))


if __name__ == "__main__":
    raise SystemExit(main())
