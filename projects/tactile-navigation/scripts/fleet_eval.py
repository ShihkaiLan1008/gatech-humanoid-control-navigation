#!/usr/bin/env python3
"""Sequential workstation evaluations with crash records and Dropbox uploads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from execution.demo_scenes import DEMO_ROOMS
from execution.evaluation import RunRecorder, compute_metrics, write_report


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def recover(directory, room, seed, controller, elapsed, reason):
    """Recover valid journal lines; an attempted startup failure remains a trial."""
    rec = RunRecorder(dict(room=room, seed=seed, controller=controller, policy="global"))
    journal = directory / "telemetry.jsonl"
    finished = False
    if journal.exists():
        with journal.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a hard exit may truncate the last buffered line
                kind, data = row["kind"], row["data"]
                if kind == "header":
                    rec.data = data
                elif kind in ("sample", "probe", "event"):
                    rec.data[{"sample": "samples", "probe": "probes", "event": "events"}[kind]].append(data)
                elif kind == "finish":
                    rec.data.update(data)
                    finished = True
    if not finished:
        rec.finish(rec.data["samples"][-1]["t"] if rec.data["samples"] else 0.,
                   outcome=reason, completed=False, wall_clock_s=elapsed)
    rec.data["metrics"] = compute_metrics(rec.data)
    rec.write(directory / "recovered.eval.json")


def upload(args, source, destination, log):
    with log.open("a") as stream:
        result = subprocess.run([args.rclone, "copy", str(source), destination,
                                 "--config", str(args.rclone_config), "--contimeout", "20s",
                                 "--timeout", "90s", "--retries", "3", "--low-level-retries", "3",
                                 "--transfers", "2", "--checkers", "4"],
                                stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode == 0


def report(out):
    return write_report(list((out / "runs").glob("*/*.eval.json")), out)


def source_provenance():
    """Describe the actual checkout, including uncommitted source changes."""
    def git(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args],
                                text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None

    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    hashes = {}
    for name in sorted(set((paths or "").split("\0")) - {""}):
        path = ROOT / name
        if path.is_file() and not path.is_symlink():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif path.is_symlink():
            hashes[name] = "symlink:" + str(path.readlink())
        else:
            hashes[name] = None
    record = dict(hostname=socket.gethostname(), source_directory=str(ROOT),
                  navigation_commit=git("rev-parse", "HEAD"),
                  navigation_status=git("status", "--porcelain=v1"),
                  source_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                  source_file_hashes=hashes, python_executable=sys.executable,
                  python_version=sys.version,
                  captured_utc=datetime.now(timezone.utc).isoformat())
    host_record = ROOT.parent / "host_provenance.json"
    if host_record.exists():
        # Preserve snapshot environment details without replacing live source identity.
        record["host_environment"] = json.loads(host_record.read_text())
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--room", choices=DEMO_ROOMS, required=True)
    p.add_argument("--controller", default="footstep", choices=("footstep", "vforce", "kinematic"))
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dropbox", required=True, help="remote campaign prefix; room names are added below it")
    p.add_argument("--rclone", default="rclone")
    p.add_argument("--rclone-config", type=Path, default=Path.home()/".config/tacnav-evaluation/rclone.conf")
    p.add_argument("--wall-timeout-s", type=float, default=14400.)
    p.add_argument("--upload-only", action="store_true", help="retry completed uploads; no evaluations")
    p.add_argument("--max-s", type=float, default=1800.)
    p.add_argument("--diagnose-moves", action="store_true",
                   help="footstep: record per-foot completion checks and primitive-end diagnostics")
    args = p.parse_args(argv)
    if args.max_s <= 0 or args.wall_timeout_s <= 0 or len(set(args.seeds)) != len(args.seeds):
        p.error("timeouts must be positive and seeds unique")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "runs").mkdir(exist_ok=True)
    # Prevent duplicate queues without relying on a tmux session name.
    lock = (args.out / ".queue.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error("this queue is already running")
    settings = dict(room=args.room, controller=args.controller, seeds=args.seeds,
                    max_s=args.max_s, bound_pos=.18, bound_yaw_deg=20.,
                    execution_pos_trigger=.18, execution_pos_policy="probe_then_continue",
                    probe_reach_profile="configs/probe_reach_ee_qpos_v1.json",
                    route_strategy="seed_refine", seed_fallback=True,
                    grid_resolution_m=.10, max_move_m=.375,
                    standoff_m=.30, controller_arrive_pos_tol_m=.03,
                    clearance_min_move_m=.08,
                    route_objective="free_distance_then_distance_and_rotation",
                    route_guard="belief", waypoint_progress="distance", thin="dense", uncertainty_scale=1.,
                    wall_timeout_s=args.wall_timeout_s, dropbox=args.dropbox,
                    diagnose_moves=args.diagnose_moves, start_heading="route",
                    naming_timezone="America/New_York")
    manifest = args.out / "queue.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != settings:
            p.error("queue settings differ from the existing output; use a fresh --out")
    else:
        save(manifest, settings)
    state_path = args.out / "status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"attempts": []}
    (args.out / "report").mkdir(exist_ok=True)
    stopped = False
    child = None
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    if not args.upload_only:
        for seed in args.seeds:
            if stopped:
                break
            previous = next((a for a in state["attempts"] if a["seed"] == seed), None)
            if previous:
                if previous["state"] == "running":
                    directory = args.out / "runs" / previous["folder"]
                    if not list(directory.glob("*.eval.json")):
                        recover(directory, args.room, seed, args.controller,
                                max(0., time.time()-previous["started_unix"]), "queue_interrupted")
                    previous.update(state="finished", upload_ok=False, recovered_after_restart=True)
                    save(state_path, state)
                continue
            stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%m_%d_%H")
            folder = f"{args.room}_original_{seed}_{stamp}"
            directory = args.out / "runs" / folder
            directory.mkdir(exist_ok=False)
            command = [sys.executable, "-u", str(ROOT / "scripts/punch_demo.py"),
                       "--room", args.room, "--controller", args.controller, "--seed", str(seed),
                       "--policy", "global", "--bound-pos", ".18", "--bound-yaw-deg", "20",
                       "--execution-pos-trigger", ".18", "--execution-pos-policy", "probe_then_continue",
                       "--probe-reach-profile", "configs/probe_reach_ee_qpos_v1.json",
                       "--start-heading", "route",
                       "--route-guard", "belief", "--thin", "dense", "--uncertainty-scale", "1",
                       "--standoff", ".30", "--override", "arrive_pos_tol_m=0.03",
                       "--max-s", str(args.max_s), "--no-video", "--out", str(directory),
                       "--planner-set", "route_strategy=seed_refine", "seed_fallback=true",
                       "grid_resolution_m=0.10", "max_move_m=0.375",
                       "route_objective=free_distance_then_distance_and_rotation"]
            if args.diagnose_moves:
                command += ["--override", "diagnose_moves=true"]
            save(directory / "command.json", command)
            attempt = dict(seed=seed, folder=folder, state="running", started_unix=time.time(), upload_ok=False)
            state["attempts"].append(attempt)
            save(state_path, state)
            save(directory / "provenance.json", source_provenance())
            env = dict(os.environ, TACNAV_EVAL_JOURNAL=str(directory / "telemetry.jsonl"))
            print(f"START {folder}", flush=True)
            started = time.monotonic()
            reason = "process_failed"
            with (directory / "run.log").open("w") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = child.wait(timeout=args.wall_timeout_s)
                except subprocess.TimeoutExpired:
                    reason = "wall_timeout"
                    os.killpg(child.pid, signal.SIGINT)
                    try:
                        code = child.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        code = child.wait()
            child = None
            if not list(directory.glob("*.eval.json")):
                recover(directory, args.room, seed, args.controller, time.monotonic()-started, reason)
            attempt.update(state="finished", returncode=code, wall_clock_s=time.monotonic()-started)
            save(directory / "attempt.json", attempt)
            save(state_path, state)
            report(args.out)
            attempt["upload_ok"] = upload(args, directory, args.dropbox.rstrip("/")+"/"+folder,
                                            args.out / "upload.log")
            save(state_path, state)
            for name in ("summary.json", "runs.csv", "queue.json", "status.json"):
                (args.out / "report" / name).write_bytes((args.out / name).read_bytes())
            upload(args, args.out / "report", args.dropbox.rstrip("/")+"/_reports/"+args.room,
                   args.out / "upload.log")
            record = json.loads(next(directory.glob("*.eval.json")).read_text())
            print(f"DONE {folder}: {record['outcome']}; upload={attempt['upload_ok']}", flush=True)
            if record["outcome"] == "process_failed" and not record["samples"]:
                print("Stopping queue after startup failure; fix the environment before resuming.", flush=True)
                break
    # Retry pending uploads without re-running trials or deleting local data.
    for attempt in state["attempts"]:
        if attempt["state"] == "finished" and not attempt["upload_ok"]:
            attempt["upload_ok"] = upload(args, args.out / "runs" / attempt["folder"],
                                            args.dropbox.rstrip("/")+"/"+attempt["folder"],
                                            args.out / "upload.log")
            save(state_path, state)
    report(args.out)
    state["queue_complete"] = len(state["attempts"]) == len(args.seeds) and all(
        a["state"] == "finished" for a in state["attempts"])
    state["uploads_complete"] = all(a["upload_ok"] for a in state["attempts"])
    save(state_path, state)
    for name in ("summary.json", "runs.csv", "queue.json", "status.json"):
        (args.out / "report" / name).write_bytes((args.out / name).read_bytes())
    report_ok = upload(args, args.out / "report", args.dropbox.rstrip("/")+"/_reports/"+args.room,
                       args.out / "upload.log")
    return 0 if state["queue_complete"] and state["uploads_complete"] and report_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
