#!/usr/bin/env python3
"""One map allocation: concurrent seeds, sequential replay, verified Dropbox copy."""
from pathlib import Path
import argparse
import json
import os
import signal
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from seed_batch import save
from execution.evaluation import write_report


def copy_checked(source, destination, config, log):
    flags = ["--contimeout", "20s", "--timeout", "90s", "--retries", "3",
             "--low-level-retries", "3", "--exclude", ".cache/**",
             "--exclude", "*.partial.mp4", "--exclude", ".queue.lock"]
    with log.open("a") as stream:
        for operation in ("copy", "check"):
            command = [config["rclone"], operation, str(source), destination, *flags]
            command += ["--transfers", "2"] if operation == "copy" else ["--one-way"]
            try:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
            except subprocess.TimeoutExpired:
                return False
            if result.returncode:
                return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", required=True)
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    campaign = json.loads((BASE / "campaign.json").read_text())
    config = json.loads((BASE / "host.json").read_text())
    out = BASE / ("smoke" if args.smoke else "evaluation") / args.room
    out.mkdir(parents=True, exist_ok=True)
    remote = campaign["dropbox"] + ("/_smoke" if args.smoke else "/evaluation")
    child = None
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, stop)

    def launch(command, log):
        nonlocal child
        with log.open("w") as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            code = child.wait()
            child = None
        return code

    def sync():
        runs = sorted((out / "runs").glob("*"))
        records = list((out / "runs").glob("*/*.eval.json"))
        write_report(records, out)
        rows = []
        for directory in runs:
            if not directory.is_dir():
                continue
            states = list(directory.glob("states_*.npz"))
            row = dict(folder=directory.name, evaluation=bool(list(directory.glob("*.eval.json"))),
                       video=bool(states) and all((directory / (p.stem.removeprefix("states_") + ".mp4")).exists()
                                                  for p in states))
            row["upload_verified"] = copy_checked(directory, remote + "/" + directory.name,
                                                    config, out / "upload.log")
            rows.append(row)
        state = dict(room=args.room, expected_seeds=campaign["seeds"], trials=rows,
                     evaluations_complete=len(records) == len(campaign["seeds"]),
                     videos_complete=len(rows) == len(campaign["seeds"]) and all(r["video"] for r in rows),
                     uploads_verified=len(rows) == len(campaign["seeds"]) and all(r["upload_verified"] for r in rows))
        save(out / "status.json", state)
        report = out / "report"
        report.mkdir(exist_ok=True)
        for source in [*out.glob("*.json"), *out.glob("*.csv"), *out.glob("*.log"),
                       BASE / "campaign.json", BASE / "host.json", BASE / "SHA256SUMS",
                       BASE / "source_provenance.json"]:
            if source.is_file():
                (report / source.name).write_bytes(source.read_bytes())
        for source in (BASE / "slurm-logs").glob(f"*{args.room}*.log"):
            (report / source.name).write_bytes(source.read_bytes())
        ok = copy_checked(report, remote + "/_reports/" + args.room, config, out / "upload.log")
        print(json.dumps(state), flush=True)
        return ok and state["uploads_verified"]

    if args.sync_only:
        return 0 if sync() else 1
    # Hardware is initialized only on the allocated compute node.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("allocated GPU is unavailable")
    torch.ones(1, device="cuda").sum().item()
    from importlib import metadata, util
    save(out / "runtime.json", dict(hostname=socket.gethostname(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
         gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda, python=sys.version,
         versions={n:metadata.version(n) for n in campaign["runtime_packages"]},
         source_paths={n:list(util.find_spec(n).submodule_search_locations or []) for n in ("execution", "rl", "assets", "curobo")}))
    extra = list(campaign["demo_args"])
    if args.smoke:
        extra[extra.index("--max-s") + 1] = "8"
    command = [sys.executable, "-u", str(ROOT / "scripts/punch_eval.py"), "run",
               "--controllers", campaign["controller"], "--rooms", args.room,
               "--seeds", *map(str, campaign["seeds"]), "--out", str(out), "--layout", "fleet",
               "--cpus-per-seed", str(campaign["cpus_per_seed"]),
               "--memory-per-seed-gb", str(campaign["memory_per_seed_gb"]),
               "--wall-timeout-s", "14400", "--", *extra]
    save(out / "batch_command.json", command)
    code = 1
    try:
        code = launch(command, out / "batch.log")
        sync()  # Save metrics and recorded states before potentially lengthy videos.
        for states in sorted((out / "runs").glob("*/states_*.npz")):
            if stopped:
                break
            directory = states.parent
            stem = states.stem.removeprefix("states_")
            output = directory / (stem + ".mp4")
            partial = directory / (stem + ".partial.mp4")
            command = [sys.executable, "-u", str(BASE / "replay/punch_video.py"),
                       "--run", str(directory / (stem + ".json")), "--states", str(states),
                       "--out", str(partial), "--stride", "1"]
            print("VIDEO", output, flush=True)
            result = launch(command, directory / "video.log")
            video_ok = result == 0 and partial.exists() and partial.stat().st_size > 0
            if video_ok:
                partial.replace(output)
            save(directory / "video.json", dict(command=command, returncode=result, complete=video_ok))
            code = code or int(not video_ok)
            copy_checked(directory, remote + "/" + directory.name, config, out / "upload.log")
    finally:
        uploaded = sync()
    status = json.loads((out / "status.json").read_text())
    return code or int(not uploaded or stopped or not status["evaluations_complete"]
                       or not status["videos_complete"])


if __name__ == "__main__":
    raise SystemExit(main())
