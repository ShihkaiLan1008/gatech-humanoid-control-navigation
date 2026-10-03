#!/usr/bin/env python3
"""Upload the local demo sources/assets and submit the evaluation on PACE."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from execution.controllers import CONTROLLERS
from execution.demo_scenes import DEMO_ROOMS

# Include authored geometry even when it lives outside maps/. Never upload old
# outputs, virtualenvs, training data or .git. Uncommitted source is intentional.
NAV_FILES = ("src", "scripts", "configs", "maps", "docs/demo_maps",
             "models/wm", "pyproject.toml", "README.md")
CTRL_FILES = ("src", "pyproject.toml", "uv.lock", "README.md")
CUROBO_FILES = ("curobo", "pyproject.toml", "setup.py", "README.md")
RUNTIME_PACKAGES = ("numpy", "scipy", "matplotlib", "torch", "torchvision",
                    "mujoco", "mujoco-warp", "mjlab", "rsl-rl-lib",
                    "warp-lang", "cuda-core", "pytorch-kinematics")

# Only metadata/module lookup runs on the login node; no simulator or CUDA init.
REMOTE_CHECK = r'''
from importlib import metadata, util
import json
from pathlib import Path
import sys
snapshot = Path(sys.argv[1])
manifest = json.loads((snapshot / "submission.json").read_text())
for name, wanted in manifest["runtime_versions"].items():
    actual = metadata.version(name)
    if actual != wanted:
        raise SystemExit(f"Runtime mismatch: {name}={actual}, workstation={wanted}")
for module, root in (("execution", "tactile-navigation/src"),
                     ("rl", "tactile_controller/src"),
                     ("assets", "tactile_controller/src"), ("curobo", "curobo")):
    spec = util.find_spec(module)
    if spec is None:
        raise SystemExit(f"Missing module: {module}")
    locations = list(spec.submodule_search_locations or [spec.origin])
    for location in locations:
        if not Path(location).resolve().is_relative_to(snapshot / root):
            raise SystemExit(f"Wrong source for {module}: {location}")
print("Remote runtime and snapshot module paths verified.", flush=True)
'''


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=["vforce"])
    p.add_argument("--rooms", nargs="+", choices=[*DEMO_ROOMS, "all"], default=["room2"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--host", default=os.environ.get("TACNAV_HOST", "Pace-ICE"))
    p.add_argument("--remote-root", default=os.environ.get(
        "TACNAV_ROOT", "/storage/ice1/2/1/jjang318/tacnav"))
    p.add_argument("--remote-python", default=os.environ.get("TACNAV_PACE_PYTHON"),
                   help="existing PACE runtime; default: <remote-root>/tactile_controller/.venv/bin/python")
    p.add_argument("--curobo", type=Path, default=ROOT.parent.parent / "curobo",
                   help="local cuRobo source matching the workstation runtime")
    p.add_argument("--time", default="01:00:00")
    p.add_argument("--partition", default="ice-gpu")
    p.add_argument("--qos", default="coe-ice")
    p.add_argument("--gres", default="gpu:a100:1")
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--mem", default="16G")
    p.add_argument("--endpoint", choices=("dropbox", "local"), default="dropbox",
                   help="default: upload results to Dropbox after evaluation; local keeps them on PACE")
    p.add_argument("--dropbox-path", default="TacNav/tactile-navigation/runs")
    p.add_argument("--rclone", help="remote executable; default: <remote-root>/bin/rclone")
    p.add_argument("--video", action=argparse.BooleanOptionalAction, default=True,
                   help="render saved physics states before upload (default: on); --no-video disables")
    p.add_argument("--video-stride", type=int, default=1,
                   help="render every Nth recorded frame; playback remains real time")
    p.add_argument("--video-time", default="01:00:00", help="wall time for the video job")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="preview only; no SSH, rsync or submission")
    mode.add_argument("--test-only", action="store_true", help="upload and check sbatch --test-only; no job queued")
    p.add_argument("demo_args", nargs=argparse.REMAINDER, help="demo arguments after --")
    return p


def run(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, text=True, **kwargs)


def remote(host, command, **kwargs):
    # SSH receives one POSIX-quoted command. Values are never interpolated as code.
    return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                host, shlex.join([str(x) for x in command])], **kwargs)


def video_command(args, destination, eval_job_id=None):
    command = ["sbatch", "--parsable", f"--qos={args.qos}",
               f"--partition={args.partition}", f"--gres={args.gres}",
               f"--time={args.video_time}", f"--output={destination}/video-%j.log"]
    if args.test_only:
        command.append("--test-only")
    else:
        command.extend([f"--dependency=afterany:{eval_job_id}", "--kill-on-invalid-dep=yes"])
    return [*command, str(destination / "tactile-navigation/scripts/pace/render_eval.sbatch"),
            str(destination), str(eval_job_id or "<evaluation-job-id>"),
            args.remote_python or str(Path(args.remote_root) / "tactile_controller/.venv/bin/python"),
            str(args.video_stride)]


def upload_command(args, destination, eval_job_id=None, after_job_id=None):
    """A CPU follow-up also preserves logs when the GPU job fails or times out."""
    command = ["sbatch", "--parsable", f"--qos={args.qos}",
               f"--output={destination}/upload-%j.log"]
    if args.test_only:
        command.append("--test-only")
    else:
        command.extend([f"--dependency=afterany:{after_job_id or eval_job_id}", "--kill-on-invalid-dep=yes"])
    return [*command, str(destination / "tactile-navigation/scripts/pace/upload_eval.sbatch"),
            str(destination), str(eval_job_id or "<evaluation-job-id>"), args.dropbox_path,
            args.rclone or str(Path(args.remote_root) / "bin/rclone")]


def copy_inputs(source, destination, names):
    for name in names:
        path = source / name
        if not path.exists():
            raise FileNotFoundError(f"Missing runtime input: {path}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        # Materialize source symlinks so workstation-absolute links cannot leak
        # into the snapshot. The controller link is constructed separately.
        run(["rsync", "-aL", "--exclude=__pycache__/", "--exclude=*.pyc",
             "--exclude=.git/", "--exclude=*.egg-info/", "--",
             str(path) + ("/" if path.is_dir() else ""), target])


def git_info(path):
    def git(*args):
        r = subprocess.run(["git", "-C", str(path), *args], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return r.stdout.strip() if r.returncode == 0 else None
    return dict(local_path=str(path), branch=git("branch", "--show-current"),
                commit=git("rev-parse", "HEAD"), status=git("status", "--short"))


def stage_checkpoint_overrides(extra, nav):
    """Make workstation checkpoint paths portable, including --override=... ."""
    extra = list(extra)
    for i, arg in enumerate(extra):
        prefix = ""
        if arg.startswith("--override="):
            prefix, value = "--override=", arg[len("--override="):]
        elif i > 0 and extra[i - 1] == "--override":
            value = arg
        else:
            continue
        field, sep, raw = value.partition("=")
        if field.strip() != "checkpoint" or not sep:
            continue
        source = Path(raw.strip()).expanduser()
        if not source.is_absolute():
            source = next((base / source for base in (Path.cwd(), ROOT / "src", ROOT)
                           if (base / source).is_file()), ROOT / source)
        if not source.is_file():
            raise FileNotFoundError(f"Missing checkpoint override: {source}")
        rel = Path("models/submitted") / f"{i}_{source.name}"
        (nav / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, nav / rel)
        extra[i] = prefix + "checkpoint=" + rel.as_posix()
    return extra


def stage(snapshot, args):
    nav = snapshot / "tactile-navigation"
    ctrl = snapshot / "tactile_controller"
    copy_inputs(ROOT, nav, NAV_FILES)
    copy_inputs((ROOT / "tactile_controller").resolve(), ctrl, CTRL_FILES)
    copy_inputs(args.curobo.resolve(), snapshot / "curobo", CUROBO_FILES)
    (nav / "tactile_controller").symlink_to("../tactile_controller")
    extra = stage_checkpoint_overrides(args.demo_args, nav)
    has_checkpoint = any("checkpoint=" in a for a in extra
                         if a.startswith(("checkpoint=", "--override=checkpoint=")))
    if not has_checkpoint:
        # Read the defaults from their owners instead of maintaining model names
        # here. These imports do not instantiate a simulator.
        if "vforce" in args.controllers:
            from execution.vforce_backend import VForceConfig
            rel = Path(VForceConfig().checkpoint)
            source = (ROOT / rel).resolve()
            target = nav / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        if "footstep" in args.controllers:
            from execution.punch_footstep import CHECKPOINT
            copy_inputs(ROOT, nav, (CHECKPOINT,))
    return ["--controllers", *args.controllers, "--rooms", *args.rooms,
            "--seeds", *map(str, args.seeds), *(["--", *extra] if extra else [])]


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.demo_args[:1] == ["--"]:
        args.demo_args = args.demo_args[1:]
    reserved = {"--room", "--seed", "--controller", "--out", "--no-eval", "--view", "--no-video"}
    if any(a.partition("=")[0] in reserved for a in args.demo_args):
        p.error("room/seed/controller/output/logging are managed by the runner; put --no-video before --; --view needs a local display")
    if not Path(args.remote_root).is_absolute() or args.host.startswith("-"):
        p.error("--remote-root must be absolute and --host must be an SSH host/alias")
    if args.cpus < 1:
        p.error("--cpus must be positive")
    if args.video_stride < 1:
        p.error("--video-stride must be positive")
    render_video = args.video and any(c != "kinematic" for c in args.controllers)
    if not args.dropbox_path.strip("/"):
        p.error("--dropbox-path must name a results directory")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    destination = Path(args.remote_root) / "punch-eval" / stamp
    repo = destination / "tactile-navigation"
    python = args.remote_python or str(Path(args.remote_root) / "tactile_controller/.venv/bin/python")
    environment = ["env", f"TACNAV_REPO={repo}", f"TACNAV_PYTHON={python}",
                   f"TACNAV_SNAPSHOT={destination}",
                   f"PYTHONPATH={repo}/scripts/pace/runtime:{repo}/src:{destination}/tactile_controller/src:{destination}/curobo"]
    slurm = ["sbatch", "--parsable", f"--chdir={repo}", f"--output={destination}/slurm-%j.log",
             f"--partition={args.partition}", f"--qos={args.qos}", f"--gres={args.gres}",
             f"--cpus-per-task={args.cpus}", f"--mem={args.mem}", f"--time={args.time}", "--export=ALL"]
    if args.test_only:
        slurm.append("--test-only")
    eval_args = ["--controllers", *args.controllers, "--rooms", *args.rooms,
                 "--seeds", *map(str, args.seeds), *(["--", *args.demo_args] if args.demo_args else [])]
    if args.dry_run:
        print(f"Sync current working tree (including uncommitted changes): {ROOT}")
        print(f"Inputs: {', '.join(NAV_FILES)}; controller source/assets; cuRobo; selected checkpoint(s)")
        print(f"Remote snapshot: {args.host}:{destination}")
        print(shlex.join(["ssh", args.host, shlex.join([
            *environment, *slurm, str(repo / "scripts/pace/punch_eval.sbatch"), *eval_args])]))
        if render_video:
            print("Then render recorded states:")
            print(shlex.join(video_command(args, destination, "<evaluation-job-id>")))
        if args.endpoint == "dropbox":
            print("Then queue upload after rendering/evaluation (including failure or timeout):")
            print(shlex.join(upload_command(args, destination, "<evaluation-job-id>",
                                           "<video-job-id>" if render_video else None)))
        return 0

    for tool in ("ssh", "rsync"):
        if not shutil.which(tool):
            p.error(f"Missing local executable: {tool}")
    if args.endpoint == "dropbox":
        rclone = args.rclone or str(Path(args.remote_root) / "bin/rclone")
        configured = remote(args.host, [rclone, "listremotes"], capture_output=True)
        if "dropbox:" not in configured.stdout.splitlines():
            p.error(f"No dropbox: remote configured for {rclone} on {args.host}; use --endpoint local to keep results on PACE")
    with tempfile.TemporaryDirectory(prefix="tacnav-submit-") as temporary:
        snapshot = Path(temporary)
        print("Preparing source, map and checkpoint snapshot...", flush=True)
        eval_args = stage(snapshot, args)
        command = [*environment, *slurm, str(repo / "scripts/pace/punch_eval.sbatch"), *eval_args]
        manifest = dict(submitted_utc=stamp, host=args.host, remote_snapshot=str(destination),
                        repositories={"navigation": git_info(ROOT),
                                      "controller": git_info((ROOT / "tactile_controller").resolve()),
                                      "curobo": git_info(args.curobo.resolve())},
                        local_python=sys.version, remote_python=python,
                        runtime_versions={name: metadata.version(name) for name in RUNTIME_PACKAGES},
                        eval_args=eval_args, remote_command=command, endpoint=args.endpoint,
                        dropbox_path=args.dropbox_path if args.endpoint == "dropbox" else None,
                        video=render_video, video_stride=args.video_stride)
        (snapshot / "submission.json").write_text(json.dumps(manifest, indent=2) + "\n")
        checksums = []
        for path in sorted(snapshot.rglob("*")):
            if path.is_file() and not path.is_symlink():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                checksums.append(f"{digest}  {path.relative_to(snapshot)}\n")
        (snapshot / "SHA256SUMS").write_text("".join(checksums))
        # mkdir without -p on the snapshot itself prevents any accidental reuse.
        remote(args.host, ["mkdir", "-p", destination.parent])
        remote(args.host, ["mkdir", destination])
        print(f"Uploading {len(checksums)} files to {args.host}:{destination}", flush=True)
        run(["rsync", "-az", "--protect-args", "--stats", "-e",
             "ssh -o BatchMode=yes -o ConnectTimeout=20", "--", f"{snapshot}/",
             f"{args.host}:{destination}/"])
        remote(args.host, [*environment, python, "-c", REMOTE_CHECK, destination])
        result = remote(args.host, command, capture_output=True)
        if result.stderr:
            print(result.stderr.strip(), flush=True)
        if args.test_only:
            print("Upload and Slurm validation passed; no job submitted.")
        else:
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdigit():
                raise ValueError(f"Cannot parse sbatch job ID: {result.stdout!r}")
            print(f"Submitted job {job_id}")
            print(f"Results: {args.host}:{repo}/out/evaluation/pace_{job_id}/")
            print(f"Log: {args.host}:{destination}/slurm-{job_id}.log")
        receipt = ROOT / "out/pace_submissions" / f"{stamp}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        manifest.update(slurm_stdout=result.stdout, slurm_stderr=result.stderr,
                        test_only=args.test_only)
        # Save the GPU job receipt before queuing the uploader; its ID must not
        # be lost if submission of the second job fails.
        receipt.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Submission record: {receipt}")
        after_job_id = None if args.test_only else job_id
        if render_video:
            command = video_command(args, destination, after_job_id)
            manifest["video_command"] = command
            receipt.write_text(json.dumps(manifest, indent=2) + "\n")
            rendered = remote(args.host, command, capture_output=True)
            if rendered.stderr:
                print(rendered.stderr.strip(), flush=True)
            if args.test_only:
                print("Video job passed Slurm validation; no job queued.")
            else:
                after_job_id = rendered.stdout.strip().split(";")[0]
                if not after_job_id.isdigit():
                    raise ValueError(f"Cannot parse video job ID: {rendered.stdout!r}")
                manifest["video_job_id"] = after_job_id
                print(f"Video job {after_job_id}: replays recorded states after evaluation {job_id}")
                print(f"Video log: {args.host}:{destination}/video-{after_job_id}.log")
            receipt.write_text(json.dumps(manifest, indent=2) + "\n")
        if args.endpoint == "dropbox":
            command = upload_command(args, destination, None if args.test_only else job_id, after_job_id)
            manifest["upload_command"] = command
            receipt.write_text(json.dumps(manifest, indent=2) + "\n")
            uploaded = remote(args.host, command, capture_output=True)
            if uploaded.stderr:
                print(uploaded.stderr.strip(), flush=True)
            manifest.update(upload_slurm_stdout=uploaded.stdout, upload_slurm_stderr=uploaded.stderr)
            if args.test_only:
                print("Dropbox upload job also passed Slurm validation; no job queued.")
            else:
                upload_id = uploaded.stdout.strip().split(";")[0]
                manifest["upload_job_id"] = upload_id
                print(f"Dropbox upload job {upload_id}: starts after job {after_job_id} ends")
                print(f"Dropbox: {args.dropbox_path.rstrip('/')}/pace_{job_id}/")
                print(f"Upload log: {args.host}:{destination}/upload-{upload_id}.log")
            receipt.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError, metadata.PackageNotFoundError) as error:
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        print(f"Submission failed: {error}", file=sys.stderr)
        raise SystemExit(1)
