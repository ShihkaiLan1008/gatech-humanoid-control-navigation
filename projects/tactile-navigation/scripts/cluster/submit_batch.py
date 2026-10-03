#!/usr/bin/env python3
"""Freeze one source/configuration; submit map jobs with one process per seed."""
from datetime import datetime, timezone
from importlib import metadata, util
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
spec = util.spec_from_file_location("pace_submit", ROOT / "scripts/pace/submit_eval.py")
pace = util.module_from_spec(spec)
spec.loader.exec_module(pace)

HOSTS = {
    "pace": dict(ssh="Pace-ICE", root="/storage/ice1/2/1/jjang318/tacnav",
                 partition="ice-gpu", qos="coe-ice", gres="gpu:a100:1", cpu_partition="ice-cpu"),
    "skynet": dict(ssh="skynet", root="/coc/flash12/jjang318/tacnav",
                   partition="overcap", qos="debug", account="overcap", gres="gpu:2080_ti:1",
                   cpu_partition="overcap"),
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def remote(host, command, **kwargs):
    # SKYNET installs Slurm through the login shell. Quote both shell layers.
    return pace.remote(host, ["bash", "-l", "-c", shlex.join(map(str, command))], **kwargs)


def sbatch(config, base, room, *, smoke=False, sync=False, dependency=None):
    root = Path(config["remote_snapshot"])
    campaign = json.loads((base / "campaign.json").read_text())
    prefix = "smoke" if smoke else "sync" if sync else "eval"
    command = ["sbatch", "--parsable", f"--job-name=tn-{prefix}-{room}",
               f"--partition={config['cpu_partition'] if sync else config['partition']}",
               f"--qos={config['qos']}", f"--chdir={root}",
               f"--cpus-per-task={1 if sync else len(campaign['seeds']) * campaign['cpus_per_seed']}",
               f"--mem={2 if sync else len(campaign['seeds']) * campaign['memory_per_seed_gb'] + 8}G",
               f"--time={'01:00:00' if sync else '00:30:00' if smoke else '08:00:00'}",
               f"--output={root}/slurm-logs/{prefix}-{room}-%j.log"]
    if config.get("account"):
        command.append(f"--account={config['account']}")
    if not sync:
        command += [f"--gres={config['gres']}", "--hint=nomultithread", "--signal=B:USR1@600"]
    if dependency:
        command += [f"--dependency={'afterany' if sync else 'afterok'}:{dependency}", "--kill-on-invalid-dep=yes"]
    command += [str(root / "run.sbatch"), room]
    if smoke:
        command.append("--smoke")
    if sync:
        command.append("--sync-only")
    return command


def prepare(args):
    source = args.source.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_batch4"
    base = args.output_root.resolve() / stamp
    base.mkdir(parents=True, exist_ok=False)
    settings = json.loads(args.settings.read_text())
    assignments = [("pace", room) for room in args.pace_rooms] + [("skynet", room) for room in args.skynet_rooms]
    if not assignments or len(set(r for _, r in assignments)) != len(assignments):
        raise ValueError("provide distinct maps across the two clusters")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("provide unique seeds")
    settings.update(campaign=stamp, seeds=args.seeds, jobs=[dict(host=h, room=r) for h, r in assignments],
                    dropbox="dropbox:TacNav/tactile-navigation/runs/" + stamp,
                    runtime_packages=list(pace.RUNTIME_PACKAGES))
    save(base / "campaign.json", settings)
    pace.ROOT = source
    pace.stage(base, SimpleNamespace(curobo=args.curobo, controllers=[settings["controller"]],
               rooms=[r for _, r in assignments], seeds=args.seeds, demo_args=[]))
    # Overlay orchestration only. The selected navigation/controller source is frozen.
    overlay = ["scripts/punch_eval.py", "scripts/seed_batch.py", "scripts/cluster/batch_worker.py"]
    for name in overlay:
        target = base / "tactile-navigation" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    replay = source.parent / "replay/punch_video.py"
    if not replay.exists():
        replay = source / "scripts/punch_video.py"
    (base / "replay").mkdir()
    text = replay.read_text()
    # Portable replay root for both existing fleet and repository renderers.
    text = text.replace("ROOT = Path(__file__).resolve().parents[1]",
                        "ROOT = Path(__file__).resolve().parents[1] / 'tactile-navigation'")
    (base / "replay/punch_video.py").write_text(text)
    save(base / "source_provenance.json", dict(navigation=pace.git_info(source),
         controller=pace.git_info((source / "tactile_controller").resolve()),
         curobo=pace.git_info(args.curobo.resolve()), orchestration_overlays=overlay,
         local_runtime_versions={n:metadata.version(n) for n in pace.RUNTIME_PACKAGES}))
    (base / "slurm-logs").mkdir()
    (base / "run.sbatch").write_text('''#!/usr/bin/env bash
set -uo pipefail
source ./env.sh
sha256sum --quiet -c SHA256SUMS || exit 1
forward_stop() { kill -USR1 "$worker_pid" 2>/dev/null || true; }
trap forward_stop USR1 TERM INT
"$TACNAV_PYTHON" -u "$TACNAV_SNAPSHOT/tactile-navigation/scripts/cluster/batch_worker.py" --room "$1" "${@:2}" &
worker_pid=$!
while true; do
    wait "$worker_pid"
    code=$?
    if ! kill -0 "$worker_pid" 2>/dev/null; then break; fi
done
exit "$code"
''')
    hosts = {}
    for key in dict(assignments):
        config = dict(HOSTS[key])
        config.update(remote_snapshot=config["root"] + "/punch-eval/" + stamp,
                      python=config["root"] + "/tactile_controller/.venv/bin/python",
                      rclone=config["root"] + "/bin/rclone")
        hosts[key] = config
        host_dir = base / "hosts" / key
        host_dir.mkdir(parents=True)
        save(host_dir / "host.json", config)
        root = config["remote_snapshot"]
        (host_dir / "env.sh").write_text(
            f"export TACNAV_SNAPSHOT={shlex.quote(root)}\n"
            f"export TACNAV_PYTHON={shlex.quote(config['python'])}\n"
            f"export PYTHONPATH={shlex.quote(':'.join(root + '/' + p for p in ['tactile-navigation/scripts/pace/runtime', 'tactile-navigation/src', 'tactile_controller/src', 'curobo']))}\n"
            "export MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\n"
            "export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3\n")
    save(base / "hosts.json", hosts)
    sums = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and not path.is_symlink():
            sums.append(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + str(path.relative_to(base)))
    (base / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    for key, config in hosts.items():
        remote(config["ssh"], ["mkdir", "-p", config["remote_snapshot"]])
        pace.run(["rsync", "-az", "--protect-args", "--exclude=hosts/", "--", str(base) + "/",
                  config["ssh"] + ":" + config["remote_snapshot"] + "/"])
        # Include host bundles because they are in the manifest, then choose the runtime.
        pace.run(["rsync", "-az", "--protect-args", "--", str(base / "hosts") + "/",
                  config["ssh"] + ":" + config["remote_snapshot"] + "/hosts/"])
        for name in ("host.json", "env.sh"):
            remote(config["ssh"], ["cp", config["remote_snapshot"] + "/hosts/" + key + "/" + name,
                                   config["remote_snapshot"] + "/" + name])
        # Login-node check reads only paths/package metadata; GPU smoke runs in Slurm.
        check = "source ./env.sh; sha256sum --quiet -c SHA256SUMS && \"$TACNAV_PYTHON\" -c " + shlex.quote(
            "from importlib import metadata,util; import json; from pathlib import Path; "
            "p=json.loads(Path('source_provenance.json').read_text()); "
            "assert all(metadata.version(n)==v for n,v in p['local_runtime_versions'].items()); "
            "root=Path.cwd(); "
            "assert all(Path(x).is_relative_to(root) for n in ['execution','rl','assets','curobo'] "
            "for x in util.find_spec(n).submodule_search_locations); print('Source and runtime verified')")
        remote(config["ssh"], ["bash", "-c", "cd " + shlex.quote(config["remote_snapshot"]) + "; " + check])
    print("Prepared:", base, flush=True)
    return base


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT)
    p.add_argument("--curobo", type=Path, default=ROOT.parent.parent / "curobo")
    p.add_argument("--settings", type=Path, default=ROOT / "configs/evaluation_batch.json")
    p.add_argument("--pace-rooms", nargs="*", default=[])
    p.add_argument("--skynet-rooms", nargs="*", default=[])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--output-root", type=Path, default=Path.home() / "tacnav-evaluation")
    p.add_argument("--prepared", type=Path, help="reuse an already uploaded immutable campaign")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="8 simulated seconds, four processes; first map per host")
    p.add_argument("--validation", type=Path, help="smoke receipt; full jobs wait for successful validation on their host")
    a = p.parse_args(argv)
    base = a.prepared.resolve() if a.prepared else prepare(a)
    if a.prepare_only:
        return 0
    campaign = json.loads((base / "campaign.json").read_text())
    hosts = json.loads((base / "hosts.json").read_text())
    receipt = base / ("smoke_submission.json" if a.smoke else "submission.json")
    if receipt.exists():
        p.error(f"submission receipt already exists: {receipt}; refusing duplicate jobs")
    submitted, seen = [], set()
    validation = {row["host"]:row["job_id"] for row in json.loads(a.validation.read_text())} if a.validation else {}
    for job in campaign["jobs"]:
        if a.smoke and job["host"] in seen:
            continue
        seen.add(job["host"])
        config = hosts[job["host"]]
        command = sbatch(config, base, job["room"], smoke=a.smoke, dependency=validation.get(job["host"]))
        check = remote(config["ssh"], [command[0], "--test-only", *command[1:]], capture_output=True)
        result = remote(config["ssh"], command, capture_output=True)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise RuntimeError(result.stdout + result.stderr)
        row = dict(job, job_id=job_id, command=command, remote_snapshot=config["remote_snapshot"])
        submitted.append(row)
        save(receipt, submitted)
        if not a.smoke:
            sync_command = sbatch(config, base, job["room"], sync=True, dependency=job_id)
            result = remote(config["ssh"], sync_command, capture_output=True)
            row["upload_job_id"] = result.stdout.strip().split(";")[0]
            save(receipt, submitted)
        print(json.dumps(row), flush=True)
    print("Submission record:", receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
