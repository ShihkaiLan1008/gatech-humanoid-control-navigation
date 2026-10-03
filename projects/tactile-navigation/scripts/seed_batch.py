"""Concurrent subprocesses with disjoint CPU affinity and bounded cleanup.

No simulator/controller imports. RSS limits are polled safeguards. A Slurm
memory request is a hard cap only when the cluster enforces memory cgroups;
CUDA needs a large virtual address space, so RLIMIT_AS is unsuitable here.
"""
from pathlib import Path
import json
import os
import signal
import subprocess
import time


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def cpu_masks(count, per_seed=0, available=None):
    available = sorted(os.sched_getaffinity(0) if available is None else available)
    # Use one hardware thread per core so different workers do not share a core.
    def core(cpu):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            return (int((base / "physical_package_id").read_text()),
                    int((base / "core_id").read_text()), cpu)
        except OSError:
            return (0, cpu, cpu)
    by_core = {}
    for cpu in available:
        by_core.setdefault(core(cpu)[:2], cpu)
    available = [by_core[key] for key in sorted(by_core)]
    if count < 1:
        raise ValueError("at least one seed is required")
    width = per_seed or len(available) // count
    if width < 1 or count * width > len(available):
        raise ValueError(f"{count} seeds need {count * max(1, width)} CPUs; only {len(available)} allocated")
    return [available[i * width:(i + 1) * width] for i in range(count)]


def group_rss():
    """Resident bytes per process group, including child processes."""
    totals = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            group, pages = int(fields[2]), int(fields[21])
            totals[group] = totals.get(group, 0) + pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            continue
    return totals


def run_batch(jobs, *, cwd, env, cpus_per_seed=0, memory_gb=0,
              timeout_s=0, on_finish=lambda row: None, poll_s=1., grace_s=60.):
    """Each job has command/directory/metadata. Only this parent writes reports."""
    masks = cpu_masks(len(jobs), cpus_per_seed)
    active, results = [], []
    stopped = False

    def interrupt(signum, frame):
        nonlocal stopped
        stopped = True

    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1)
    handlers = {s: signal.signal(s, interrupt) for s in signals}
    try:
        for job, mask in zip(jobs, masks):
            if stopped:
                break
            directory = Path(job["directory"])
            directory.mkdir(parents=True, exist_ok=True)
            command = ["taskset", "--cpu-list", ",".join(map(str, mask)), *job["command"]]
            child_env = dict(env, TACNAV_EVAL_JOURNAL=str(directory / "telemetry.jsonl"))
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                child_env[name] = str(len(mask))
            # Separate compilation caches; concurrent first-use kernels cannot race.
            cache = directory / ".cache"
            child_env.update(WARP_CACHE_PATH=str(cache / "warp"),
                             TORCH_EXTENSIONS_DIR=str(cache / "torch"),
                             CUDA_CACHE_PATH=str(cache / "cuda"))
            save(directory / "command.json", job["command"])
            row = dict(job.get("metadata", {}), directory=str(directory),
                       started_unix=time.time(), cpu_affinity=mask,
                       rss_limit_gb=memory_gb, rss_limit_kind="polled process-group RSS",
                       peak_rss_bytes=0, state="running")
            log = (directory / "run.log").open("w")
            try:
                proc = subprocess.Popen(command, cwd=cwd, env=child_env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                log.close()
                raise
            row["pid"] = proc.pid
            save(directory / "process.json", row)
            active.append(dict(proc=proc, log=log, row=row, stop_at=None, started=time.monotonic()))
            print(f"START {row}", flush=True)
        while active:
            now, rss = time.monotonic(), group_rss()
            for item in list(active):
                proc, row = item["proc"], item["row"]
                row["peak_rss_bytes"] = max(row["peak_rss_bytes"], rss.get(proc.pid, 0))
                reason = ("interrupted" if stopped else
                          "memory_limit" if memory_gb and rss.get(proc.pid, 0) > memory_gb * 2**30 else
                          "wall_timeout" if timeout_s and now - item["started"] > timeout_s else None)
                if reason and item["stop_at"] is None and proc.poll() is None:
                    item["stop_at"] = now
                    row["stop_reason"] = reason
                    os.killpg(proc.pid, signal.SIGINT)
                if item["stop_at"] is not None and now - item["stop_at"] > grace_s and proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                if proc.poll() is None:
                    continue
                item["log"].close()
                row.update(returncode=proc.returncode, state="finished", ended_unix=time.time(),
                           wall_clock_s=time.monotonic() - item["started"])
                save(Path(row["directory"]) / "process.json", row)
                active.remove(item)
                results.append(row)
                on_finish(row)
                print(f"DONE seed={row.get('seed')} exit={proc.returncode}", flush=True)
            if active:
                time.sleep(poll_s)
    finally:
        # An exception in reporting/launch must not leave GPU children alive.
        for item in active:
            if item["proc"].poll() is None:
                os.killpg(item["proc"].pid, signal.SIGKILL)
            item["proc"].wait()
            item["log"].close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return results, stopped
