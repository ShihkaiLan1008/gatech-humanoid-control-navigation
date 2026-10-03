"""What a run owes the operator no matter how it ends.

Three ways a route stops -- it reaches the goal, it gets stuck, the operator
kills it with Ctrl-C -- and all three leave the same thing worth keeping: a
believed trajectory, a true trajectory, the uncertainty at every step, and
every touch taken so far.  That is precisely what the 2D panel draws.

It used to depend on ``run()`` returning.  A ``KeyboardInterrupt`` propagated
out of the executor, ``rec`` stayed ``None``, the JSON was never written, and
a run the operator had watched for ten minutes left nothing behind but a
states file no tool would open.  The two helpers here close that: the record
is finalized from whatever accumulated, and the panel is rendered from it.

The renderer runs in a SUBPROCESS on purpose.  It builds a second MuJoCo env
to replay the states into, and doing that inside a process that already holds
one deadlocks on this machine (measured: two wrapper scripts that imported the
run module in-process hung for hours with no output and had to be killed).
"""
from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def guarded_run(ex, run_fn):
    """Run the route and come back with a record whatever happens.

    Returns ``(rec, how)`` where ``how`` is ``None`` for a route that ended on
    its own terms, and names the interruption otherwise.  The exception is not
    re-raised: the caller still has a states file to write and a video to make,
    and those are the point of the run.
    """
    try:
        return run_fn(), None
    except KeyboardInterrupt:
        print("\n[interrupted] Ctrl-C -- finalizing the record so far",
              flush=True)
        return ex.finalize("interrupted"), "interrupted"
    except Exception as exc:                  # noqa: BLE001 - see docstring
        traceback.print_exc()
        print(f"\n[crashed] {type(exc).__name__} -- finalizing the record",
              flush=True)
        return ex.finalize(f"crashed:{type(exc).__name__}"), "crashed"


def render_panel(run_json, states_npz, out_mp4, *, stride: int = 1,
                 fps: float | None = None, distance: float | None = None,
                 elevation: float | None = None) -> bool:
    """Render the MuJoCo replay + 2D belief-vs-truth panel.  Never raises.

    A missing or empty states file is normal for a run killed in its first
    seconds; say so and return False rather than failing the run that already
    succeeded in producing a record.
    """
    run_json, states_npz, out_mp4 = map(Path, (run_json, states_npz, out_mp4))
    if not states_npz.exists() or states_npz.stat().st_size == 0:
        print(f"[video] no states recorded yet -- skipping {out_mp4.name}",
              flush=True)
        return False
    venv = ROOT / "tactile_controller" / ".venv" / "bin" / "python"
    py = str(venv) if venv.exists() else sys.executable
    cmd = [py, str(ROOT / "scripts" / "demo_room_video.py"),
           "--run", str(run_json), "--states", str(states_npz),
           "--out", str(out_mp4), "--stride", str(int(stride))]
    if fps:
        cmd += ["--fps", str(float(fps))]
    if distance:
        cmd += ["--distance", str(float(distance))]
    if elevation:
        cmd += ["--elevation", str(float(elevation))]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src:tactile_controller/src"
    env["MUJOCO_GL"] = "egl"          # the replay is offscreen, always
    print(f"[video] rendering {out_mp4} ...", flush=True)
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    except Exception as exc:                  # noqa: BLE001
        print(f"[video] failed to launch: {type(exc).__name__}: {exc}",
              flush=True)
        return False
    if r.returncode != 0 or not out_mp4.exists():
        print(f"[video] renderer exited {r.returncode}; no {out_mp4.name}",
              flush=True)
        return False
    mb = out_mp4.stat().st_size / 1e6
    print(f"[video] wrote {out_mp4}  ({mb:.1f} MB)", flush=True)
    return True
