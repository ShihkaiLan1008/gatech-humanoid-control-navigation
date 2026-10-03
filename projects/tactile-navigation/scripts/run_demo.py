#!/usr/bin/env python3
"""One entry point for both rooms.

    scripts/run_demo.py --room room2     --controller vforce --view
    scripts/run_demo.py --room demo_room --controller vforce --view

Dispatches to the room's own runner in a SUBPROCESS rather than importing it.
That is not fastidiousness: importing a runner in-process and calling its
`main()` deadlocked on this machine -- two wrapper scripts sat at 100% CPU with
no output for hours before being killed -- because both the runner and the
wrapper race to initialise the same GPU/warp context.  A subprocess has one.

Every run leaves the same three files in `--out`, whether it arrives, gets
stuck, or you press Ctrl-C -- the record, the states, and the video:

    room2       room2_fwd_s0.json  states_room2_fwd_s0.npz  room2_fwd_s0.mp4
    demo_room   demo_run_s0.json   states_demo_s0.npz       demo_room_s0.mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ROOMS = {
    "room2": ROOT / "scripts" / "room2_run.py",
    "demo_room": ROOT / "scripts" / "demo_room_run.py",
}


def project_python() -> str:
    """The interpreter the simulator lives in, NOT the one that ran this file.

    This script's shebang is `env python3`, so launching it from a shell picks
    up whatever python is first on PATH -- on this machine a conda base with
    none of the stack.  Measured: the child died on `ModuleNotFoundError:
    scipy` while `tactile_controller/.venv` had scipy 1.18.1 all along.
    """
    venv = ROOT / "tactile_controller" / ".venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    print(f"[warn] {venv} not found -- falling back to {sys.executable}.\n"
          f"       Run ./scripts/setup_env.sh first.", file=sys.stderr)
    return sys.executable


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run a tactile-navigation demo in MuJoCo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--room", choices=sorted(ROOMS), required=True,
                    help="which scanned room to walk")
    ap.add_argument("--controller", choices=("vforce", "hand_force"),
                    default="vforce",
                    help="vforce (default): the decoupled vel-force checkpoint "
                         "vforce_model_60000_v1, force-commanded presses.  "
                         "hand_force: the older wall-following stack.")
    ap.add_argument("--view", action="store_true",
                    help="watch the physics live in a MuJoCo window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: out/<room>")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    a, extra = ap.parse_known_args(argv)

    out = a.out or (ROOT / "out" / a.room)
    cmd = [project_python(), "-u", str(ROOMS[a.room]),
           "--controller", a.controller, "--seed", str(a.seed),
           "--out", str(out), "--stride", str(a.stride)]
    if a.view:
        cmd.append("--view")
    if a.no_video:
        cmd.append("--no-video")
    cmd += extra                       # room-specific flags pass straight on

    env = dict(os.environ)
    env["PYTHONPATH"] = "src:tactile_controller/src"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # A live window needs a real GL context; EGL is headless and cannot serve
    # one.  Recording never goes through this env -- the replay subprocess sets
    # its own EGL -- so choosing here costs the video nothing.
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "glfw" if a.view else "egl")
    if a.view and env["MUJOCO_GL"] == "egl":
        print("[view] MUJOCO_GL=egl is headless; unset it or use glfw",
              file=sys.stderr)

    print("+", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            # The terminal sent SIGINT to the whole foreground group, so the
            # child already has it and is finalizing its record and video.
            # Dying here would orphan that work; wait it out.  A second Ctrl-C
            # means the operator wants out now.
            print("\n[run_demo] Ctrl-C -- waiting for the record and the "
                  "video (Ctrl-C again to abandon them)", flush=True)
            try:
                return proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                return 130


if __name__ == "__main__":
    raise SystemExit(main())
