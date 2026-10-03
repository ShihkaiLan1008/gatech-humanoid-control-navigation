"""Visualize the real-robot deploy log in MuJoCo (browser viser viewer).

Default view: the MEASURED robot (solid, from logged dof_pos + IMU attitude,
feet re-grounded per frame) with a translucent green GHOST overlaid on the
same base showing the policy's PD TARGET pose (action*scale + default) — the
gap between the two is what the real actuators did not deliver.

  uv run python scripts/rl/test/real_log_play.py --start 20 --end 42
  uv run python scripts/rl/test/real_log_play.py --no-ghost        # measured only
  # open the printed http://localhost:8080 in a browser

The log has no odometry, so the base stays at the origin (walks in place);
base roll/pitch from projected_gravity, yaw integrated from the gyro, base
height set each frame so the lower ankle_roll link sits at its resting height.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mujoco
import numpy as np

import assets
from rl.viz.viser_play import _aim_camera, _disable_mesh_lod
from rl.decoupled.mdp import LOWER_JOINT_NAMES, UPPER_JOINT_NAMES
from rl.g1_assets.g1_constants import G1_ACTION_SCALE

CACHE = Path("data/debug_real_0805/debug_0806_0.extract.npz")
BASE_Z = 0.7837
ANKLE_REST_Z = 0.035  # ankle_roll link origin height when the sole rests on the plane

JOINTS = list(LOWER_JOINT_NAMES) + list(UPPER_JOINT_NAMES)
DEFAULT = np.zeros(29)
DEFAULT[[0, 6]] = -0.1
DEFAULT[[3, 9]] = 0.3
DEFAULT[[4, 10]] = -0.2
SCALE = np.array([next(s for p, s in G1_ACTION_SCALE.items()
                       if re.fullmatch(p, j)) for j in JOINTS])


def rpy_to_quat(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.stack([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ], axis=-1)  # wxyz


def build_model(with_ghost: bool) -> mujoco.MjModel:
    path = str(assets.ASSET_DIR / assets.TACTILE_SCENE_XML)
    spec = mujoco.MjSpec.from_file(path)
    if with_ghost:
        ghost = mujoco.MjSpec.from_file(path)
        for j in ghost.joints:
            if j.type == mujoco.mjtJoint.mjJNT_FREE:
                ghost.delete(j)
                break
        for g in ghost.geoms:
            g.contype = 0
            g.conaffinity = 0
            g.rgba = np.array([0.2, 0.9, 0.4, 0.25])
        for a in list(ghost.actuators):
            ghost.delete(a)
        for s in list(ghost.sensors):
            ghost.delete(s)
        body = spec.worldbody.add_frame().attach_body(ghost.body("pelvis"), "ghost_", "")
        body.mocap = True
    return spec.compile()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ghost", action="store_true",
                    help="measured robot only, no PD-target overlay")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    d = dict(np.load(CACHE))
    t = (d["t_ns"] - d["t_ns"][0]) / 1e9
    m = (t >= args.start) & (t <= (args.end if args.end is not None else t[-1]))
    t = t[m]

    measured = d["dof_pos"][m] + DEFAULT
    target = d["action"][m] * SCALE + DEFAULT

    grav = d["projected_gravity"][m]
    pitch = np.arcsin(np.clip(grav[:, 0], -1, 1))
    roll = np.arctan2(-grav[:, 1], -grav[:, 2])
    wz = d["base_ang_vel"][m][:, 2]
    yaw = np.concatenate([[0.0], np.cumsum(wz[:-1] * np.diff(t))])
    quat = rpy_to_quat(roll, pitch, yaw)

    model = build_model(with_ghost=not args.no_ghost)
    data = mujoco.MjData(model)
    qadr = {}
    for j in range(model.njnt):
        qadr[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)] = \
            model.jnt_qposadr[j]
    ankle_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                      f"{s}_ankle_roll_link")
                    for s in ("left", "right")]

    n = len(t)
    qpos = np.zeros((n, model.nq))
    qpos[:, 2] = BASE_Z
    qpos[:, 3:7] = quat
    for k, jn in enumerate(JOINTS):
        qpos[:, qadr[jn]] = measured[:, k]
        if not args.no_ghost:
            qpos[:, qadr["ghost_" + jn]] = target[:, k]

    # ground the measured robot: per-frame z so the lower ankle_roll link rests
    for i in range(n):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        low = min(data.xpos[b][2] for b in ankle_bodies)
        qpos[i, 2] += ANKLE_REST_Z - low

    dt = float(np.median(np.diff(t)))
    print(f"[real_log_play] t {t[0]:.1f}..{t[-1]:.1f}s, {n} frames @ {1 / dt:.0f} Hz"
          + ("" if args.no_ghost else "  (green ghost = policy PD target)"))

    # viser playback (play_qpos + a mocap write for the ghost base)
    import viser
    from mjviser import Viewer

    model.opt.timestep = dt
    state = {"k": 0}

    def step(mm, dd):
        i = state["k"]
        dd.qpos[:] = qpos[i]
        dd.qvel[:] = 0.0
        if not args.no_ghost:
            dd.mocap_pos[0] = qpos[i, 0:3]
            dd.mocap_quat[0] = qpos[i, 3:7]
        mujoco.mj_forward(mm, dd)
        state["k"] = (i + 1) % n

    def reset(mm, dd):
        state["k"] = 0
        step(mm, dd)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    _disable_mesh_lod(server)
    _aim_camera(server)
    reset(model, data)
    print(f"[real_log_play] serving on http://localhost:{args.port}  (Ctrl-C to stop)",
          flush=True)
    Viewer(model, data, step_fn=step, reset_fn=reset, server=server).run()


if __name__ == "__main__":
    main()
