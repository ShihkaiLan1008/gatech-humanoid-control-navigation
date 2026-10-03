"""Closed-loop MuJoCo rollout of the checkpoint under the REAL robot's command
stream, starting from the real robot's logged state.

The sim robot is the TRAINING model exactly (mjlab entity spec + a ground
plane, same PD actuators / solver settings); the policy runs in the loop; the
step + EE commands are replayed from the deploy log by wall time; the upper
policy's measured-EE / EE-error obs are recomputed from SIM FK (they must be
consistent with the sim state). Where sim and real diverge is the sim2real
gap — the policy, commands, and initial state are identical by construction.

  uv run python scripts/rl/test/real_cmd_sim_rollout.py --t0 20 --dur 25
  uv run python scripts/rl/test/real_cmd_sim_rollout.py --mode openloop --t0 20 --dur 6
  uv run python scripts/rl/test/real_cmd_sim_rollout.py --view --t0 20 --dur 25   # + viser playback

openloop mode applies the LOGGED action sequence instead of the policy (pure
dynamics replay — expect divergence within ~1 s, balance is unstable).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import mujoco
import numpy as np
import torch

from rl.viz.viser_play import play_qpos
from rl.decoupled.mdp import LOWER_JOINT_NAMES, UPPER_JOINT_NAMES
from rl.g1_assets.g1_constants import G1_ACTION_SCALE, get_g1_robot_home_cfg

CACHE = Path("data/debug_real_0805/debug_0806_0.extract.npz")
CKPT = Path("logs/rsl_rl/g1_decoupled_joint/2026-08-06_18-29-40/model_17500.pt")

JOINTS = list(LOWER_JOINT_NAMES) + list(UPPER_JOINT_NAMES)
DEFAULT = np.zeros(29)
DEFAULT[[0, 6]] = -0.1
DEFAULT[[3, 9]] = 0.3
DEFAULT[[4, 10]] = -0.2
SCALE = np.array([next(s for p, s in G1_ACTION_SCALE.items()
                       if re.fullmatch(p, j)) for j in JOINTS])
H = 5
CTRL_DT = 0.02
PHYS_DT = 0.005


def build_actor(sd: dict, in_dim: int, out_dim: int) -> torch.nn.Sequential:
    net = torch.nn.Sequential(
        torch.nn.Linear(in_dim, 512), torch.nn.ELU(),
        torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, out_dim),
    )
    net.load_state_dict({k.replace("mlp.", ""): v for k, v in sd.items()
                         if k.startswith("mlp.")})
    return net.eval().double()


def build_model() -> mujoco.MjModel:
    """Training robot spec + a ground plane, training solver settings."""
    from mjlab.entity.entity import Entity

    spec = Entity(get_g1_robot_home_cfg()).spec
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1],
        contype=1, conaffinity=1,
    )
    model = spec.compile()
    model.opt.timestep = PHYS_DT
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    return model


def rpy_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    q = np.zeros(4)
    for ax, ang in (([0, 0, 1], yaw), ([0, 1, 0], pitch), ([1, 0, 0], roll)):
        r = np.zeros(4)
        mujoco.mju_axisAngle2Quat(r, np.array(ax, dtype=float), ang)
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, q if np.any(q) else np.array([1., 0, 0, 0]), r)
        q = out
    return q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def quat_inv(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("closedloop", "openloop"), default="closedloop")
    ap.add_argument("--t0", type=float, default=20.0)
    ap.add_argument("--dur", type=float, default=25.0)
    ap.add_argument("--view", action="store_true", help="viser playback after rollout")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    d = dict(np.load(CACHE))
    t_log = (d["t_ns"] - d["t_ns"][0]) / 1e9
    i0 = int(np.searchsorted(t_log, args.t0))
    assert i0 >= H, "t0 must be at least 5 log frames in"

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    upper = build_actor(ck["upper_body.actor_state_dict"], 519, 14)
    lower = build_actor(ck["lower_body.actor_state_dict"], 480, 15)

    model = build_model()
    data = mujoco.MjData(model)

    jid = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): j
           for j in range(model.njnt)}
    qadr = np.array([model.jnt_qposadr[jid[j]] for j in JOINTS])
    vadr = np.array([model.jnt_dofadr[jid[j]] for j in JOINTS])
    # ctrl slots are ACTUATOR-declaration order — map joint k -> its actuator
    ctrl_of = np.full(29, -1)
    for a in range(model.nu):
        ctrl_of[JOINTS.index(mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[a, 0]))] = a
    assert (ctrl_of >= 0).all()
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    tactile = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
               for n in ("left_tactile", "right_tactile")]

    # ---- initial state from the log at i0 ----
    grav0 = d["projected_gravity"][i0]
    pitch0 = float(np.arcsin(np.clip(grav0[0], -1, 1)))
    roll0 = float(np.arctan2(-grav0[1], -grav0[2]))
    data.qpos[:] = 0.0
    data.qpos[2] = 0.7837
    data.qpos[3:7] = rpy_quat(roll0, pitch0, 0.0)
    data.qpos[qadr] = d["dof_pos"][i0] + DEFAULT
    data.qvel[:] = 0.0
    data.qvel[3:6] = d["base_ang_vel"][i0]
    data.qvel[vadr] = d["dof_vel"][i0]
    mujoco.mj_forward(model, data)
    ankle_z = min(data.xpos[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link")][2]
        for s in ("left", "right"))
    data.qpos[2] += 0.035 - ankle_z  # ankle_roll link rests at 0.035 on the plane
    mujoco.mj_forward(model, data)

    # ---- obs history seeded from the REAL log frames i0-4..i0 ----
    hist = {
        "ang": [d["base_ang_vel"][k] for k in range(i0 - H + 1, i0 + 1)],
        "grav": [d["projected_gravity"][k] for k in range(i0 - H + 1, i0 + 1)],
        "jpos": [d["dof_pos"][k] for k in range(i0 - H + 1, i0 + 1)],
        "jvel": [d["dof_vel"][k] for k in range(i0 - H + 1, i0 + 1)],
        "act": [d["action"][k] for k in range(i0 - H, i0)],
    }
    last_action = d["action"][i0 - 1].copy()

    n_steps = int(args.dur / CTRL_DT)
    qpos_traj, grav_traj, act_traj, cmd_traj = [], [], [], []

    for k in range(n_steps):
        t_now = args.t0 + k * CTRL_DT
        i_log = min(int(np.searchsorted(t_log, t_now)), len(t_log) - 1)
        cv = d["command_vector"][i_log]

        # ---- sim measurements (same quantities the deploy stack logs) ----
        q_b = data.qpos[3:7].copy()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q_b)
        R = R.reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        ang = data.qvel[3:6].copy()
        jpos = data.qpos[qadr] - DEFAULT
        jvel = data.qvel[vadr].copy()

        if args.mode == "openloop":
            action = d["action"][i_log].copy()
        else:
            for key, val in (("ang", ang), ("grav", grav), ("jpos", jpos),
                             ("jvel", jvel), ("act", last_action)):
                hist[key].append(val)
                hist[key] = hist[key][-H:]
            proprio = np.concatenate([np.concatenate(hist[key])
                                      for key in ("ang", "grav", "jpos", "jvel", "act")])
            # EE obs recomputed from SIM FK, commands replayed from the log
            ee_obs, err_obs = [], []
            for hi, sl in ((0, slice(7, 14)), (1, slice(14, 21))):
                cmd = cv[sl]
                p_ee_w = data.xpos[tactile[hi]]
                q_ee_w = np.zeros(4)
                mujoco.mju_mat2Quat(q_ee_w, data.xmat[tactile[hi]])
                p_pel, q_pel = data.xpos[pelvis], np.zeros(4)
                mujoco.mju_mat2Quat(q_pel, data.xmat[pelvis])
                p_ee_b = R.T @ (p_ee_w - p_pel)  # anchor = pelvis (root) frame
                q_ee_b = quat_mul(quat_inv(q_pel), q_ee_w)
                if q_ee_b[0] < 0:
                    q_ee_b = -q_ee_b
                ee_obs.append(np.concatenate([p_ee_b, q_ee_b]))
                dp = cmd[:3] - p_ee_b
                rel_q = quat_mul(quat_inv(q_ee_b), cmd[3:7])
                Rm = np.zeros(9)
                mujoco.mju_quat2Mat(Rm, rel_q)
                Rm = Rm.reshape(3, 3)
                err_obs.append(np.concatenate([dp, Rm[:, 0], Rm[:, 1]]))
            up_in = np.concatenate([proprio, cv[7:14], cv[14:21],
                                    *ee_obs, *err_obs, np.zeros(8)])
            low_in = np.concatenate([proprio, cv[0:7], np.zeros(8)])
            with torch.no_grad():
                a_up = upper(torch.from_numpy(up_in[None]).double())[0].numpy()
                a_low = lower(torch.from_numpy(low_in[None]).double())[0].numpy()
            action = np.concatenate([a_low, a_up])
            last_action = action

        data.ctrl[ctrl_of] = np.clip(action, -100, 100) * SCALE + DEFAULT
        for _ in range(int(CTRL_DT / PHYS_DT)):
            mujoco.mj_step(model, data)

        qpos_traj.append(data.qpos.copy())
        grav_traj.append(grav)
        act_traj.append(action)
        cmd_traj.append(cv[0:7])

    qpos_traj = np.array(qpos_traj)
    grav_traj = np.array(grav_traj)
    out = Path(f"data/debug_real_0805/sim_rollout_{args.mode}.npz")
    with open(out, "wb") as f:
        np.savez_compressed(f, qpos=qpos_traj, grav=grav_traj,
                            action=np.array(act_traj), step_cmd=np.array(cmd_traj),
                            t0=args.t0, ctrl_dt=CTRL_DT)
    print(f"saved -> {out}")

    # ---- divergence report: sim vs real pitch on the same wall clock ----
    print(f"\n{args.mode} rollout from t0={args.t0:.1f}s "
          f"({'policy in loop' if args.mode == 'closedloop' else 'logged actions'}):")
    print("   t     grav_x sim   grav_x real   |base pitch sim-real|")
    for sec in np.arange(0, args.dur, 2.0):
        k = int(sec / CTRL_DT)
        if k >= len(grav_traj):
            break
        i_log = min(int(np.searchsorted(t_log, args.t0 + sec)), len(t_log) - 1)
        gs, gr = grav_traj[k, 0], d["projected_gravity"][i_log, 0]
        print(f"  {args.t0 + sec:5.1f}   {gs:+.4f}      {gr:+.4f}       "
              f"{abs(np.arcsin(np.clip(gs, -1, 1)) - np.arcsin(np.clip(gr, -1, 1))):.4f} rad")
    fell = qpos_traj[:, 2].min() < 0.45
    print(f"\nsim base z min {qpos_traj[:, 2].min():.3f} "
          f"({'FELL' if fell else 'stayed up'}); "
          f"sim grav_x p1/p99 {np.percentile(grav_traj[:, 0], 1):+.4f}/"
          f"{np.percentile(grav_traj[:, 0], 99):+.4f}")

    if args.view:
        play_qpos(model, qpos_traj, CTRL_DT, port=args.port)


if __name__ == "__main__":
    main()
