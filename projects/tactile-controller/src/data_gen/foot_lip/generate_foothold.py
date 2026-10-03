"""Headless single-step dataset generator for the foot_lip (LIPM) pipeline.

KINEMATIC-ONLY clips (no torques/wrenches): CoM + feet + joint pos/vel, per
the 2026-08-19 decision. Randomization mirrors foot_placement v3
(randomize_cfg.yml): quota cells = swing(2) x landing-x(9) x dy(4) x yaw(6) x
tempo(3), equal quota per cell, gridded dims jitter inside their cell,
lift-off / apex full-range uniform. Lift-off ranges are the mirror image of
the landing set so the state distribution is closed under chaining. Left and
right swings share one canonical distribution; the builder mirrors yaw signs.

Gates (in order, cheapest first; a clip must pass all):
  1. plan-A DSP ZMP levels inside the phase's two-feet support hull (margin)
  2. 4.70 multibody-ZMP refinement converged (< 1 cm)
  3. IK residuals: feet < 3 mm, CoM xy < 2 mm
  4. left-right lower-leg mesh distance >= 5 mm (most expensive, LAST)

Usage:
  uv run python -m data_gen.foot_lip.generate_foothold --out data/foot_lip_v1 --limit 100   # smoke
  uv run python -m data_gen.foot_lip.generate_foothold --out data/foot_lip_v1              # full 9072

``--mode orbit`` builds STEADY-STATE steps on the FootHoldCommand clock instead
(ORBIT_CFG): one clip = one step of T_step with lift_frac*T_step of glide, the
CoM on the H-LIP P1/P2 orbit of its landing (the FootholdMPCCommand reference,
z_orbit = the MPC's com_height), the swing foot on the command's smoothstep +
sin^2 arc, plus the standing pose over the initial and the terminal feet
(q_rest_start / q_rest). Cells = swing(2) x landing-x x dy x yaw x prev-step
delta-x x delta-y (the lift-off foot is the previous step's stance foot, that
step = this one up to the sampler's jitter); delta-yaw and apex uniform. Gates: glide ZMP inside the
initial two-feet hull, IK residuals, leg mesh distance; the multibody ZMP is
stored as a diagnostic (no 4.70 refinement unless ``--refine`` -- the clip
CoM stays ON the orbit so the joint posture and the MPC reference agree).

  uv run python -m data_gen.foot_lip.generate_foothold --mode orbit --out data/foot_lip_orbit_v1 --limit 60
  uv run python -m data_gen.foot_lip.generate_foothold --mode orbit --out data/foot_lip_orbit_v1
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial import ConvexHull

import assets
from data_gen.foot_lip.lipm import LIPM, WalkPrimitive
from data_gen.common.collision import LegCollisionChecker

CFG = {
    "dt": 0.02,
    # canonical frame: stance-foot yaw frame, dy = width offset from nominal
    # (positive outward), yaw mirrored by the builder for left swings.
    "landing": {"x": (-0.20, 0.25), "dy": (-0.08, 0.18), "yaw": (-0.30, 0.30)},
    "liftoff": {"dx": (-0.25, 0.20), "dy": (-0.08, 0.18), "yaw": (-0.30, 0.30)},
    "tempo": {"T": (0.75, 1.25),
              "fractions": {"ds_pre": 0.25, "swing": 0.45, "ds_post": 0.30}},
    "apex": (0.05, 0.08),
    "grid": {"x_cells": 9, "dy_cells": 4, "yaw_cells": 6, "tempo_bins": 3},
    "gates": {"dsp_hull_margin_m": 0.005, "zmp_err_m": 0.010,
              "ik_foot_m": 0.003, "ik_com_m": 0.002, "min_gap_m": 0.005},
    "attempts_per_slot": 3,
}

# Steady-state steps on the FootHoldCommand clock (joint_foot_mpc COMMANDS):
# Timing(step_period 0.5, lift_frac 0.25); sampler landing x in [-0.2, 0.25],
# yaw +-0.3, apex [0.05, 0.09]; width w = w0 + dy against the sampler's
# w >= 0.14 floor. The lift-off pose is DERIVED from the previous step: the
# sampler keeps one nominal per episode and jitters each step by +-0.04 m /
# +-0.08 rad, so the previous step equals this one up to prev_step_delta
# (twice the jitter); the lift-off foot is that step's stance foot seen from
# the new stance frame. z_orbit = PELVIS_KNEES_BENT_Z, the MPC's com_height.
ORBIT_CFG = {
    "dt": 0.02,
    "T_step": 0.5,
    "lift_frac": 0.25,
    "z_orbit": 0.757,
    "landing": {"x": (-0.20, 0.25), "dy": (-0.10, 0.11), "yaw": (-0.30, 0.30)},
    "prev_step_delta": {"dx": (-0.08, 0.08), "dy": (-0.08, 0.08), "yaw": (-0.16, 0.16)},
    "apex": (0.05, 0.09),
    "grid": {"x_cells": 9, "dy_cells": 4, "yaw_cells": 6,
             "dprev_x_cells": 3, "dprev_y_cells": 3},
    "gates": {"dsp_hull_margin_m": 0.005, "ik_foot_m": 0.003, "ik_com_m": 0.002,
              "min_gap_m": 0.005},
    "attempts_per_slot": 3,
}

# CoP box about the foot SITE (site frame; from the measured sole polygon,
# ankle-frame x in [-0.054, +0.132] with the site 0.04 ahead of the ankle).
FOOT_BOX_X = (-0.09, 0.09)
FOOT_BOX_Y = (-0.026, 0.026)


def _sub(lo, hi, cells, i, u):
    """Uniform draw inside cell i of [lo, hi] split into `cells` bins."""
    return lo + (hi - lo) * (i + u) / cells


def _hull_margin(foot_poses, point_xy):
    """Signed margin of point_xy w.r.t. the convex hull of the feet CoP boxes
    (negative = inside by that much)."""
    corners = []
    for pose in foot_poses:
        R, t = pose[:2, :2], pose[:2, 3]
        for cx in FOOT_BOX_X:
            for cy in FOOT_BOX_Y:
                corners.append(t + R @ np.array([cx, cy]))
    hull = ConvexHull(np.array(corners))
    return float(np.max(hull.equations[:, :2] @ point_xy + hull.equations[:, 2]))


_W = {}


def _init_worker(refine=False):
    lipm = LIPM(assets.ASSET_DIR / assets.UNITREE_TACTILE_XML, viz=False)
    lipm.setup_robot()
    _W["lipm"] = lipm
    _W["checker"] = LegCollisionChecker(lipm.robot)
    _W["refine"] = refine


def _ik_residuals(lipm, qs, com_xy, foot_traj):
    """Max foot-position and CoM-xy IK error [m] over the knots of qs."""
    model, data = lipm.robot.model, lipm.robot.data
    foot_err = com_err = 0.0
    for i in range(len(qs)):
        pin.forwardKinematics(model, data, qs[i])
        pin.updateFramePlacements(model, data)
        com_err = max(com_err, float(np.linalg.norm(
            pin.centerOfMass(model, data, qs[i])[:2] - com_xy[i])))
        for name in ("left_foot", "right_foot"):
            M = data.oMf[model.getFrameId(name)]
            foot_err = max(foot_err, float(np.linalg.norm(
                foot_traj[name][i][:3, 3] - M.translation)))
    return foot_err, com_err


def _foot_channels(model, qs, ft, dt):
    """Joint velocities and world feet pos/quat(xyzw)/vel from qs + foot poses."""
    N = len(qs)
    vs = np.stack([pin.difference(model, qs[i], qs[min(i + 1, N - 1)]) / dt
                   for i in range(N)])
    vs[-1] = vs[-2]
    feet_pos = np.stack([ft["left_foot"][:, :3, 3], ft["right_foot"][:, :3, 3]], axis=1)
    feet_quat = np.zeros((N, 2, 4))
    feet_yaw = np.zeros((N, 2))
    for k, name in enumerate(("left_foot", "right_foot")):
        for i in range(N):
            R = ft[name][i][:3, :3]
            feet_quat[i, k] = pin.Quaternion(R).coeffs()  # xyzw
            feet_yaw[i, k] = np.arctan2(R[1, 0], R[0, 0])
    feet_vel = np.zeros((N, 2, 6))
    feet_vel[:, :, :3] = np.gradient(feet_pos, dt, axis=0)
    feet_vel[:, :, 5] = np.gradient(np.unwrap(feet_yaw, axis=0), dt, axis=0)
    return vs, feet_pos, feet_quat, feet_vel


def _liftoff_from_prev_step(prev, sgn, w0):
    """Lift-off offsets (lift_dx, lift_dy, lift_dyaw) of the swing foot from
    the previous step (ell_p, dy_p, yaw_p) that placed the CURRENT stance
    foot. prev is in the previous stance frame; sgn = +1 for a left swing.
    The previous stance foot (= this swing foot) sits at the inverse of that
    transform in the new stance frame."""
    ell_p, dy_p, yaw_p = prev
    c, s_ = np.cos(yaw_p), np.sin(yaw_p)
    # previous step landing in ITS stance frame: the current stance foot,
    # lateral sign opposite to this swing's side
    land = np.array([ell_p, -sgn * (w0 + dy_p)])
    loc = -np.array([[c, s_], [-s_, c]]) @ land      # R(yaw_p)^T applied to -land
    return float(loc[0]), float(sgn * loc[1] - w0), float(-yaw_p)


def _build_orbit_clip(task):
    """Generate + gate one steady-state step. Returns (idx, meta | None, reason)."""
    idx, out_dir, seed, cell = task
    lipm, checker = _W["lipm"], _W["checker"]
    cfg = ORBIT_CFG
    dt, T, f = cfg["dt"], cfg["T_step"], cfg["lift_frac"]
    g = cfg["grid"]
    swing_i, xi, dyi, yi, pxi, pyi = cell

    for attempt in range(cfg["attempts_per_slot"]):
        rng = np.random.default_rng([seed, idx, attempt, 7])
        x = _sub(*cfg["landing"]["x"], g["x_cells"], xi, rng.uniform())
        dy = _sub(*cfg["landing"]["dy"], g["dy_cells"], dyi, rng.uniform())
        yaw = _sub(*cfg["landing"]["yaw"], g["yaw_cells"], yi, rng.uniform())
        d = cfg["prev_step_delta"]
        prev = (x + _sub(*d["dx"], g["dprev_x_cells"], pxi, rng.uniform()),
                dy + _sub(*d["dy"], g["dprev_y_cells"], pyi, rng.uniform()),
                yaw + rng.uniform(*d["yaw"]))
        prev = (float(np.clip(prev[0], *cfg["landing"]["x"])),
                float(np.clip(prev[1], *cfg["landing"]["dy"])),
                float(np.clip(prev[2], *cfg["landing"]["yaw"])))
        apex = rng.uniform(*cfg["apex"])

        swing = ("left", "right")[swing_i]
        stance = ("right", "left")[swing_i]
        mirror = -1.0 if swing == "left" else 1.0  # canonical = right swing
        sgn = 1.0 if swing == "left" else -1.0     # swing lands on +y of stance
        lift = _liftoff_from_prev_step(
            (prev[0], prev[1], mirror * prev[2]), sgn, lipm.nominal_foot_y_distance)
        wp = WalkPrimitive(
            stance_foot=f"{stance}_foot", swing_foot=f"{swing}_foot",
            swing_start_pose=np.eye(4),
            step_dx=x, step_dy=dy, step_dyaw=mirror * yaw,
            step_Tsupport=(1.0 - f) * T, swing_height=apex,
            Tdsp_start=f * T, Tdsp_end=0.0,
            lift_dx=lift[0], lift_dy=lift[1], lift_dyaw=lift[2],
        )

        out = lipm.generate_orbit_com_trajectory(wp, cfg["z_orbit"], T, f, dt=dt)
        ft = lipm.generate_swing_trajectory_clock(out, wp, T, f)
        # gate 1: the glide ZMP (= CoM, both ends of the segment) inside the
        # hull of the stance and the lift-off feet
        dsp = out["phase"] == 0
        k_end = int(np.nonzero(dsp[:-1])[0][-1])
        feet0 = [out["stance_pose"], out["swing_start"]]
        m1 = _hull_margin(feet0, out["zmp_ref"][0])
        m2 = _hull_margin(feet0, out["zmp_ref"][k_end])
        margin = -cfg["gates"]["dsp_hull_margin_m"]
        if m1 > margin or m2 > margin:
            reason = "glide_zmp_outside_hull"
            continue

        if _W["refine"]:
            qs = lipm.refine_with_multibody_zmp(out, ft, dt=dt, max_iters=12, tol=0.01)
            if not out["zmp_refine"]["converged"]:
                reason = "zmp_refine_diverged"
                continue
        else:
            qs = lipm.solve_ik(out, ft)
        q_rest_start = lipm.solve_rest_pose(
            {"left_foot": ft["left_foot"][0], "right_foot": ft["right_foot"][0]})
        q_rest = lipm.solve_rest_pose(
            {"left_foot": ft["left_foot"][-1], "right_foot": ft["right_foot"][-1]})

        # gate 2: IK residuals (steps + both rest poses)
        foot_err, com_err = _ik_residuals(lipm, qs, out["com"][:, :2], ft)
        rest_ok = True
        for q_r, k in ((q_rest_start, 0), (q_rest, -1)):
            fe, _ = _ik_residuals(lipm, q_r[None], np.zeros((1, 2)),
                                  {n: ft[n][k][None] for n in ft})
            rest_ok &= fe <= cfg["gates"]["ik_foot_m"]
        if (foot_err > cfg["gates"]["ik_foot_m"] or com_err > cfg["gates"]["ik_com_m"]
                or not rest_ok):
            reason = "ik_residual"
            continue

        # gate 3 (most expensive, LAST): leg-vs-leg mesh distance
        min_gap = checker.min_gap(np.concatenate([qs, q_rest_start[None], q_rest[None]]),
                                  stride=2)
        if min_gap < cfg["gates"]["min_gap_m"]:
            reason = "leg_collision"
            continue

        # diagnostics: full-model ZMP vs the stance CoP box during SSP
        model = lipm.robot.model
        zmp_mb = lipm._multibody_zmp(qs, dt)
        ssp = np.nonzero(out["phase"] == 1)[0][1:-1]
        R, tst = out["stance_pose"][:2, :2], out["stance_pose"][:2, 3]
        loc = (zmp_mb[ssp] - tst) @ R
        excess = np.maximum(0.0, np.abs(loc) - np.array([FOOT_BOX_X[1], FOOT_BOX_Y[1]]))
        zmp_excess = excess.max(axis=0)

        vs, feet_pos, feet_quat, feet_vel = _foot_channels(model, qs, ft, dt)
        N = len(qs)
        phase = out["phase"]
        feet_contact = np.ones((N, 2))
        feet_contact[phase == 1, 0 if swing == "left" else 1] = 0.0
        com_vel = np.concatenate([out["com_vel"], np.zeros((N, 1))], axis=1)
        orb = out["orbit"]

        meta = {
            "task": "foot_lip", "mode": "orbit", "version": 1, "dt": dt,
            "seed": seed, "idx": idx, "attempt": attempt,
            "cell": {"swing": swing_i, "x": xi, "dy": dyi, "yaw": yi,
                     "dprev_x": pxi, "dprev_y": pyi},
            "swing": swing, "stance": stance,
            "landing": [x, dy, yaw], "prev_step": list(prev),
            "liftoff": list(lift),
            "mirror": mirror, "T": T, "T_step": T, "lift_frac": f,
            "z_orbit": cfg["z_orbit"], "apex": apex,
            "orbit": {"lam": orb["lam"], "ell": orb["ell"], "w": orb["w"],
                      "p_iss": orb["p_iss"].tolist(), "v_iss": orb["v_iss"].tolist()},
            "width0": float(lipm.nominal_foot_y_distance),
            "com_z": float(lipm.com0[2]), "zc": float(lipm.zc),
            "feet_order": ["left", "right"], "quat": "xyzw",
            "kinematic_only": True, "refined": bool(_W["refine"]),
            "boundary_shift": out["zmp_boundary_shift"].tolist(),
            "metrics": {
                "ik_foot_err": foot_err, "ik_com_err": com_err,
                "min_gap": float(min_gap),
                "dsp_hull_margin": [m1, m2],
                "zmp_mb_excess_xy": zmp_excess.tolist(),
                **({"zmp_refine_iters": out["zmp_refine"]["iters"],
                    "zmp_err_max": out["zmp_refine"]["zmp_err_max"]}
                   if _W["refine"] else {}),
            },
        }
        np.savez_compressed(
            Path(out_dir) / f"fl_{idx:05d}.npz",
            meta=json.dumps(meta),
            q=qs.astype(np.float32), v=vs.astype(np.float32),
            q_rest_start=q_rest_start.astype(np.float32),
            q_rest=q_rest.astype(np.float32),
            com_pos=out["com"].astype(np.float32),
            com_vel=com_vel.astype(np.float32),
            feet_pos_w=feet_pos.astype(np.float32),
            feet_quat_w=feet_quat.astype(np.float32),
            feet_vel_w=feet_vel.astype(np.float32),
            feet_contact=feet_contact.astype(np.float32),
            countdown=(1.0 - out["t"] / T).astype(np.float32),
            zmp_ref=out["zmp_ref"].astype(np.float32),
            zmp_multibody=zmp_mb.astype(np.float32),
            phase=phase.astype(np.int8),
        )
        return idx, meta, None
    return idx, None, reason


def _build_clip(task):
    """Generate + gate one clip. Returns (idx, meta_dict | None, reject_reason)."""
    idx, out_dir, seed, cell = task
    lipm, checker = _W["lipm"], _W["checker"]
    cfg = CFG
    dt = cfg["dt"]
    g = cfg["grid"]
    swing_i, xi, dyi, yi, ti = cell

    for attempt in range(cfg["attempts_per_slot"]):
        rng = np.random.default_rng([seed, idx, attempt])
        # gridded dims: jitter inside the cell; liftoff/apex full-range
        x = _sub(*cfg["landing"]["x"], g["x_cells"], xi, rng.uniform())
        dy = _sub(*cfg["landing"]["dy"], g["dy_cells"], dyi, rng.uniform())
        yaw = _sub(*cfg["landing"]["yaw"], g["yaw_cells"], yi, rng.uniform())
        T = _sub(*cfg["tempo"]["T"], g["tempo_bins"], ti, rng.uniform())
        lift = (rng.uniform(*cfg["liftoff"]["dx"]),
                rng.uniform(*cfg["liftoff"]["dy"]),
                rng.uniform(*cfg["liftoff"]["yaw"]))
        apex = rng.uniform(*cfg["apex"])

        swing = ("left", "right")[swing_i]
        stance = ("right", "left")[swing_i]
        mirror = -1.0 if swing == "left" else 1.0  # canonical = right swing
        fr = cfg["tempo"]["fractions"]
        wp = WalkPrimitive(
            stance_foot=f"{stance}_foot", swing_foot=f"{swing}_foot",
            swing_start_pose=np.eye(4),
            step_dx=x, step_dy=dy, step_dyaw=mirror * yaw,
            step_Tsupport=fr["swing"] * T, swing_height=apex,
            Tdsp_start=fr["ds_pre"] * T, Tdsp_end=fr["ds_post"] * T,
            lift_dx=lift[0], lift_dy=lift[1], lift_dyaw=mirror * lift[2],
        )

        out = lipm.generate_com_trajectory(wp, dt=dt)
        # gate 1: shifted DSP ZMP levels inside the phase support hulls
        m1 = _hull_margin([out["stance_pose"], out["swing_start"]],
                          out["zmp_ref"][0])
        m2 = _hull_margin([out["stance_pose"], out["swing_goal"]],
                          out["zmp_ref"][-1])
        margin = -cfg["gates"]["dsp_hull_margin_m"]
        if m1 > margin or m2 > margin:
            reason = "dsp_zmp_outside_hull"
            continue

        ft = lipm.generate_swing_trajectory(out, wp)
        qs = lipm.refine_with_multibody_zmp(out, ft, dt=dt, max_iters=12,
                                            tol=cfg["gates"]["zmp_err_m"])
        # gate 2: refinement converged
        if not out["zmp_refine"]["converged"]:
            reason = "zmp_refine_diverged"
            continue

        # gate 3: IK residuals on the final qs
        model, data = lipm.robot.model, lipm.robot.data
        N = len(qs)
        foot_err = com_err = 0.0
        for i in range(N):
            pin.forwardKinematics(model, data, qs[i])
            pin.updateFramePlacements(model, data)
            com_err = max(com_err, float(np.linalg.norm(
                pin.centerOfMass(model, data, qs[i])[:2] - out["com"][i, :2])))
            for name in ("left_foot", "right_foot"):
                M = data.oMf[model.getFrameId(name)]
                foot_err = max(foot_err, float(np.linalg.norm(
                    ft[name][i][:3, 3] - M.translation)))
        if foot_err > cfg["gates"]["ik_foot_m"] or com_err > cfg["gates"]["ik_com_m"]:
            reason = "ik_residual"
            continue

        # gate 4 (most expensive, LAST): leg-vs-leg mesh distance
        min_gap = checker.min_gap(qs, stride=2)
        if min_gap < cfg["gates"]["min_gap_m"]:
            reason = "leg_collision"
            continue

        # ---- channels (kinematic only), v3-compatible names/layout ----
        vs = np.stack([pin.difference(model, qs[i], qs[min(i + 1, N - 1)]) / dt
                       for i in range(N)])
        vs[-1] = vs[-2]
        feet_pos = np.stack([ft["left_foot"][:, :3, 3],
                             ft["right_foot"][:, :3, 3]], axis=1)
        feet_quat = np.zeros((N, 2, 4))
        feet_yaw = np.zeros((N, 2))
        for k, name in enumerate(("left_foot", "right_foot")):
            for i in range(N):
                R = ft[name][i][:3, :3]
                feet_quat[i, k] = pin.Quaternion(R).coeffs()  # xyzw
                feet_yaw[i, k] = np.arctan2(R[1, 0], R[0, 0])
        feet_vel = np.zeros((N, 2, 6))
        feet_vel[:, :, :3] = np.gradient(feet_pos, dt, axis=0)
        feet_vel[:, :, 5] = np.gradient(np.unwrap(feet_yaw, axis=0), dt, axis=0)
        phase = out["phase"]
        feet_contact = np.ones((N, 2))
        swing_col = 0 if swing == "left" else 1
        feet_contact[phase == 1, swing_col] = 0.0
        com_pos = out["com"].copy()
        com_vel = np.concatenate([out["com_vel"], np.zeros((N, 1))], axis=1)

        meta = {
            "task": "foot_lip", "version": 1, "dt": dt, "seed": seed,
            "idx": idx, "attempt": attempt,
            "cell": {"swing": swing_i, "x": xi, "dy": dyi, "yaw": yi,
                     "tempo": ti},
            "swing": swing, "stance": stance,
            "landing": [x, dy, yaw], "liftoff": list(lift),
            "mirror": mirror, "T": T, "fractions": fr, "apex": apex,
            "width0": float(lipm.nominal_foot_y_distance),
            "com_z": float(lipm.com0[2]), "zc": float(lipm.zc),
            "feet_order": ["left", "right"], "quat": "xyzw",
            "kinematic_only": True,
            "boundary_shift": out["zmp_boundary_shift"].tolist(),
            "metrics": {
                "zmp_refine_iters": out["zmp_refine"]["iters"],
                "zmp_err_max": out["zmp_refine"]["zmp_err_max"],
                "ik_foot_err": foot_err, "ik_com_err": com_err,
                "min_gap": float(min_gap),
                "dsp_hull_margin": [m1, m2],
            },
        }
        np.savez_compressed(
            Path(out_dir) / f"fl_{idx:05d}.npz",
            meta=json.dumps(meta),
            q=qs.astype(np.float32), v=vs.astype(np.float32),
            com_pos=com_pos.astype(np.float32),
            com_vel=com_vel.astype(np.float32),
            feet_pos_w=feet_pos.astype(np.float32),
            feet_quat_w=feet_quat.astype(np.float32),
            feet_vel_w=feet_vel.astype(np.float32),
            feet_contact=feet_contact.astype(np.float32),
            countdown=np.linspace(1.0, 0.0, N).astype(np.float32),
            zmp_ref=out["zmp_ref"].astype(np.float32),
            phase=phase.astype(np.int8),
        )
        return idx, meta, None
    return idx, None, reason


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/foot_lip_v1")
    ap.add_argument("--n-per-cell", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total clips (smoke runs)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mode", default="rest", choices=["rest", "orbit"],
                    help="rest = rest-to-rest clips (CFG); orbit = steady-state "
                         "H-LIP steps on the FootHoldCommand clock (ORBIT_CFG)")
    ap.add_argument("--refine", action="store_true",
                    help="orbit mode: apply the 4.70 multibody-ZMP refinement "
                         "(moves the CoM off the orbit)")
    args = ap.parse_args()
    orbit = args.mode == "orbit"
    build = _build_orbit_clip if orbit else _build_clip
    cfg = ORBIT_CFG if orbit else CFG

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"REFUSING to write into non-empty {out_dir} -- "
                 "pick a fresh directory (dataset safety rule).")
    out_dir.mkdir(parents=True, exist_ok=True)

    g = cfg["grid"]
    if orbit:
        cells = [(s, xi, dyi, yi, pxi, pyi)
                 for s in range(2) for xi in range(g["x_cells"])
                 for dyi in range(g["dy_cells"]) for yi in range(g["yaw_cells"])
                 for pxi in range(g["dprev_x_cells"]) for pyi in range(g["dprev_y_cells"])]
    else:
        cells = [(s, xi, dyi, yi, ti)
                 for s in range(2) for xi in range(g["x_cells"])
                 for dyi in range(g["dy_cells"]) for yi in range(g["yaw_cells"])
                 for ti in range(g["tempo_bins"])]
    tasks = [(i, str(out_dir), args.seed, cells[i % len(cells)])
             for i in range(len(cells) * args.n_per_cell)]
    if args.limit:
        # evenly strided subset so a smoke run still spans the whole grid
        stride = max(1, len(tasks) // args.limit)
        tasks = tasks[::stride][:args.limit]
    print(f"{len(cells)} cells x {args.n_per_cell}/cell -> {len(tasks)} clips "
          f"-> {out_dir}  ({args.workers} workers)")

    t0 = time.time()
    rejects, cell_rejects, n_ok = {}, {}, 0
    if args.workers > 1:
        with Pool(args.workers, initializer=_init_worker,
                  initargs=(args.refine,)) as pool:
            results = pool.imap_unordered(build, tasks, chunksize=8)
            for k, (idx, meta, reason) in enumerate(results):
                if meta is None:
                    rejects[reason] = rejects.get(reason, 0) + 1
                    key = str(cells[idx % len(cells)])
                    cell_rejects[key] = cell_rejects.get(key, 0) + 1
                else:
                    n_ok += 1
                if (k + 1) % 200 == 0:
                    print(f"  {k+1}/{len(tasks)}  ok {n_ok}  rej {sum(rejects.values())}"
                          f"  ({time.time()-t0:.0f}s)")
    else:
        _init_worker(args.refine)
        for k, task in enumerate(tasks):
            idx, meta, reason = build(task)
            if meta is None:
                rejects[reason] = rejects.get(reason, 0) + 1
                key = str(cells[idx % len(cells)])
                cell_rejects[key] = cell_rejects.get(key, 0) + 1
            else:
                n_ok += 1
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(tasks)}  ok {n_ok}  rej {sum(rejects.values())}")

    report = {
        "mode": args.mode, "refine": args.refine,
        "cfg": cfg, "n_tasks": len(tasks), "n_ok": n_ok,
        "rejects_by_reason": rejects,
        "reject_rate": 1.0 - n_ok / max(len(tasks), 1),
        "worst_cells": dict(sorted(cell_rejects.items(),
                                   key=lambda kv: -kv[1])[:20]),
        "wall_s": time.time() - t0,
    }
    (out_dir / "gen_report.json").write_text(json.dumps(report, indent=2))
    print(f"DONE: {n_ok}/{len(tasks)} accepted "
          f"({100*report['reject_rate']:.1f}% rejected) in {report['wall_s']:.0f}s")
    print("rejects by reason:", rejects)


if __name__ == "__main__":
    main()
