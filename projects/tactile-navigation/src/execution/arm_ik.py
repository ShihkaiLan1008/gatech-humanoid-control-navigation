"""Arm-joint anchors for tactile-tip targets, on the G1 tactile model.

The upper policy in both controllers mimics a joint-space ANCHOR while it
tracks a tip; a tip with no matching anchor makes it fight itself (see
`vforce_backend.CARRY_ANCHOR_Q`).  The room2 postures were solved offline;
the demo room derives its strikes from the geometry in front of the robot,
so the same damped-least-squares solve runs here, once per distinct pose.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw")
"""The vforce reach-pool column order; `anchor_q` is aligned to it."""

G1_XML = (Path(__file__).resolve().parents[2] / "tactile_controller" / "src"
          / "assets" / "g1_description" / "g1_tactile.xml")

SEED_FORWARD = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
"""HOME posture: upper arm hanging, elbow 90 deg, forearm forward -- the
carry.  The right seed for anything in front of the robot."""


@functools.lru_cache(maxsize=1)
def _model():
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(G1_XML))
    return m, mujoco.MjData(m)


def _fk(side: str, q) -> tuple[np.ndarray, np.ndarray]:
    import mujoco
    m, d = _model()
    mujoco.mj_resetData(m, d)
    for j, v in zip(ARM_JOINTS, q):
        d.qpos[m.jnt_qposadr[m.joint(f"{side}_{j}_joint").id]] = float(v)
    mujoco.mj_kinematics(m, d)
    pelvis, tip = m.body("pelvis").id, m.body(f"{side}_tactile").id
    R = d.xmat[pelvis].reshape(3, 3)
    p = R.T @ (d.xpos[tip] - d.xpos[pelvis])
    ex = R.T @ d.xmat[tip].reshape(3, 3)[:, 0]
    return p, ex


def _limits(side: str) -> np.ndarray:
    m, _ = _model()
    return np.array([m.jnt_range[m.joint(f"{side}_{j}_joint").id]
                     for j in ARM_JOINTS])


@functools.lru_cache(maxsize=256)
def solve_anchor(side: str, pos: tuple, yaw: float,
                 seed: tuple = SEED_FORWARD, *, w_dir: float = 0.3,
                 iters: int = 200) -> tuple:
    """7 arm joints that put the tip at ``pos`` (pelvis frame) with its +x
    axis at ``yaw`` about z.  Position-weighted DLS with a weak pull to the
    seed posture, clamped to the joint limits."""
    target = np.asarray(pos, float)
    ex_des = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    q = np.array(seed, float)
    lo, hi = _limits(side).T
    seed_q = np.array(seed, float)
    for _ in range(iters):
        p, ex = _fk(side, q)
        e = np.concatenate([target - p, w_dir * (ex_des - ex)])
        if np.linalg.norm(e[:3]) < 1e-3 and np.linalg.norm(e[3:]) < 1e-2:
            break
        J = np.zeros((6, 7))
        h = 1e-5
        for k in range(7):
            qq = q.copy()
            qq[k] += h
            pk, exk = _fk(side, qq)
            J[:3, k] = (pk - p) / h
            J[3:, k] = w_dir * (exk - ex) / h
        dq = J.T @ np.linalg.solve(J @ J.T + 0.05 ** 2 * np.eye(6), e)
        dq += 0.02 * (seed_q - q)
        q = np.clip(q + 0.7 * dq, lo, hi)
    return tuple(round(float(v), 4) for v in q)


def tip_error_m(side: str, pos, yaw: float, q) -> float:
    p, _ = _fk(side, q)
    return float(np.linalg.norm(p - np.asarray(pos, float)))
