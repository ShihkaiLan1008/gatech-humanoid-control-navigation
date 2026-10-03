"""G1 loading + stance IK for the foot_placement task.

Fresh minimal rewrite of mpc_offline.utils.load_robot: stock vendor URDF
(g1_29dof_rev_1_0 -- user decision 2026-08-07: the lower-body dataset decouples
from the tactile asset; verified against it: same 29-joint order, same total
mass 33.3411 kg, half-sitting CoM within 0.03 mm), same half-sitting reference
pose, plus a small damped-least-squares IK that realizes a sampled initial
stance (both feet at given ground poses) using only the free-flyer + leg
columns, so arms/waist stay at the reference pose.
"""

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

import assets

LEFT_FOOT = "left_ankle_roll_link"
RIGHT_FOOT = "right_ankle_roll_link"

LEG_JOINTS = [
    f"{side}_{j}_joint"
    for side in ("left", "right")
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
]
ARM_JOINTS = [
    f"{side}_{j}_joint"
    for side in ("left", "right")
    for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw")
]

# base height + bent-knee pose (same values as mpc_offline's half_sitting)
_BASE_HEIGHT = 0.793
_HALF_SITTING = {
    "left_hip_pitch_joint": -0.10,
    "left_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.20,
    "right_hip_pitch_joint": -0.10,
    "right_knee_joint": 0.30,
    "right_ankle_pitch_joint": -0.20,
}


def load_g1():
    """RobotWrapper on the stock rev_1_0 URDF, free-flyer base, half-sitting q0."""
    robot = RobotWrapper.BuildFromURDF(
        str(assets.ASSET_DIR / assets.REV_1_0_URDF), str(assets.ASSET_DIR),
        pin.JointModelFreeFlyer())
    q0 = pin.neutral(robot.model)
    q0[2] = _BASE_HEIGHT
    for name, value in _HALF_SITTING.items():
        j = robot.model.joints[robot.model.getJointId(name)]
        q0[j.idx_q] = value
    robot.q0 = q0
    robot.model.referenceConfigurations["half_sitting"] = q0
    return robot


def foot_frames(model):
    return model.getFrameId(LEFT_FOOT), model.getFrameId(RIGHT_FOOT)


def nominal_stance(robot):
    """(left_pos, right_pos, width, ground_z, com_z) at the half-sitting pose."""
    model = robot.model
    data = model.createData()
    q0 = robot.q0
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    lf, rf = foot_frames(model)
    lp = data.oMf[lf].translation.copy()
    rp = data.oMf[rf].translation.copy()
    com_z = float(pin.centerOfMass(model, data, q0)[2])
    return lp, rp, float(lp[1] - rp[1]), float(min(lp[2], rp[2])), com_z


def _yaw_se3(yaw, pos):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return pin.SE3(R, np.asarray(pos, float))


def stance_ik(robot, targets, q_init=None, iters=200, tol=1e-8, damp=1e-6):
    """Realize both-feet SE3 `targets` ({frame_id: SE3}) from `q_init` by DLS on
    the free-flyer + leg columns only. Returns (q, residual_norm)."""
    model = robot.model
    data = model.createData()
    q = (robot.q0 if q_init is None else q_init).copy()
    cols = list(range(6))
    for name in LEG_JOINTS:
        cols.append(model.joints[model.getJointId(name)].idx_v)
    cols = np.asarray(cols, int)
    err_norm = np.inf
    for _ in range(iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        errs, jacs = [], []
        for fid, M in targets.items():
            dM = M.actInv(data.oMf[fid])
            errs.append(pin.log6(dM).vector)
            jacs.append(pin.computeFrameJacobian(model, data, q, fid, pin.LOCAL))
        e = np.concatenate(errs)
        err_norm = float(np.linalg.norm(e))
        if err_norm < tol:
            break
        J = np.vstack(jacs)[:, cols]
        dq_c = -J.T @ np.linalg.solve(J @ J.T + damp * np.eye(J.shape[0]), e)
        dq = np.zeros(model.nv)
        dq[cols] = dq_c
        q = pin.integrate(model, q, dq)
    return q, err_norm
