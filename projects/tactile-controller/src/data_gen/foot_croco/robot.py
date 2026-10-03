"""G1 loading for the foot_croco task: rev_1_0 URDF, KNEES_BENT rest pose
(decree 2026-08-19), sole-centre contact frames.

Reuses `data_gen.common.robot` for the URDF load and the stance IK; adds the
two `left_foot` / `right_foot` OP frames on the soles (ankle_roll frame +
(0.04, 0, -0.035), the MJCF site placement) — contacts, wrench cones, foot
targets and the emitted feet channels all live on these frames, so the box is
about the sole centre and the cone acts at the contact plane.
"""

import numpy as np
import pinocchio as pin

from data_gen.common import robot as _base

# ankle_roll frame -> sole-centre site, in the ankle frame (MJCF site pos).
SITE_IN_ANKLE = np.array([0.04, 0.0, -0.035])
# sole-centre -> toe contact point, along the sole x (toe edge is +0.092).
TOE_IN_SITE = 0.09

# KNEES_BENT keyframe (rl.g1_assets.g1_constants; HOME/half-sitting is BANNED
# for new work).
KNEES_BENT_BASE_HEIGHT = 0.757
KNEES_BENT = {
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
}

stance_ik = _base.stance_ik
_yaw_se3 = _base._yaw_se3
LEG_JOINTS = _base.LEG_JOINTS
ARM_JOINTS = _base.ARM_JOINTS


def load_g1():
    """RobotWrapper on the rev_1_0 URDF, free-flyer base, KNEES_BENT q0, with
    `left_foot` / `right_foot` sole frames added. Side effect: rebuilds
    robot.data (frames added after load)."""
    robot = _base.load_g1()
    model = robot.model
    q0 = pin.neutral(model)
    q0[2] = KNEES_BENT_BASE_HEIGHT
    for name, value in KNEES_BENT.items():
        q0[model.joints[model.getJointId(name)].idx_q] = value
    robot.q0 = q0
    model.referenceConfigurations["knees_bent"] = q0

    for side in ("left", "right"):
        ankle = model.frames[model.getFrameId(f"{side}_ankle_roll_link")]
        placement = ankle.placement * pin.SE3(np.eye(3), SITE_IN_ANKLE)
        model.addFrame(pin.Frame(f"{side}_foot", ankle.parentJoint,
                                 ankle.parentFrame, placement,
                                 pin.FrameType.OP_FRAME))
        toe = placement * pin.SE3(np.eye(3), np.array([TOE_IN_SITE, 0.0, 0.0]))
        model.addFrame(pin.Frame(f"{side}_toe", ankle.parentJoint,
                                 ankle.parentFrame, toe,
                                 pin.FrameType.OP_FRAME))
    robot.data = model.createData()
    return robot


def foot_frames(model):
    """(left_foot_fid, right_foot_fid) — the SOLE frames, not the ankles."""
    return model.getFrameId("left_foot"), model.getFrameId("right_foot")


def nominal_stance(robot):
    """(left_site, right_site, width, site_ground_z, com_z, com_offset_xy) at
    the KNEES_BENT pose. `com_offset_xy` is the natural CoM xy bias w.r.t. the
    mid-feet SITE point — rest anchors must sit at the CoM projection, not the
    geometric midpoint (the foot_placement_v3 lesson)."""
    model, data = robot.model, robot.data
    q0 = robot.q0
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    lf, rf = foot_frames(model)
    lp = data.oMf[lf].translation.copy()
    rp = data.oMf[rf].translation.copy()
    com = pin.centerOfMass(model, data, q0).copy()
    com_offset_xy = com[:2] - 0.5 * (lp[:2] + rp[:2])
    return (lp, rp, float(lp[1] - rp[1]), float(min(lp[2], rp[2])),
            float(com[2]), com_offset_xy)


def perturbed_model(model, torso_com_dx: float):
    """Copy of `model` with the torso_link body CoM shifted by `torso_com_dx`
    along its local x — the measured real-robot signature (harness + sensors).
    Used by the robustness gate; total mass unchanged."""
    m2 = model.copy()
    jid = m2.frames[m2.getFrameId("torso_link")].parentJoint
    inertia = m2.inertias[jid]
    lever = inertia.lever.copy()
    lever[0] += torso_com_dx
    m2.inertias[jid] = pin.Inertia(inertia.mass, lever, inertia.inertia)
    return m2
