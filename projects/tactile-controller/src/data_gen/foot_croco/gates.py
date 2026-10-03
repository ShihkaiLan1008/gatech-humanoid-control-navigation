"""Post-solve gates: ZMP-in-support-polygon on the nominal AND a CoM-perturbed
model, and the touchdown-velocity check. All from (q, v, dv) via centroidal
dynamics — no re-solve.
"""

import numpy as np
import pinocchio as pin
from scipy.spatial import ConvexHull

# TRUE support polygon about the SOLE frame [m] (measured, ankle-frame
# x [-0.054, +0.132] shifted by the site's +0.04).
TRUE_BOX_X = (-0.094, 0.092)
TRUE_BOX_Y = (-0.026, 0.026)


def zmp_series(model, q: np.ndarray, v: np.ndarray, dv: np.ndarray,
               ground_z: float) -> np.ndarray:
    """ZMP trajectory on the plane z = `ground_z` from centroidal dynamics.
    Shape (F, 2), world frame, m. `dv` is the acceleration at each frame
    (solver xout; last row may repeat)."""
    data = model.createData()
    g = 9.81
    m = pin.computeTotalMass(model)
    out = np.zeros((len(q), 2))
    for k in range(len(q)):
        pin.computeCentroidalMomentumTimeVariation(model, data, q[k], v[k], dv[k])
        c = pin.centerOfMass(model, data, q[k])
        f = np.asarray(data.dhg.linear) + np.array([0.0, 0.0, m * g])
        dH = np.asarray(data.dhg.angular)          # about the CoM
        fz = max(f[2], 1e-6)
        cz = c[2] - ground_z
        out[k, 0] = c[0] - (cz * f[0] + dH[1]) / fz
        out[k, 1] = c[1] + (dH[0] - cz * f[1]) / fz
    return out


def support_margin_series(feet_pos: np.ndarray, feet_yaw: np.ndarray,
                          contact: np.ndarray, zmp: np.ndarray,
                          push_mask: np.ndarray | None = None) -> np.ndarray:
    """Signed distance of the ZMP to the support hull per frame (negative =
    inside by that much). Hull = union of the TRUE sole boxes of the feet in
    contact; a foot flagged in `push_mask` (toe-only contact) contributes just
    a small patch around its toe. Shapes: feet_pos (F, 2, 3), feet_yaw (F, 2),
    contact (F, 2), zmp (F, 2) -> (F,)."""
    corners_local = np.array([[x, y] for x in TRUE_BOX_X for y in TRUE_BOX_Y])
    toe_local = np.array([[x, y] for x in (0.08, 0.092) for y in (-0.01, 0.01)])
    out = np.zeros(len(zmp))
    for k in range(len(zmp)):
        pts = []
        for i in range(2):
            if contact[k, i] < 0.5:
                continue
            c, s = np.cos(feet_yaw[k, i]), np.sin(feet_yaw[k, i])
            R = np.array([[c, -s], [s, c]])
            local = corners_local
            if push_mask is not None and push_mask[k, i] > 0.5:
                local = toe_local
            pts.append(feet_pos[k, i, :2] + local @ R.T)
        pts = np.concatenate(pts)
        hull = ConvexHull(pts)
        out[k] = float(np.max(hull.equations[:, :2] @ zmp[k] + hull.equations[:, 2]))
    return out


def _feet_yaw(ch: dict) -> np.ndarray:
    """Per-frame yaw of both sole frames from feet_quat_w (xyzw). Shape (F, 2), rad."""
    qf = ch["feet_quat_w"]
    return np.arctan2(
        2 * (qf[..., 3] * qf[..., 2] + qf[..., 0] * qf[..., 1]),
        1 - 2 * (qf[..., 1] ** 2 + qf[..., 2] ** 2))


def margin_for_model(model, ch: dict, ground_z: float) -> tuple[np.ndarray, np.ndarray]:
    """(zmp (F, 2), signed support margin (F,)) of one clip on `model`
    (negative margin = inside). One centroidal recompute — no re-solve."""
    zmp = zmp_series(model, ch["q"], ch["v"], ch["dv"], ground_z)
    m = support_margin_series(ch["feet_pos_w"], _feet_yaw(ch), ch["feet_contact"],
                              zmp, ch.get("push_mask"))
    return zmp, m


def dcm_liftoff(model, ch: dict, ground_z: float) -> dict:
    """DCM crossing certificate at the liftoff frame (first SSP frame), in the
    stance sole's yaw frame. `margin` [m] is the distance of the DCM to the
    sole edge it must clear for the direction the CoM actually travels
    (positive = the step CAN be completed with CoP inside the sole; negative =
    guaranteed fall toward the departed foot). Near-zero-travel clips use the
    tighter of both edges. Also returns gap (com−sole x), v (com vx), xi_x."""
    phase = np.asarray(ch["phase"])
    lo = int(np.argmax(phase == 1))
    if phase[lo] != 1:
        return dict(margin=float("nan"), gap=0.0, v=0.0, xi_x=0.0)
    st = int(np.argmax(ch["feet_contact"][lo]))          # the foot still down
    data = model.createData()
    c = pin.centerOfMass(model, data, np.asarray(ch["q"][lo], dtype=np.float64),
                         np.asarray(ch["v"][lo], dtype=np.float64))
    vc = data.vcom[0].copy()
    om = np.sqrt(9.81 / max(c[2] - ground_z, 1e-3))
    yaw = float(_feet_yaw(ch)[lo, st])
    cs, sn = np.cos(yaw), np.sin(yaw)
    rel = c[:2] - ch["feet_pos_w"][lo, st, :2]
    gap = cs * rel[0] + sn * rel[1]
    v = cs * vc[0] + sn * vc[1]
    xi = gap + v / om
    c_end = pin.centerOfMass(model, model.createData(),
                             np.asarray(ch["q"][-1], dtype=np.float64))
    travel = cs * (c_end[0] - c[0]) + sn * (c_end[1] - c[1])
    fwd, back = xi - TRUE_BOX_X[0], TRUE_BOX_X[1] - xi
    if travel > 0.02:
        margin = fwd
    elif travel < -0.02:
        margin = back
    else:
        margin = min(fwd, back)
    return dict(margin=float(margin), gap=float(gap), v=float(v), xi_x=float(xi))


def gate_report(model, model_shifted, ch: dict, ground_z: float) -> dict:
    """ZMP margins on the nominal and CoM-shifted model over one clip's
    channels (`ch` from StepProblem.channels; feet already grounded, so
    `ground_z` here is the SAVED sole ground). Returns per-model worst margins,
    the full nominal zmp series and both margin series."""
    zmp_n, m_n = margin_for_model(model, ch, ground_z)
    _, m_s = margin_for_model(model_shifted, ch, ground_z)
    ssp = ch["phase"] == 1
    return dict(
        zmp_nominal=zmp_n,
        margin_nominal=m_n, margin_shifted=m_s,
        worst_nominal=float(m_n.max()), worst_shifted=float(m_s.max()),
        worst_nominal_ssp=float(m_n[ssp].max()) if ssp.any() else float("nan"),
        worst_shifted_ssp=float(m_s[ssp].max()) if ssp.any() else float("nan"),
    )
