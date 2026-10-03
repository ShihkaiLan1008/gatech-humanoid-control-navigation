"""Frame-rate conversion for motion clips.

Matches the reference implementation (Axellwppr/GMR
`scripts/bvh_to_robot_dataset.py::resample_motion_to_fps`) so our clips are
interchangeable with theirs:

* rotations  -> SLERP (scipy `Slerp`); linear interpolation of raw quaternion
                components would shrink the norm and bend the path.
* everything -> PCHIP (shape-preserving cubic).  Plain cubic splines overshoot,
  else          which on a joint angle near a limit produces a pose the robot
                cannot reach; PCHIP cannot overshoot between samples.
* joint angles are UNWRAPPED first: a clip that crosses +/-pi would otherwise be
  interpolated the long way round, injecting a ~2pi spike over one frame.

This matters here because the raw sources arrive at SIX different frame rates
(measured over the 17,717 AMASS clips: 120 / 100 / 60 / 250 / 59 / 150 Hz), so
resampling is the rule, not an edge case.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp


def _pchip(t_src: np.ndarray, x_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    """(N, D) -> (M, D), PCHIP per column."""
    if x_src.shape[0] == 1:
        return np.repeat(x_src, len(t_tgt), axis=0)
    return np.stack(
        [PchipInterpolator(t_src, x_src[:, k])(t_tgt) for k in range(x_src.shape[1])],
        axis=1,
    )


def _slerp_xyzw(t_src: np.ndarray, q_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    """(N, 4) XYZW -> (M, 4) XYZW."""
    if q_src.shape[0] == 1:
        return np.repeat(q_src, len(t_tgt), axis=0)
    return Slerp(t_src, Rotation.from_quat(q_src))(t_tgt).as_quat()


def resample(
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    src_fps: float,
    tgt_fps: float,
    unwrap_angles: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a clip to `tgt_fps`. Returns (root_pos, root_rot_xyzw, dof_pos).

    The duration is preserved exactly; the frame count becomes
    round(duration * tgt_fps) + 1, i.e. both endpoints are kept.  A no-op
    (`src_fps == tgt_fps`) returns the inputs untouched rather than round-tripping
    them through the interpolators.
    """
    T = root_pos.shape[0]
    if T == 0:
        raise ValueError("empty clip")
    if abs(float(src_fps) - float(tgt_fps)) < 1e-9:
        return root_pos, root_rot_xyzw, dof_pos

    duration = (T - 1) / float(src_fps) if T > 1 else 0.0
    M = int(np.round(duration * float(tgt_fps))) + 1 if duration > 0 else 1
    t_src = np.linspace(0.0, duration, T)
    t_tgt = np.linspace(0.0, duration, M)

    dof_src = np.unwrap(dof_pos, axis=0) if (unwrap_angles and T > 1) else dof_pos
    return (
        _pchip(t_src, root_pos, t_tgt),
        _slerp_xyzw(t_src, root_rot_xyzw, t_tgt),
        _pchip(t_src, dof_src, t_tgt),
    )


__all__ = ["resample"]
