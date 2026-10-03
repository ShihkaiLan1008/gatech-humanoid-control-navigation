"""Per-dataset ingest: raw file -> `RawClip`, the one intermediate the converter
understands.

Adding a dataset means adding a module here that can (a) enumerate its raw files
and (b) turn one into a `RawClip`.  Nothing else in the pipeline changes: FK,
resampling, the ground fix and the output schema are all source-agnostic.

Every source must deliver the SAME 29-joint order as `assets.TACTILE_XML`
(left_hip_pitch ... right_wrist_yaw).  That is checked at convert time against
`G1Kinematics.joint_names`, not assumed -- a silently permuted source would
produce clips that look fine and animate wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RawClip:
    """One raw retargeted clip, before FK / resampling / the ground fix."""

    root_pos: np.ndarray        # (T, 3) -- z is source-relative, NOT world
    root_rot_xyzw: np.ndarray   # (T, 4)
    dof_pos: np.ndarray         # (T, 29)
    fps: float
    rel_path: str               # relative to raw/<source>/, for provenance

    def __post_init__(self) -> None:
        T = self.root_pos.shape[0]
        if self.root_rot_xyzw.shape != (T, 4) or self.dof_pos.shape[0] != T:
            raise ValueError(f"{self.rel_path}: inconsistent frame counts")
        if self.fps <= 0:
            raise ValueError(f"{self.rel_path}: bad fps {self.fps}")


def split_36(arr: np.ndarray, rel_path: str) -> tuple[np.ndarray, ...]:
    """Both G1 sources use the identical `[T, 36]` row layout:
    ``root_pos(3) | root_quat XYZW(4) | 29 joint angles``.
    Returns (root_pos, root_rot_xyzw, dof_pos)."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 36:
        raise ValueError(f"{rel_path}: expected [T,36], got {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"{rel_path}: {arr.shape[0]} frame(s), too short to use")
    root_pos, quat, dof = arr[:, 0:3], arr[:, 3:7], arr[:, 7:36]
    n = np.linalg.norm(quat, axis=1)
    if not np.allclose(n, 1.0, atol=1e-3):
        # Normalise rather than reject: a few source clips drift ~1e-4 and SLERP
        # needs unit input. Anything further off is a real problem, so shout.
        if not np.allclose(n, 1.0, atol=5e-2):
            raise ValueError(f"{rel_path}: root quaternion norms {n.min():.4f}"
                             f"..{n.max():.4f}, not a rotation")
        quat = quat / n[:, None]
    return root_pos, quat, dof


def relpath_under(root: Path, path: Path) -> str:
    """`path` relative to `root`, with the upstream `g1/` prefix stripped.

    Both Hub repos nest everything under `g1/` (their per-robot folder); keeping
    it would put a pointless `g1/` level in our output tree and break the
    `<root>/AMASS/ACCAD/...` layout the reference consumer expects."""
    rel = path.relative_to(root)
    parts = rel.parts
    if parts and parts[0] == "g1":
        rel = Path(*parts[1:])
    return str(rel)


__all__ = ["RawClip", "relpath_under", "split_36"]
