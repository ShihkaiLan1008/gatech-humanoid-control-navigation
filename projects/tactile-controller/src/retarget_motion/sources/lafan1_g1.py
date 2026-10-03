"""LAFAN1 retargeted to the G1 -- `lvhaidong/LAFAN1_Retargeting_Dataset`
(a public mirror of Unitree's release).

40 clips: walk / run / sprint / jumps / dance / fight / fightAndSports /
fallAndGetUp, across 5 subjects.  Retargeted by Unitree with Interaction-Mesh +
IK under end-effector, joint-position and joint-velocity constraints, which is
why the feet do not slip.  Kinematic constraints only -- no dynamics, no
actuator limits -- so the robot cannot execute these perfectly, which is exactly
what the downstream tracking policy is for.

Layout:  raw/LAFAN/g1/<name>.csv
Payload: CSV, no header, one row per frame, **30 fps fixed** (stated on the
         dataset card; unlike AMASS there is no per-file rate), same [T, 36]
         root-pos / XYZW-quat / 29-joint column order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from retarget_motion import paths
from retarget_motion.sources import RawClip, relpath_under, split_36

SOURCE = paths.LAFAN
FPS = 30.0  # fixed for this dataset (dataset card: "G1: (30 FPS)")


def root_dir() -> Path:
    return paths.RAW_DIR / SOURCE


def find_files(root: Path | None = None) -> list[Path]:
    root = root or root_dir()
    return sorted(root.rglob("*.csv"))


def load(path: Path, root: Path | None = None) -> RawClip:
    root = root or root_dir()
    rel = relpath_under(root, path)
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    root_pos, quat, dof = split_36(arr, rel)
    return RawClip(root_pos=root_pos, root_rot_xyzw=quat, dof_pos=dof,
                   fps=FPS, rel_path=rel)


__all__ = ["FPS", "SOURCE", "find_files", "load", "root_dir"]
