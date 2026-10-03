"""AMASS retargeted to the G1 -- `fleaven/Retargeted_AMASS_for_robotics`.

17,717 clips over 25 AMASS sub-datasets (KIT 4164, BioMotionLab_NTroje 3031,
WEIZMANN 2216, CMU 1978, BMLmovi 1754, GRAB 1340, ...), CC-BY-4.0 on the
retargeting -- **the upstream AMASS sub-dataset licences still apply and are
research/non-commercial; cite them.**

Layout:  raw/AMASS/g1/<SUB_DATASET>/<SEQUENCE>/<name>_poses_<fps>_jpos.npy
Payload: [T, 36] float64, the shared root-pos / XYZW-quat / 29-joint layout.

**The frame rate lives in the FILENAME and varies** -- measured over the full
set: 120 Hz (12,222 files), 100 (4,546), 60 (871), 250 (56), 59 (13), 150 (9).
Assuming 120 would time-stretch a quarter of the dataset, so the rate is parsed
per file and a clip whose name does not carry one is rejected rather than
guessed.

The dataset card names `g1_29dof_lock_waist_rev_1_0.xml` as the retarget target,
which would imply a frozen waist -- it does NOT match the data: waist roll/pitch
genuinely move (AMASS CMU 01_01 reaches 0.45 rad of waist pitch).  The joint
list in the card is the full 29 and agrees with ours; the data is what we
believe.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from retarget_motion import paths
from retarget_motion.sources import RawClip, relpath_under, split_36

# "<anything>_poses_<fps>_jpos.npy"; the fps group is what we need.
_FPS_RE = re.compile(r"_(\d+)_jpos\.npy$")

SOURCE = paths.AMASS


def root_dir() -> Path:
    return paths.RAW_DIR / SOURCE


def find_files(root: Path | None = None) -> list[Path]:
    """Every raw clip, sorted for reproducible ordering."""
    root = root or root_dir()
    return sorted(root.rglob("*.npy"))


def parse_fps(path: Path) -> float:
    m = _FPS_RE.search(path.name)
    if not m:
        raise ValueError(
            f"{path.name}: no '_<fps>_jpos.npy' suffix -- refusing to guess the "
            f"frame rate (this dataset mixes 59/60/100/120/150/250 Hz)"
        )
    return float(m.group(1))


def load(path: Path, root: Path | None = None) -> RawClip:
    root = root or root_dir()
    rel = relpath_under(root, path)
    root_pos, quat, dof = split_36(np.load(path), rel)
    return RawClip(root_pos=root_pos, root_rot_xyzw=quat, dof_pos=dof,
                   fps=parse_fps(path), rel_path=rel)


__all__ = ["SOURCE", "find_files", "load", "parse_fps", "root_dir"]
