"""THE canonical converted-motion schema (one .npz per clip).

Field-for-field compatible with GentleHumanoid / the Axellwppr GMR fork, so
their `scripts/data_process/generate_dataset.py` can consume `motions/`
unchanged:

    fps             ()          float32   frames per second of THIS file
    root_pos        (T, 3)      float32   world root (pelvis) position, metres
    root_rot        (T, 4)      float32   world root quaternion, XYZW  <-- not wxyz
    dof_pos         (T, D)      float32   joint angles, radians, `joint_names` order
    local_body_pos  (T, B, 3)   float32   body positions with the root pinned to
                                          the origin at identity (i.e. in the
                                          root frame), `body_names` order
    local_body_rot  (T, B, 4)   float32   the same bodies' quaternions, XYZW
    body_names      (B,)        <U        MuJoCo body names, world body excluded
    joint_names     (D,)        <U        actuated joint names, MuJoCo order

Two conventions that bite if you get them wrong, both inherited deliberately:

* **Quaternions are XYZW** (scalar LAST) everywhere in this file.  MuJoCo's
  `qpos`/`xquat` are WXYZ (scalar FIRST) -- `to_qpos()` below is the only place
  the swap should happen.
* **`root_pos` is world**, already ground-adjusted so the lowest body touches
  z=0, and xy-zeroed at frame 0.  The raw G1 sources do NOT have this: their
  root z is frame-0-relative, which puts the robot ~0.72-0.75 m into the floor
  if you believe it (measured, see convert.py).

EXTRA keys we add on top of the reference schema (all prefixed `src_`, ignored
by every reference consumer, kept because losing provenance on 17k clips is
unrecoverable):

    src_dataset     ()  <U   "AMASS" | "LAFAN" | "InterX"
    src_path        ()  <U   path of the raw file, relative to raw/<dataset>/
    src_fps         ()  float32  the raw file's frame rate BEFORE resampling
    src_ground_dz   ()  float32  metres added to root z by the ground adjust
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reference-schema keys, in the order the GentleHumanoid README lists them.
REQUIRED_KEYS = (
    "fps", "root_pos", "root_rot", "dof_pos",
    "local_body_pos", "local_body_rot", "body_names", "joint_names",
)
PROVENANCE_KEYS = ("src_dataset", "src_path", "src_fps", "src_ground_dz")


@dataclass
class Motion:
    """One converted clip, in memory."""

    fps: float
    root_pos: np.ndarray        # (T, 3)
    root_rot: np.ndarray        # (T, 4) xyzw
    dof_pos: np.ndarray         # (T, D)
    local_body_pos: np.ndarray  # (T, B, 3)
    local_body_rot: np.ndarray  # (T, B, 4) xyzw
    body_names: list[str]
    joint_names: list[str]
    src_dataset: str = ""
    src_path: str = ""
    src_fps: float = 0.0
    src_ground_dz: float = 0.0

    @property
    def num_frames(self) -> int:
        return int(self.root_pos.shape[0])

    @property
    def duration(self) -> float:
        return self.num_frames / float(self.fps)

    def validate(self) -> None:
        """Raise on any shape/convention violation. Cheap -- always call it."""
        T, D, B = self.num_frames, len(self.joint_names), len(self.body_names)
        for name, arr, shape in (
            ("root_pos", self.root_pos, (T, 3)),
            ("root_rot", self.root_rot, (T, 4)),
            ("dof_pos", self.dof_pos, (T, D)),
            ("local_body_pos", self.local_body_pos, (T, B, 3)),
            ("local_body_rot", self.local_body_rot, (T, B, 4)),
        ):
            if tuple(arr.shape) != shape:
                raise ValueError(f"{name}: expected {shape}, got {tuple(arr.shape)}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name}: contains NaN/Inf")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        for name, q in (("root_rot", self.root_rot),
                        ("local_body_rot", self.local_body_rot)):
            n = np.linalg.norm(q, axis=-1)
            if not np.allclose(n, 1.0, atol=1e-3):
                raise ValueError(
                    f"{name}: not unit quaternions (norm {n.min():.4f}..{n.max():.4f})"
                )

    def save(self, path: str | Path) -> Path:
        """Write the npz (atomically: temp file + rename, so an interrupted run
        never leaves a half-written clip that reads as valid)."""
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        # Write through an open handle, NOT a path: np.savez_compressed appends
        # ".npz" to any path that lacks it, so passing "x.npz.tmp" produces
        # "x.npz.tmp.npz" and the rename below then fails on a missing file.
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                fps=np.float32(self.fps),
                root_pos=self.root_pos.astype(np.float32),
                root_rot=self.root_rot.astype(np.float32),
                dof_pos=self.dof_pos.astype(np.float32),
                local_body_pos=self.local_body_pos.astype(np.float32),
                local_body_rot=self.local_body_rot.astype(np.float32),
                body_names=np.array(self.body_names),
                joint_names=np.array(self.joint_names),
                src_dataset=np.array(self.src_dataset),
                src_path=np.array(self.src_path),
                src_fps=np.float32(self.src_fps),
                src_ground_dz=np.float32(self.src_ground_dz),
            )
        tmp.replace(path)
        return path

    def to_qpos(self) -> np.ndarray:
        """(T, 7+D) MuJoCo qpos: [pos(3), quat WXYZ(4), dof]. The one place the
        xyzw -> wxyz swap happens."""
        T = self.num_frames
        qpos = np.zeros((T, 7 + self.dof_pos.shape[1]), dtype=np.float64)
        qpos[:, 0:3] = self.root_pos
        qpos[:, 3] = self.root_rot[:, 3]        # w
        qpos[:, 4:7] = self.root_rot[:, 0:3]    # x y z
        qpos[:, 7:] = self.dof_pos
        return qpos


def load(path: str | Path) -> Motion:
    """Read a converted clip. Accepts files without the `src_*` extras (e.g. one
    produced by the upstream GMR fork itself)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in REQUIRED_KEYS if k not in z]
        if missing:
            raise KeyError(f"{path}: missing schema keys {missing}")
        m = Motion(
            fps=float(z["fps"]),
            root_pos=z["root_pos"].astype(np.float64),
            root_rot=z["root_rot"].astype(np.float64),
            dof_pos=z["dof_pos"].astype(np.float64),
            local_body_pos=z["local_body_pos"].astype(np.float64),
            local_body_rot=z["local_body_rot"].astype(np.float64),
            body_names=[str(s) for s in z["body_names"]],
            joint_names=[str(s) for s in z["joint_names"]],
            src_dataset=str(z["src_dataset"]) if "src_dataset" in z else "",
            src_path=str(z["src_path"]) if "src_path" in z else "",
            src_fps=float(z["src_fps"]) if "src_fps" in z else 0.0,
            src_ground_dz=float(z["src_ground_dz"]) if "src_ground_dz" in z else 0.0,
        )
    return m


__all__ = ["PROVENANCE_KEYS", "REQUIRED_KEYS", "Motion", "load"]
