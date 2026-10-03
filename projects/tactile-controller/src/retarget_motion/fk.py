"""Forward kinematics on OUR G1, for the two things the raw sources don't ship.

The raw retargeted datasets give only (root_pos, root_rot, dof_pos).  The
canonical schema also needs `local_body_pos` / `local_body_rot`, and the root
height has to be fixed up -- both are FK jobs, done here against
`assets.TACTILE_XML`, i.e. the same G1 the RL side trains on and the offline TO
solves against.

Doing FK on OUR model rather than the upstream retargeter's is deliberate: the
body sets differ (theirs carries toe/mocap/mimic helper bodies, ours carries the
`{left,right}_tactile` sensor links), and every downstream consumer in this repo
speaks our body names.  The 29 JOINTS are identical in name and order, which is
what makes this sound -- `dof_pos` needs no remapping.

Two measured facts that drive the code:

* **The raw root z is NOT world height.**  It is frame-0-relative: with root
  z = 0 the FK puts the lowest body at -0.744 m (AMASS "A1 - Stand") / -0.715 m
  (AMASS CMU 01_01), i.e. the robot is three quarters buried.  `ground_offset`
  fixes this the way the reference pipeline does.
* Only ONE FK pass per frame is needed.  Body poses in the world are just the
  root transform applied to the local ones, so `world_from_local` recovers them
  with batched numpy instead of a second pass through MuJoCo.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import assets


@lru_cache(maxsize=4)
def _model(xml: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(os.path.join(str(assets.ASSET_DIR), xml))


class G1Kinematics:
    """FK helper bound to one G1 MJCF. Cheap to construct (the model is cached)."""

    def __init__(self, xml: str = assets.TACTILE_XML) -> None:
        self.model = _model(xml)
        self.data = mujoco.MjData(self.model)
        self.xml = xml

        # Bodies, world (id 0) excluded -- it is not part of the robot and would
        # be a constant zero row in every clip.
        self.body_ids = list(range(1, self.model.nbody))
        self.body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in self.body_ids
        ]
        # Actuated joints in MuJoCo order, free/floating base excluded.
        self.joint_names, self._qadr = [], []
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            self.joint_names.append(
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            )
            self._qadr.append(int(self.model.jnt_qposadr[j]))
        self._qadr = np.asarray(self._qadr, dtype=int)

    @property
    def num_dof(self) -> int:
        return len(self.joint_names)

    @property
    def num_bodies(self) -> int:
        return len(self.body_names)

    def local_body_pose(self, dof_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(T, D) joint angles -> body poses in the ROOT frame.

        Returns (pos (T, B, 3), quat (T, B, 4) XYZW). The root is pinned at the
        origin with identity rotation, so MuJoCo's world frame IS the root frame
        and `xpos`/`xquat` can be read straight out.
        """
        dof_pos = np.asarray(dof_pos, dtype=np.float64)
        if dof_pos.ndim != 2 or dof_pos.shape[1] != self.num_dof:
            raise ValueError(
                f"dof_pos must be (T, {self.num_dof}), got {dof_pos.shape}"
            )
        T, B = dof_pos.shape[0], self.num_bodies
        pos = np.empty((T, B, 3))
        quat_wxyz = np.empty((T, B, 4))

        m, d = self.model, self.data
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0                       # identity root quaternion (wxyz)
        for t in range(T):
            d.qpos[self._qadr] = dof_pos[t]
            mujoco.mj_kinematics(m, d)        # xpos/xquat only -- no dynamics
            pos[t] = d.xpos[self.body_ids]
            quat_wxyz[t] = d.xquat[self.body_ids]

        return pos, _wxyz_to_xyzw(quat_wxyz)

    def world_body_pos(
        self,
        root_pos: np.ndarray,
        root_rot_xyzw: np.ndarray,
        local_body_pos: np.ndarray,
    ) -> np.ndarray:
        """Body positions in the world: root_pos + R(root_rot) @ local. (T, B, 3).

        One rotation per frame applied to all B bodies of that frame, hence the
        einsum rather than `Rotation.apply` (which pairs one rotation with one
        vector)."""
        R = Rotation.from_quat(np.asarray(root_rot_xyzw, dtype=np.float64)).as_matrix()
        rotated = np.einsum("tij,tbj->tbi", R, np.asarray(local_body_pos, float))
        return rotated + np.asarray(root_pos, dtype=np.float64)[:, None, :]

    def ground_offset(
        self,
        root_pos: np.ndarray,
        root_rot_xyzw: np.ndarray,
        local_body_pos: np.ndarray,
        per_frame: bool = False,
    ) -> np.ndarray:
        """dz to add to root z so the robot rests on z=0.

        `per_frame=False` (default, and what the reference GMR exporter does):
        ONE offset for the whole clip, from the single lowest body over all
        frames -- preserves the vertical motion (a jump stays a jump).
        `per_frame=True` reproduces GentleHumanoid's `always_on_ground` variant,
        which glues the lowest body to the floor every frame and therefore
        FLATTENS airborne phases; use it only if a consumer demands it.

        Returns a scalar array (per_frame=False) or (T,) (per_frame=True).
        """
        z = self.world_body_pos(root_pos, root_rot_xyzw, local_body_pos)[..., 2]
        return -z.min(axis=1) if per_frame else np.asarray(-z.min())


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """MuJoCo quaternions (scalar first) -> the schema's scalar-last form."""
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


__all__ = ["G1Kinematics"]
