"""Batched G1 leg kinematics on GPU (pytorch_kinematics over the mjlab MJCF).

Provides what the 1-step QP/WBC needs and mjlab does not expose: leg Jacobians
(for stance ``tau = J^T (-u)``), swing-foot FK/IK, and hip positions — all
[B, ...] torch tensors on the env device. The chain is built from mjlab's own
``g1.xml`` (freejoint stripped: the chain is pelvis-relative by construction,
matching the layer contract — no world quantities inside the kinematics).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytorch_kinematics as pk
import torch

from mjlab.asset_zoo.robots.unitree_g1 import g1_constants

_XML_PATH = Path(g1_constants.__file__).parent / "xmls" / "g1.xml"

LEG_JOINTS = {
    0: [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ],
    1: [
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ],
}
FOOT_LINK = {0: "left_ankle_roll_link", 1: "right_ankle_roll_link"}


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) wxyz -> (..., 3, 3)."""
    w, x, y, z = q.unbind(-1)
    two = 2.0
    R = torch.stack(
        [
            1 - two * (y * y + z * z),
            two * (x * y - w * z),
            two * (x * z + w * y),
            two * (x * y + w * z),
            1 - two * (x * x + z * z),
            two * (y * z - w * x),
            two * (x * z - w * y),
            two * (y * z + w * x),
            1 - two * (x * x + y * y),
        ],
        dim=-1,
    )
    return R.reshape(q.shape[:-1] + (3, 3))


def rot_z(yaw: torch.Tensor) -> torch.Tensor:
    """(B,) -> (B, 3, 3)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    zero = torch.zeros_like(c)
    one = torch.ones_like(c)
    return torch.stack([c, -s, zero, s, c, zero, zero, zero, one], dim=-1).reshape(
        -1, 3, 3
    )


def skew(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3)."""
    zero = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            zero,
            -v[..., 2],
            v[..., 1],
            v[..., 2],
            zero,
            -v[..., 0],
            -v[..., 1],
            v[..., 0],
            zero,
        ],
        dim=-1,
    ).reshape(v.shape[:-1] + (3, 3))


class G1LegKin:
    """Two pelvis-rooted 6-DoF serial chains with batched FK / Jacobian / DLS-IK."""

    def __init__(self, device: str, dtype: torch.dtype = torch.float32):
        xml = _XML_PATH.read_text()
        xml = re.sub(r"<freejoint[^/]*/>", "", xml)
        import os

        cwd = os.getcwd()
        try:
            os.chdir(_XML_PATH.parent)  # mesh asset paths resolve relative to the xml
            chain = pk.build_chain_from_mjcf(xml)
        finally:
            os.chdir(cwd)
        self.chains = {
            f: pk.SerialChain(chain, FOOT_LINK[f], "pelvis").to(
                device=device, dtype=dtype
            )
            for f in range(2)
        }
        for f in range(2):
            got = self.chains[f].get_joint_parameter_names()
            assert got == LEG_JOINTS[f], f"leg {f} chain order mismatch: {got}"
        self.device = device
        self.dtype = dtype

    # ------------------------------------------------------------------ FK / J
    def fk(self, f: int, q_leg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q_leg [B,6] -> (pos [B,3], R [B,3,3]) of the foot link, pelvis frame."""
        m = self.chains[f].forward_kinematics(q_leg, end_only=True).get_matrix()
        return m[:, :3, 3], m[:, :3, :3]

    def jac(self, f: int, q_leg: torch.Tensor) -> torch.Tensor:
        """[B, 6, 6] spatial Jacobian (linear; angular rows), pelvis frame."""
        return self.chains[f].jacobian(q_leg)

    def jac_world(
        self,
        f: int,
        q_leg: torch.Tensor,
        R_wb: torch.Tensor,
        r_point_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """World-frame Jacobian, optionally shifted to a point.

        r_point_b [B,3]: point minus foot-link origin, pelvis frame. The shifted
        linear rows satisfy v_point = Jv q̇ for a point rigidly attached to the
        foot — the correct lever arm for wrenches referenced at the contact
        center rather than the ankle origin.
        """
        J = self.jac(f, q_leg)
        Jv, Jw = J[:, :3], J[:, 3:]
        if r_point_b is not None:
            Jv = Jv - torch.bmm(skew(r_point_b), Jw)
        Jv_w = torch.bmm(R_wb, Jv)
        Jw_w = torch.bmm(R_wb, Jw)
        return torch.cat([Jv_w, Jw_w], dim=1)

    # ------------------------------------------------------------------ IK
    def ik_6dof(
        self,
        f: int,
        q_init: torch.Tensor,  # [B, 6]
        p_des_b: torch.Tensor,  # [B, 3] pelvis frame
        R_des_b: torch.Tensor,  # [B, 3, 3] pelvis frame
        iters: int = 3,
        damping: float = 1e-3,
    ) -> torch.Tensor:
        """Batched damped-least-squares 6-DoF IK (position + sole orientation).

        Solves the same square task as mpc_mujoco's swing IK (position +
        sole-flat + heading); a few warm-started iterations at 50 Hz suffice
        because targets move slowly between calls.
        """
        q = q_init.clone()
        eye = torch.eye(6, device=q.device, dtype=q.dtype).unsqueeze(0)
        for _ in range(iters):
            p, R = self.fk(f, q)
            J = self.jac(f, q)
            e_p = p_des_b - p
            # so3 log via the skew-symmetric part (small-angle: exact enough
            # for the per-tick correction the DLS step applies)
            R_err = torch.bmm(R_des_b, R.transpose(1, 2))
            e_r = 0.5 * torch.stack(
                [
                    R_err[:, 2, 1] - R_err[:, 1, 2],
                    R_err[:, 0, 2] - R_err[:, 2, 0],
                    R_err[:, 1, 0] - R_err[:, 0, 1],
                ],
                dim=-1,
            )
            e = torch.cat([e_p, e_r], dim=-1)
            JJt = torch.bmm(J, J.transpose(1, 2)) + damping * eye
            dq = torch.bmm(
                J.transpose(1, 2), torch.linalg.solve(JJt, e.unsqueeze(-1))
            ).squeeze(-1)
            q = q + dq
        return q
