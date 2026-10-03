"""Batched gait clock, Raibert foothold, weight-shift fractions, swing spline.

Torch port of ``mpc_mujoco/srbd/gait.py`` + ``swing.py`` + the weight-shift
trapezoid of ``mpc_mujoco/srbd/one_step.py`` — all [B]-batched, phase carried
per env (advanced by the action term, reset on episode reset).

Stance-default: envs whose velocity command is ~zero freeze the clock in
double support (wb_humanoid_mpc's FSM philosophy) — this is also what makes
the stand-portion of the velocity distribution trainable without a gait.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GaitParams:
    t_swing: float = 0.18
    t_ds: float = 0.12
    swing_height: float = 0.08
    raibert_x: float = 0.05  # sagittal velocity-error gain
    raibert_y: float = 0.27  # coronal: capture-point strength (Stage 0 finding)
    touchdown_overdrive: float = 0.015
    t_ramp: float = 0.10  # weight-shift crossfade
    cmd_deadband: float = 0.05  # |cmd| below this -> stance-default (clock frozen)

    @property
    def cycle(self) -> float:
        return 2.0 * (self.t_swing + self.t_ds)


class BatchedGait:
    """Per-env phase clock. Phase advances only while the command is active."""

    def __init__(self, p: GaitParams, num_envs: int, device: str):
        self.p = p
        self.num_envs = num_envs
        self.device = device
        self.phase = torch.zeros(num_envs, device=device)  # [0, cycle)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids) -> None:
        self.phase[env_ids] = 0.0
        self.active[env_ids] = False

    def update(
        self,
        dt: float,
        cmd: torch.Tensor,
        rate_scale: torch.Tensor | None = None,
        v_meas_xy: torch.Tensor | None = None,
    ) -> None:
        """cmd [B,3] (vx, vy, wz). Freezing happens only at the DS boundary.

        rate_scale [B]: policy gait-frequency residual (g1_wall 9-D channel) —
        the clock advances at dt*(1+rate_scale).
        v_meas_xy [B,2]: measured CoM velocity — a pushed robot activates the
        clock even at zero command (reactive protective stepping; without this
        a standing robot is structurally forbidden from stepping).
        """
        want = cmd.norm(dim=-1) > self.p.cmd_deadband
        if v_meas_xy is not None:
            want = want | (v_meas_xy.norm(dim=-1) > 0.20)
        # activate immediately; deactivate only once the cycle wraps through DS
        self.active |= want
        step = dt * (1.0 + rate_scale) if rate_scale is not None else dt
        self.phase = torch.where(
            self.active, (self.phase + step) % self.p.cycle, self.phase
        )
        # deactivate at the start-of-cycle DS window so a foot is never frozen mid-swing
        in_ds0 = self.phase < self.p.t_ds
        stop = (~want) & in_ds0
        self.active &= ~stop
        self.phase = torch.where(stop, torch.zeros_like(self.phase), self.phase)

    # ---------------------------------------------------------------- queries
    def contact_state(self, shift: float = 0.0) -> torch.Tensor:
        """[B, 2] bool (left, right) scheduled contact at phase + shift."""
        p = self.p
        ph = (self.phase + shift) % p.cycle
        left_swing = (ph >= p.t_ds) & (ph < p.t_ds + p.t_swing)
        right_swing = (ph >= 2 * p.t_ds + p.t_swing) & (ph < 2 * p.t_ds + 2 * p.t_swing)
        contact = torch.ones(self.num_envs, 2, dtype=torch.bool, device=self.device)
        contact[:, 0] = ~(left_swing & self.active)
        contact[:, 1] = ~(right_swing & self.active)
        return contact

    def swing_phase(self) -> torch.Tensor:
        """[B, 2] in [0,1] while swinging, -1 otherwise."""
        p = self.p
        ph = self.phase
        out = torch.full((self.num_envs, 2), -1.0, device=self.device)
        ls = (ph >= p.t_ds) & (ph < p.t_ds + p.t_swing) & self.active
        rs_start = 2 * p.t_ds + p.t_swing
        rs = (ph >= rs_start) & (ph < rs_start + p.t_swing) & self.active
        out[:, 0] = torch.where(ls, (ph - p.t_ds) / p.t_swing, out[:, 0])
        out[:, 1] = torch.where(rs, (ph - rs_start) / p.t_swing, out[:, 1])
        return out

    def load_fractions(self, shift: float = 0.0) -> torch.Tensor:
        """[B, 2] per-foot load share (trapezoid over each stance interval)."""
        p = self.p
        cyc = p.cycle
        ph = (self.phase + shift) % cyc
        swing_start = torch.tensor([p.t_ds, 2 * p.t_ds + p.t_swing], device=self.device)
        swing_end = torch.tensor(
            [p.t_ds + p.t_swing, (2 * p.t_ds + 2 * p.t_swing) % cyc],
            device=self.device,
        )
        ph2 = ph.unsqueeze(-1)  # [B,1]
        t_td = (ph2 - swing_end) % cyc
        t_lo = (swing_start - ph2) % cyc
        rho = torch.clamp(torch.minimum(t_td, t_lo) / max(p.t_ramp, 1e-6), 0.0, 1.0)
        in_swing = (ph2 >= swing_start) & (ph2 < swing_start + p.t_swing)
        rho = torch.where(in_swing, torch.zeros_like(rho), rho)
        rho = torch.where(self.active.unsqueeze(-1), rho, torch.full_like(rho, 1.0))
        tot = rho.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        half = torch.full_like(rho, 0.5)
        return torch.where(tot > 1e-5, rho / tot, half)


def raibert_foothold(
    hip_w: torch.Tensor,  # [B, 2] hip xy, world
    v_w: torch.Tensor,  # [B, 2] measured com velocity xy
    v_des_w: torch.Tensor,  # [B, 2]
    t_stance: float,
    gain_xy: torch.Tensor,  # [2] axis-split gains, world axes after yaw rotation
    R_yaw: torch.Tensor,  # [B, 2, 2] yaw rotation (body->world) for the gain axes
) -> torch.Tensor:
    """[B, 2] touchdown xy. Nominal term uses measured velocity (blend 0 —
    the Stage-0/mpc_mujoco tuning), correction gain is axis-split in the yaw
    frame (sagittal 0.05 / coronal 0.27, measured in Stage 0)."""
    err_w = v_w - v_des_w
    err_b = torch.einsum("bij,bj->bi", R_yaw.transpose(1, 2), err_w)
    corr_b = gain_xy.unsqueeze(0) * err_b
    corr_w = torch.einsum("bij,bj->bi", R_yaw, corr_b)
    return hip_w + 0.5 * t_stance * v_w + corr_w


def swing_pos_vel(
    p_lift: torch.Tensor,  # [B, 3]
    p_land: torch.Tensor,  # [B, 3]
    height: "torch.Tensor | float",  # scalar or [B] (policy apex residual)
    s: torch.Tensor,  # [B] phase in [0,1]
    t_swing: float,
    overdrive: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched port of swing.swing_position_velocity (smoothstep xy, bell z)."""
    s = s.clamp(0.0, 1.0)
    ds_dt = 1.0 / max(t_swing, 1e-6)

    sx = (s / 0.80).clamp(0.0, 1.0)
    sxy = sx * sx * (3 - 2 * sx)
    xy = p_lift[:, :2] + (p_land[:, :2] - p_lift[:, :2]) * sxy.unsqueeze(-1)
    dsxy = torch.where(s < 0.80, 6.0 * sx * (1.0 - sx) / 0.80, torch.zeros_like(s))
    v_xy = (p_land[:, :2] - p_lift[:, :2]) * (dsxy * ds_dt).unsqueeze(-1)

    z_end = p_land[:, 2] - overdrive
    lo = s < 0.5
    u1 = (s / 0.5).clamp(max=1.0)
    u2 = ((s - 0.5) / 0.5).clamp(min=0.0)
    z_up = p_lift[:, 2] + (height - p_lift[:, 2]) * (u1 * u1 * (3 - 2 * u1))
    z_dn = height + (z_end - height) * (u2 * u2 * (3 - 2 * u2))
    z = torch.where(lo, z_up, z_dn)
    dz_up = (height - p_lift[:, 2]) * 6.0 * u1 * (1.0 - u1) * 2.0
    dz_dn = (z_end - height) * 6.0 * u2 * (1.0 - u2) * 2.0
    dz = torch.where(lo, dz_up, dz_dn) * ds_dt

    p = torch.cat([xy, z.unsqueeze(-1)], dim=-1)
    v = torch.cat([v_xy, dz.unsqueeze(-1)], dim=-1)
    return p, v
