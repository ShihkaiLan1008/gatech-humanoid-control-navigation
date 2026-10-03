"""Batched 1-step SRBD force-distribution QP (18 inputs, ADMM backend).

Torch port of ``mpc_mujoco/srbd/srbd_mpc.py`` collapsed to horizon 1 with the
Stage-0 input construction (``mpc_mujoco/srbd/one_step.py``) baked in:

- Q position/attitude rows are dropped entirely — at one Euler step they are
  inert; the active objective is acceleration matching on the omega/v rows
  (omega weights carry the measured x10 boost) plus ``|u - u_ref|^2_R``.
- All pose feedback is *in the plan*: u_ref carries m*(a_des + g) split by the
  weight-shift fractions, and the omega reference row carries the attitude PD.
- Constraints are the validated set: friction pyramid, CoP rectangle, |Mz|
  torsional bound, contact-gated force bounds. Hand slots exist (u is 18-D)
  but are pinned to zero until Stage A2 enables them.

Decision variable u = [F_L(3), M_L(3), F_R(3), M_R(3) | F_LH(3), F_RH(3)],
world frame, reaction ON the robot, moments about the foot contact center.
Solver: ``g1_mpc.admm_qp.ADMMSolver`` (pi^nMPC-style batched GPU ADMM).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from rl.mpc_online_tracking.mpc_training.cd_mpc.admm_qp import ADMMSolver, QPData

from .kin import rot_z, skew

NU = 18
BIG = 1.0e6


@dataclass
class OneStepParams:
    mass: float = 35.0  # overwritten from the compiled model at init
    inertia_body: np.ndarray = field(default_factory=lambda: np.eye(3))
    dt: float = 0.02  # env step dt (QP solved at policy rate, ZOH between)
    mu: float = 0.5
    fz_max: float = 500.0
    l_toe: float = 0.080
    l_heel: float = 0.080
    l_side: float = 0.025
    r_spin: float = 0.035
    # omega rows carry the Stage-0 x10 boost (the only attitude channel at n=1)
    q_omega: tuple[float, float, float] = (50.0, 50.0, 100.0)
    q_v: tuple[float, float, float] = (40.0, 40.0, 40.0)
    r_force: float = 2e-3
    r_moment: float = 5e-3
    r_hand: float = 1e-3
    v_hand: float = 5e-2  # hand-force tracking weight while pressing (A2)
    f_hand_max: float = 30.0
    mu_hand: float = 0.6
    # folded PD (Stage-0 values, from mpc_track_ref)
    kp_com: float = 15.0
    kd_com: float = 6.0
    a_clip: float = 2.5
    kp_att: float = 8.0


class OneStepQP:
    def __init__(self, p: OneStepParams, num_envs: int, device: str):
        self.p = p
        self.num_envs = num_envs
        self.device = device
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)  # noqa: E731

        self.I_b = t(p.inertia_body)
        mu, lt, lh, ls = p.mu, p.l_toe, p.l_heel, p.l_side
        ms = p.r_spin * mu
        cone = t(
            [
                [1, 0, -mu, 0, 0, 0],
                [-1, 0, -mu, 0, 0, 0],
                [0, 1, -mu, 0, 0, 0],
                [0, -1, -mu, 0, 0, 0],
                [0, 0, -ls, 1, 0, 0],
                [0, 0, -ls, -1, 0, 0],
                [0, 0, -lh, 0, 1, 0],
                [0, 0, -lt, 0, -1, 0],
                [0, 0, -ms, 0, 0, 1],
                [0, 0, -ms, 0, 0, -1],
            ]
        )
        G = torch.zeros(20, NU, device=device)
        G[0:10, 0:6] = cone
        G[10:20, 6:12] = cone
        self.G = G.unsqueeze(0).expand(num_envs, -1, -1).contiguous()
        self.b = torch.zeros(num_envs, 20, device=device)

        r_diag = [p.r_force] * 3 + [p.r_moment] * 3
        self.R_diag = t(r_diag * 2 + [p.r_hand] * 6)
        self.Q6 = t(list(p.q_omega) + list(p.q_v))
        self.g_vec = t([0.0, 0.0, 9.81])

        # tolerances are in solution units (Newtons / N.m): centi-Newton
        # precision is plenty for control and lets the convergence flag mean
        # what it says (1e-3 flagged fine solutions as non-converged)
        self.solver = ADMMSolver(max_iter=120, eps_abs=1e-2, eps_rel=1e-2)
        self.warm_z = torch.zeros(num_envs, NU, device=device)
        self.last_converged = torch.ones(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids) -> None:
        self.warm_z[env_ids] = 0.0

    @torch.no_grad()
    def solve(
        self,
        com: torch.Tensor,  # [B, 3] whole-body CoM, world
        v: torch.Tensor,  # [B, 3] CoM velocity (filtered)
        omega: torch.Tensor,  # [B, 3] base angular velocity, world (filtered)
        rpy: torch.Tensor,  # [B, 3]
        r_feet: torch.Tensor,  # [B, 2, 3] com -> foot contact center, world
        contact: torch.Tensor,  # [B, 2] bool, scheduled AND (for force bounds) valid
        w_frac: torch.Tensor,  # [B, 2] weight-shift load fractions
        pos_ref_xy: torch.Tensor,  # [B, 2]
        height_ref: torch.Tensor,  # [B]
        yaw_ref: torch.Tensor,  # [B]
        v_cmd_w: torch.Tensor,  # [B, 3] commanded velocity, world
        wz_cmd: torch.Tensor,  # [B]
        delta_a: torch.Tensor | None = None,  # [B, 6] policy residual [lin, ang]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (u [B, 18], a_des [B, 3])."""
        p = self.p
        B = com.shape[0]
        dt = p.dt
        yaw = rpy[:, 2]

        # --- folded CoM PD -> desired acceleration (rides in the plan)
        a_des = torch.zeros(B, 3, device=self.device)
        a_des[:, :2] = p.kp_com * (pos_ref_xy - com[:, :2]) + p.kd_com * (
            v_cmd_w[:, :2] - v[:, :2]
        )
        a_des[:, 2] = p.kp_com * (height_ref - com[:, 2]) + p.kd_com * (0.0 - v[:, 2])
        if delta_a is not None:
            a_des = a_des + delta_a[:, :3]
        a_des = a_des.clamp(-p.a_clip, p.a_clip)

        # --- attitude PD -> omega reference (the only attitude channel at n=1)
        Rz = rot_z(yaw)
        e_yaw = torch.remainder(yaw_ref - yaw + torch.pi, 2 * torch.pi) - torch.pi
        e_rp = torch.stack([-rpy[:, 0], -rpy[:, 1], torch.zeros_like(yaw)], dim=-1)
        omega_ref = torch.einsum("bij,bj->bi", Rz, p.kp_att * e_rp)
        omega_ref[:, 2] = wz_cmd + p.kp_att * e_yaw
        if delta_a is not None:
            omega_ref = omega_ref + delta_a[:, 3:] * dt

        # --- dynamics map u -> [d_omega, d_v] over one step
        I_w = Rz @ self.I_b @ Rz.transpose(1, 2)
        I_inv = torch.linalg.inv(I_w)
        A = torch.zeros(B, 6, NU, device=self.device)
        eye3 = torch.eye(3, device=self.device).expand(B, 3, 3)
        for i in range(2):
            c = 6 * i
            A[:, 0:3, c : c + 3] = torch.bmm(I_inv, skew(r_feet[:, i])) * dt
            A[:, 0:3, c + 3 : c + 6] = I_inv * dt
            A[:, 3:6, c : c + 3] = eye3 * (dt / p.mass)

        # residual err = A u + cvec, target 0:
        #   omega row: omega1 - omega_ref = A_w u + (omega - omega_ref)
        #   v row:     v1 - (v + a_des dt) = A_v u - g dt - a_des dt
        cvec = torch.cat(
            [omega - omega_ref, -self.g_vec.unsqueeze(0) * dt - a_des * dt],
            dim=-1,
        )

        # --- u_ref: load-share force plan + quasi-static moment share
        f_tot = p.mass * (a_des + self.g_vec.unsqueeze(0))  # [B,3]
        u_ref = torch.zeros(B, NU, device=self.device)
        shares = w_frac.unsqueeze(-1) * f_tot.unsqueeze(1)  # [B,2,3]
        contact_f = contact.float().unsqueeze(-1)
        shares = shares * contact_f
        v_ref_norm = (v + a_des * dt).norm(dim=-1)
        quasi_static = v_ref_norm < 0.05
        m_res = -torch.cross(r_feet, shares, dim=-1).sum(dim=1)  # [B,3]
        n_st = contact.float().sum(dim=-1).clamp_min(1.0)
        m_share = m_res / n_st.unsqueeze(-1)
        m_share = torch.where(
            quasi_static.unsqueeze(-1), m_share, torch.zeros_like(m_share)
        )
        for i in range(2):
            c = 6 * i
            u_ref[:, c : c + 3] = shares[:, i]
            u_ref[:, c + 3 : c + 6] = m_share * contact_f[:, i]

        # --- QP matrices. The raw objective is R-dominated with R ~ 1e-3,
        # which is pathologically scaled for ADMM at rho ~ 1 (the consensus
        # terms swamp H and the residual tolerance accepts far-from-optimal
        # points — measured: standing fz came out 58 N instead of 327 N).
        # Scaling H and h by 1/r_force (same argmin) puts H at O(1).
        s = 1.0 / self.p.r_force
        QA = self.Q6.view(1, 6, 1) * A
        H = torch.bmm(A.transpose(1, 2), QA)
        H[:, range(NU), range(NU)] += self.R_diag
        h = torch.bmm(A.transpose(1, 2), (self.Q6 * cvec).unsqueeze(-1)).squeeze(-1)
        h = h - self.R_diag * u_ref
        H = H * s
        h = h * s

        lb = torch.full((B, NU), -BIG, device=self.device)
        ub = torch.full((B, NU), BIG, device=self.device)
        for i in range(2):
            c = 6 * i
            on = contact[:, i]
            lb[:, c + 2] = torch.where(
                on, torch.zeros_like(on, dtype=torch.float32), lb[:, c + 2]
            )
            ub[:, c + 2] = torch.where(
                on, torch.full_like(ub[:, c + 2], p.fz_max), ub[:, c + 2]
            )
            off = ~on
            for k in range(6):
                lb[:, c + k] = torch.where(
                    off, torch.zeros_like(lb[:, c + k]), lb[:, c + k]
                )
                ub[:, c + k] = torch.where(
                    off, torch.zeros_like(ub[:, c + k]), ub[:, c + k]
                )
        lb[:, 12:] = 0.0  # hands pinned in A1
        ub[:, 12:] = 0.0

        sol = self.solver.solve(
            QPData(H=H, h=h, G=self.G, b=self.b, lb=lb, ub=ub),
            warm_z=self.warm_z,
        )
        self.warm_z = sol.z.clone()
        self.last_converged = sol.converged
        return sol.z, a_des
