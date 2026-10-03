"""Contact-scheduled centroidal QP-MPC (bipedal locomotion).

Port of ``themis_mpc.centroidal_mpc`` from mpc-rl-main (Li, Wu et al., 2026,
arXiv:2606.05687) — the solver is robot-agnostic; only the ``MPCConfig``
defaults are re-targeted to the Unitree G1 (mass, foot geometry, force caps).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import torch
from torch import Tensor
from rl.decoupled.cd_mpc.admm_qp import QPSolution
from rl.decoupled.cd_mpc.contact_schedule import ContactSchedule
from rl.decoupled.cd_mpc.utils import jax_to_torch, skew, torch_to_jax


_JAX_PIMPC_MODULE = None
def _load_jax_pimpc():
    """Lazily import the JAX PiMPC solver (``g1_mpc.jax_pimpc``)."""
    global _JAX_PIMPC_MODULE
    if _JAX_PIMPC_MODULE is not None:
        return _JAX_PIMPC_MODULE
    from rl.decoupled.cd_mpc import jax_pimpc as _mod
    _JAX_PIMPC_MODULE = _mod
    return _mod

@dataclass
class MPCConfig:
    """Centroidal QP-MPC configuration (bipedal locomotion)."""

    N: int = 10
    dt: float = 0.07
    locomanipulation: bool = False # True to include hand force

    # Unitree G1 29-DoF (assets/g1_description/unitree_g1_tactile.xml, HOME
    # keyframe): mj_getTotalmass = 33.68 kg.  (33.34 is the older g1.xml.)
    mass: float = 33.68
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    # Foot geometry about the `left_foot`/`right_foot` MJCF site (ankle frame
    # x=+0.04): the 7 sole capsules span x in [-0.064, 0.142] i.e. +-0.10
    # about the site; 1 cm margin. Lateral: capsules at y=+-0.026, r=0.01.
    foot_x_toe:  float = 0.09
    foot_x_heel: float = 0.09
    foot_y_half: float = 0.03

    mu_foot: float = 0.6
    mu_foot_yaw: float = 0.1
    mu_hand: float = 0.6

    # ~1.5x weight-on-one-foot for a 33 kg robot (Themis used 800 N at 37 kg,
    # but G1's ankles saturate earlier; matches the validated mpc_mujoco cap).
    fz_max_foot: float = 500.0
    f_hand_max: float = 150.0

    # ============================================================
    # State residual
    # ============================================================
    # NOTE: slots 3:9 of the state are MOMENTA, not velocities (see `A` below:
    # the position row divides by `mass`).  Q_l therefore weights an error in
    # kg*m/s, i.e. it is m^2 ~ 1134x heavier than the same number would be on a
    # velocity error -- do not compare Q_c and Q_l at face value.
    Q_c: tuple[float, ...] = (100.0, 100.0, 200.0) # CoM position
    Q_l: tuple[float, ...] = (10.0, 10.0, 20.0) # linear momentum (m * v_com)
    Q_k: tuple[float, ...] = (50.0, 50.0, 50.0) # centroidal angular momentum
    Qf_scale: float = 10.0 # terminal state weight scale

    # ============================================================
    # Control residual
    # ============================================================
    R_f_foot: float = 1e-4 # foot contact force residual weight
    R_tau_foot: float = 1e-4 # foot contact torque residual weight
    R_delta: float = 1e-3 # control rate delta weight
    R_f_hand: float = 1e-1 # hand push force residual weight

    # ============================================
    # Solver Config -- JAX PiMPC is the ONLY backend here.  The upstream port's
    # condensed / sparse-ADMM paths and its `unconstrained` debug mode are
    # deliberately not carried over (see centroidal_mpc.py for those).
    # ============================================
    max_iter: int = 200
    pimpc_accel: bool = True
    pimpc_rho: float = 1.0
    pimpc_precondition: bool = True
    profile: bool = False


@dataclass
class MPCInput:
    """Centroidal QP-MPC input (bipedal locomotion)."""

    x0: Tensor # [B, NX] initial state
    schedule: ContactSchedule # contact schedule
    x_ref: Tensor # [B, N+1, NX] reference state trajectory
    u_ref: Tensor # [B, N, NU] reference control trajectory
    u_prev: Tensor # [B, NU] previous control input (for delta cost)

    # other body contact
    other_body_force: Tensor # [B, 3] world-frame force on the robot
    other_body_torque: Tensor # [B, 3] world-frame torque about the CoM

    # Linearization point for the moment arm (r_i - c) in B_k.  When None the
    # reference CoM path x_ref[:, :N, :3] is used.  NOTE this is a PREDICTION of
    # where the CoM will be, not a target: the modelling error is (c - c_bar) x f
    # and it lands entirely on the angular-momentum row.
    c_bar: Tensor | None = None # [B, N, 3]

    # hand contact -- only read when cfg.locomanipulation is True
    r_LH: Tensor | None = None # [B, 3] left hand contact point (world)
    r_RH: Tensor | None = None # [B, 3] right hand contact point (world)
    hand_contact: Tensor | None = None # [B, 2] 0/1 for left/right hand contact
    hand_force_lb: Tensor | None = None # [B, 6] per-env world-frame hand force lb
    hand_force_ub: Tensor | None = None # [B, 6] per-env world-frame hand force ub


@dataclass
class MPCOutput:
    """Outputs from the MPC solve."""
    u_star: Tensor
    x_pred: Tensor
    u_pred: Tensor
    feasible: Tensor
    obj: Tensor
    z_mpc: Tensor

class CentroidalMPC:
    """Contact-scheduled centroidal QP-MPC (bipedal locomotion)."""

    def __init__(self, cfg: MPCConfig, device: torch.device | str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = torch.float32
        self._jax_pimpc_solver = None
        self._jax_pimpc_shape: tuple[int, int] | None = None
        self.N = cfg.N
        # state: [com_pos, linear MOMENTUM (m*v), centroidal ANGULAR MOMENTUM]
        self.N_X = 9
        self.N_U = 18 if cfg.locomanipulation else 12 # control: [foot_6d_wrench, hand_3d_force]
        self._build_constants()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _tick(self) -> float: 
        self._sync()
        return time.perf_counter() * 1e3

    def _build_constants(self) -> None:
        """Build constant matrices for linearized dynamics and cost."""
        dt, mass = self.cfg.dt, self.cfg.mass
        g = torch.tensor(self.cfg.gravity, dtype=self.dtype, device=self.device)
        I = torch.eye(3, dtype=self.dtype, device=self.device)

        # state matrix [NX, NX] for linearized dynamics: x_{k+1} = A x_k + B u_k + c.
        self.A = torch.zeros(self.N_X, self.N_X, dtype=self.dtype, device=self.device)
        self.A[:3, :3]   = I
        self.A[:3, 3:6]  = (dt / mass) * I
        self.A[3:6, 3:6] = I
        self.A[6:9, 6:9] = I

        # constant term
        self.d = torch.zeros(self.N_X, dtype=self.dtype, device=self.device)
        self.d[:3] = 0.5 * dt**2 * g
        self.d[3:6] = mass * dt * g

        # state residual weight matrix [NX, NX]
        Q_diag = torch.tensor(
            list(self.cfg.Q_c) + list(self.cfg.Q_l) + list(self.cfg.Q_k),
            dtype=self.dtype, device=self.device)
        self.Q = torch.diag(Q_diag) # state cost
        self.Qf = self.cfg.Qf_scale * self.Q # terminal cost


        # control residual weight matrix [NU, NU]
        R_diag = torch.zeros(self.N_U, dtype=self.dtype, device=self.device)
        R_diag[0:3] = self.cfg.R_f_foot # foot contact force residual weight
        R_diag[3:6] = self.cfg.R_tau_foot # foot contact torque
        R_diag[6:9] = self.cfg.R_f_foot # foot contact force residual weight
        R_diag[9:12] = self.cfg.R_tau_foot # foot contact torque
        if self.cfg.locomanipulation:
            R_diag[12:15] = self.cfg.R_f_hand # hand push force residual weight
            R_diag[15:18] = self.cfg.R_f_hand # hand push force residual weight
        self.R = torch.diag(R_diag) # control cost
        self.R_delta = self.cfg.R_delta * torch.eye(self.N_U, dtype=self.dtype, device=self.device) # control delta cost

    def _build_control_matrix(self, mpc_in: MPCInput) -> Tensor:
        """
        Build the control matrix B for linearized dynamics: x_{k+1} = A x_k + B u_k + c.
        Returns [B, N, NX, NU].
        """
        B, N = mpc_in.schedule.batch_size, self.N
        dt, mass = self.cfg.dt, self.cfg.mass
        device, dtype = self.device, self.dtype
        I = torch.eye(3, dtype=dtype, device=device)
        # use xref as the cbar
        c_bar = (mpc_in.c_bar if mpc_in.c_bar is not None
                         else mpc_in.x_ref[:, :N, :3])
        
        # foot contact schedule
        sigma = mpc_in.schedule.sigma # [B, N, 4] 0/1 for left/right foot contact
        r_LF = mpc_in.schedule.r_LF # [B, N, 3] left foot contact position
        r_RF = mpc_in.schedule.r_RF # [B, N, 3] right foot contact

        # hand contact
        if self.cfg.locomanipulation:
            sigma_LH = mpc_in.hand_contact[:, 0].view(B, 1, 1) # [B, 1, 1] left hand contact
            sigma_RH = mpc_in.hand_contact[:, 1].view(B, 1, 1) # [B, 1, 1] right hand contact
            r_LH = mpc_in.r_LH.unsqueeze(1).expand(B, N, 3) # [B, N, 3] left hand contact position
            r_RH = mpc_in.r_RH.unsqueeze(1).expand(B, N, 3) # [B, N, 3] right hand contact position


        # construct dynamics matrices for each time step
        Bk = torch.zeros(B, N, self.N_X, self.N_U, dtype=dtype, device=device)
        for k in range(N):
            # reference
            c_k = c_bar[:, k, :] # [B, NX]
            sigma_LF_k = sigma[:, k, 0:1].unsqueeze(-1) # [B, 1]
            sigma_RF_k = sigma[:, k, 1:2].unsqueeze(-1) # [B, 1]

            Ef = torch.zeros(B, 3, self.N_U, dtype=dtype, device=device)
            Ef[:, :, 0:3] = sigma_LF_k * I.unsqueeze(0) # left foot force
            Ef[:, :, 6:9] = sigma_RF_k * I.unsqueeze(0) # right foot force
            if self.cfg.locomanipulation:
                Ef[:, :, 12:15] = sigma_LH * I.unsqueeze(0) # left hand force
                Ef[:, :, 15:18] = sigma_RH * I.unsqueeze(0) # right hand force

            Et = torch.zeros(B, 3, self.N_U, dtype=dtype, device=device)
            Et[:, :, 0:3] = sigma_LF_k * skew(r_LF[:, k, :] - c_k) # left foot torque
            Et[:, :, 3:6] = sigma_LF_k * I.unsqueeze(0) # left foot force
            Et[:, :, 6:9] = sigma_RF_k * skew(r_RF[:, k, :] - c_k) # right foot torque
            Et[:, :, 9:12] = sigma_RF_k * I.unsqueeze(0) # right foot force
            if self.cfg.locomanipulation:
                Et[:, :, 12:15] = sigma_LH * skew(r_LH[:, k, :] - c_k) # left hand torque
                Et[:, :, 15:18] = sigma_RH * skew(r_RH[:, k, :] - c_k) # right hand torque

            Bk[:, k, 0:3, :] = (dt ** 2 / (2 * mass)) * Ef
            Bk[:, k, 3:6, :] = dt * Ef
            Bk[:, k, 6:9, :] = dt * Et

        return Bk

    def _compute_c_extra(self, mpc_in: MPCInput) -> Tensor:
        """Centroidal disturbance from non-hand contacts."""
        cfg = self.cfg
        dt, m  = cfg.dt, cfg.mass
        device = self.device
        dtype  = self.dtype

        F_total = mpc_in.other_body_force
        tau_other = mpc_in.other_body_torque

        B_ = mpc_in.x0.shape[0]
        c_extra = torch.zeros(B_, 9, device=device, dtype=dtype)
        c_extra[:, 0:3] = (dt**2 / (2 * m)) * F_total
        c_extra[:, 3:6] = dt * F_total
        c_extra[:, 6:9] = dt * tau_other
        return c_extra

    def _make_output(
        self,
        mpc_in: MPCInput,
        x_pred: Tensor,
        u_pred: Tensor,
        sol,
    ) -> MPCOutput:
        """Pack the solve result into :class:`MPCOutput`."""
        N = self.cfg.N
        u_star  = u_pred[:, 0, :]
        sigma_0 = mpc_in.schedule.sigma[:, 0, :]
        x1      = x_pred[:, 0, :]
        x2      = x_pred[:, min(1, N - 1), :]
        z_mpc   = torch.cat([sigma_0, u_star, x1, x2], dim=-1)

        return MPCOutput(
            u_star=u_star,
            x_pred=x_pred,
            u_pred=u_pred,
            feasible=sol.converged,
            obj=sol.obj,
            z_mpc=z_mpc,
        )


    def solve(self, mpc_in: MPCInput) -> MPCOutput:
        """Solve the centroidal QP-MPC.  JAX PiMPC is the only backend."""
        return self._solve_jax_pimpc(mpc_in)

    def reset(self, env_ids: Tensor | None = None) -> None:
        """No-op: the JAX PiMPC backend keeps no warm-start state across solves.

        Kept so callers can reset() uniformly alongside the ADMM-backed MPCs.
        """
        del env_ids

    def _solve_jax_pimpc(self, mpc_in: MPCInput) -> MPCOutput:
        """Solve the centroidal QP-MPC using JAX PIMPC solver."""
        import jax.numpy as jnp
        n = self.N
        batch_size = mpc_in.x0.shape[0]
        nx, nu = self.N_X, self.N_U

        # build B matrix
        Bk = self._build_control_matrix(mpc_in) # [B, N, NX, NU]
        Bk = torch.nan_to_num(Bk, nan=0.0, posinf=0.0, neginf=0.0)
        # add extra disturbance term to the constant term d
        c_extra = self._compute_c_extra(mpc_in) # [B, NX]
        c_extra = torch.nan_to_num(c_extra, nan=0.0, posinf=0.0, neginf=0.0)
        d = self.d.unsqueeze(0) + c_extra # [B, NX]

        # box constraints for control inputs
        mu     = self.cfg.mu_foot
        fz_max = self.cfg.fz_max_foot
        yh     = self.cfg.foot_y_half
        toe    = self.cfg.foot_x_toe
        heel   = self.cfg.foot_x_heel
        mu_z   = self.cfg.mu_foot_yaw
        mu_h   = self.cfg.mu_hand
        f_h    = self.cfg.f_hand_max
        umin = torch.tensor([
            -mu*fz_max, -mu*fz_max, 0.0, -yh*fz_max, -heel*fz_max, -mu_z*fz_max,
            -mu*fz_max, -mu*fz_max, 0.0, -yh*fz_max, -heel*fz_max, -mu_z*fz_max,
        ], dtype=self.dtype, device=self.device)
        if self.cfg.locomanipulation:
            umin_hand = torch.tensor([
                -f_h, -mu_h*f_h, -mu_h*f_h,
                -f_h, -mu_h*f_h, -mu_h*f_h,
            ], dtype=self.dtype, device=self.device)
            umin = torch.cat([umin, umin_hand], dim=0)

        umax = torch.tensor([
            mu*fz_max, mu*fz_max, fz_max, yh*fz_max, toe*fz_max, mu_z*fz_max,
            mu*fz_max, mu*fz_max, fz_max, yh*fz_max, toe*fz_max, mu_z*fz_max,
        ], device=self.device, dtype=self.dtype)
        if self.cfg.locomanipulation:
            u_max_hand = torch.tensor([
                0.0, mu_h*f_h, mu_h*f_h,
                0.0, mu_h*f_h, mu_h*f_h,
            ], dtype=self.dtype, device=self.device)
            umax = torch.cat([umax, u_max_hand], dim=0)
        # box constraints per step
        sigma = mpc_in.schedule.sigma # [B, N, 4]
        lf_act = sigma[:, :, 0:1].permute(0, 2, 1).float() # [B, 1, N]
        rf_act = sigma[:, :, 1:2].permute(0, 2, 1).float() # [B, 1, N]

        umin_steps = umin.view(1, nu, 1).expand(batch_size, nu, n).clone()
        umax_steps = umax.view(1, nu, 1).expand(batch_size, nu, n).clone()
        umin_steps[:, :6,    :] *= lf_act
        umax_steps[:, :6,    :] *= lf_act
        umin_steps[:, 6:12,  :] *= rf_act
        umax_steps[:, 6:12,  :] *= rf_act
        # NOTE: assume hand contact is constant over the horizon
        if self.cfg.locomanipulation:
            lh_val  = mpc_in.hand_contact[:, 0:1].unsqueeze(-1)
            rh_val  = mpc_in.hand_contact[:, 1:2].unsqueeze(-1)
            lh_act  = lh_val.expand(batch_size, 1, n).float()
            rh_act  = rh_val.expand(batch_size, 1, n).float()
    
            if mpc_in.hand_force_lb is not None:
                assert mpc_in.hand_force_ub is not None
                umin_steps[:, 12:18, :] = (
                    mpc_in.hand_force_lb.to(self.dtype).view(batch_size, 6, 1).expand(batch_size, 6, n)
                )
                umax_steps[:, 12:18, :] = (
                    mpc_in.hand_force_ub.to(self.dtype).view(batch_size, 6, 1).expand(batch_size, 6, n)
                )
            umin_steps[:, 12:15, :] *= lh_act
            umax_steps[:, 12:15, :] *= lh_act
            umin_steps[:, 15:18, :] *= rh_act
            umax_steps[:, 15:18, :] *= rh_act

        # reference trajectory: [B, NX, N], [B, NU, N]
        xref = mpc_in.x_ref[:, 1:n+1, :].permute(0, 2, 1).contiguous() # [B, NX, N]
        uref = mpc_in.u_ref.permute(0, 2, 1).contiguous() # [B, NU, N]

        # load jax PIMPC
        jax_mod = _load_jax_pimpc()
        shape_key = (batch_size, n)
        if (self._jax_pimpc_solver is None
                or self._jax_pimpc_shape != shape_key):
            jax_dtype = jnp.float32 if self.dtype == torch.float32 else jnp.float64
            self._jax_pimpc_solver = jax_mod.PiMPCSolver(
                B=batch_size, N=n, nx=nx, nu=nu,
                maxiter=self.cfg.max_iter,
                accel=self.cfg.pimpc_accel,
                precondition=self.cfg.pimpc_precondition,
                dtype=jax_dtype,
                compile_now=True,
            )
            self._jax_pimpc_shape = shape_key

        # solve.  Key names are fixed by jax_pimpc._ARG_KEYS -- "B_s"/"e", NOT
        # "B"/"d"; a typo here raises KeyError INSIDE the try below and is
        # silently swallowed into the fallback.
        try:
            problem = {
                "A": torch_to_jax(self.A),
                "B_s": torch_to_jax(Bk),
                "e": torch_to_jax(d),
                "Wy": torch_to_jax(self.Q),
                "Wu": torch_to_jax(self.R),
                "Wdu": torch_to_jax(self.R_delta),
                "Wf": torch_to_jax(self.Qf),
                "x0": torch_to_jax(mpc_in.x0),
                "u0": torch_to_jax(mpc_in.u_prev),
                "yref": torch_to_jax(xref),
                "uref": torch_to_jax(uref),
                "umin_steps": torch_to_jax(umin_steps),
                "umax_steps": torch_to_jax(umax_steps),
            }
            x_jax, u_jax, res_jax = self._jax_pimpc_solver.solve(problem, rho=self.cfg.pimpc_rho)
            x_jax.block_until_ready()
            x_pred = jax_to_torch(x_jax).to(device=self.device, dtype=self.dtype)
            u_pred = jax_to_torch(u_jax).to(device=self.device, dtype=self.dtype)
            x_pred = x_pred[:, :, 1:].permute(0, 2, 1).contiguous()
            u_pred = u_pred.permute(0, 2, 1).contiguous()
            try:
                residual = float(res_jax)
            except Exception:
                residual = 0.0
            converged = torch.ones(batch_size, device=self.device, dtype=torch.bool)
            obj = torch.full((batch_size,), residual, device=self.device, dtype=self.dtype)
            pimpc_solution = QPSolution(z=torch.empty(0), converged=converged,
                                        iters=0, obj=obj)
        except (RuntimeError, ValueError, FloatingPointError) as e:
            # Deliberately NARROW: a bare `except Exception` here swallows
            # KeyError/AttributeError/TypeError from our own wiring and turns a
            # dead solver into a silent "fell back to reference" run.
            print(
                f"[CentroidalMPC] JAX PiMPC solve failed ({e}); "
                f"falling back to reference + warm-start u_prev."
            )
            x_pred = mpc_in.x_ref[:, 1:n+1, :].contiguous()
            u_pred = mpc_in.u_prev.unsqueeze(1).expand(batch_size, n, nu).contiguous()
            converged = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
            obj = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
            pimpc_solution = QPSolution(z=torch.empty(0), converged=converged, iters=0, obj=obj)

        x_pred = torch.nan_to_num(x_pred, nan=0.0, posinf=0.0, neginf=0.0)
        u_pred = torch.nan_to_num(u_pred, nan=0.0, posinf=0.0, neginf=0.0)
        return self._make_output(mpc_in, x_pred, u_pred, pimpc_solution)