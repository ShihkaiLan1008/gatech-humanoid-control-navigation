"""Control-Lyapunov-Function evaluator for relative-degree-2 outputs.

Port of Zolkin1/robot_rl `mdp/commands/traj_tracking/clf.py` (the generic
implementation shared by their walking / running / waving / bow / bend_up CLF
tasks; the LIP task's copy lives in a `clf_cmd/` directory that is absent from
the public tree). Same math, mjlab APIs, plus one addition we need that they do
not: `invalidate()`, because our reference is discontinuously REPLACED at every
chain switch, not only at env reset.

The construction (CLF-RL, arXiv:2508.09354):

    eta = [y_d - y ; dy_d - dy]   interleaved per channel
    V   = eta^T P eta            P from CARE on a double integrator
    r_v = exp(-V / sigma_v)                       sigma_v = mu_max(P) eta_max^2
    r_vdot = -clip((Vdot + lam V) / sigma_vdot, 0, 1)

`A`/`B` are BLOCK-DIAGONAL (one 2x2 double integrator per output channel) and
`Q`/`R` are diagonal, so `P` is block-diagonal too and `V` is a SUM of
independent per-channel quadratic forms. Consequence worth stating plainly:
CARE only sets the position-vs-velocity trade-off WITHIN a channel — the
cross-channel weighting is entirely `Q_weights`, which is why that dict is a
declaration-surface input and never a constant in here.

NOTHING in this module is a tuning constant. Every weight arrives through the
constructor; the dicts live in the task's `env_configs/`.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import solve_continuous_are

from mjlab.utils.lab_api.math import quat_box_minus

# Channel-name suffixes. A quaternion channel contributes 4 entries to the
# POSITION name list (x, y, z, w on the manifold) and 3 to the VELOCITY list
# (x, y, z in the tangent space) — the same split their code uses.
_ORI_MANIFOLD = (":ori_x", ":ori_y", ":ori_z", ":ori_w")
_ORI_TANGENT = (":ori_x", ":ori_y", ":ori_z")


class CLF:
    """Evaluates V and Vdot for a named set of relative-degree-2 outputs.

    Args:
        sim_dt: seconds between `compute` calls (the finite-difference step).
        batch_size: number of envs.
        pos_output_names: ordered manifold channels; quaternions occupy 4.
        vel_output_names: ordered tangent channels; quaternions occupy 3.
        Q_weights: {vel_channel_name: (q_pos, q_vel)}. Must cover every entry
            of `vel_output_names`; extra keys are ignored, so one dict can
            serve several `y` compositions (their pattern — the `y` layout is
            defined by the reference data, the weights only look up by name).
        R_weights: {vel_channel_name: (r,)} or {name: r}.
        vdot_clamp: |Vdot| above this reads as a reference discontinuity or a
            reset artefact and is zeroed (their guard, default 1e4).
    """

    def __init__(
        self,
        sim_dt: float,
        batch_size: int,
        pos_output_names: tuple[str, ...],
        vel_output_names: tuple[str, ...],
        Q_weights: dict[str, tuple[float, float]],
        R_weights: dict[str, tuple[float, ...] | float],
        device: torch.device | str,
        vdot_clamp: float = 1.0e4,
    ) -> None:
        self.device = device
        self.sim_dt = float(sim_dt)
        self.pos_output_names = tuple(pos_output_names)
        self.vel_output_names = tuple(vel_output_names)
        self.n_channels = len(self.vel_output_names)
        self.n_states = 2 * self.n_channels
        self.vdot_clamp = float(vdot_clamp)

        missing = [n for n in self.vel_output_names if n not in Q_weights]
        if missing:
            raise KeyError(f"Q_weights is missing channels: {missing}")

        q = np.ones(self.n_states)
        r = np.ones(self.n_channels)
        for i, name in enumerate(self.vel_output_names):
            q[2 * i], q[2 * i + 1] = Q_weights[name]
            rw = R_weights[name]
            r[i] = float(rw[0] if isinstance(rw, (tuple, list)) else rw)

        # One double integrator per channel: block-diagonal A, B.
        a_blk = np.array([[0.0, 1.0], [0.0, 0.0]])
        b_blk = np.array([[0.0], [1.0]])
        a_full = np.kron(np.eye(self.n_channels), a_blk)
        b_full = np.kron(np.eye(self.n_channels), b_blk)
        p_np = solve_continuous_are(a_full, b_full, np.diag(q), np.diag(r))
        p_np = 0.5 * (p_np + p_np.T)                       # symmetrize

        self.P = torch.as_tensor(p_np, dtype=torch.float32, device=device)
        self.lambda_max = float(np.linalg.eigvalsh(p_np).max())
        self.norm_P = float(np.linalg.norm(p_np, ord=2))

        self._build_quat_index()
        self._build_subgroup_index()

        # Three-sample history for the second-order backward difference.
        self._v_hist = torch.zeros(batch_size, 3, device=device)
        self._n_hist = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.v_subgroups: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------- indexing
    def _build_quat_index(self) -> None:
        """Group the `:ori_*` channels per body, on the manifold and in the
        tangent space, and mark everything else for plain subtraction."""
        self._quat_manifold: dict[str, list[int]] = {}
        self._quat_tangent: dict[str, list[int]] = {}
        self._plain_manifold = torch.ones(
            len(self.pos_output_names), dtype=torch.bool, device=self.device)
        self._plain_tangent = torch.ones(
            self.n_channels, dtype=torch.bool, device=self.device)

        for i, name in enumerate(self.pos_output_names):
            for suffix in _ORI_MANIFOLD:
                if name.endswith(suffix):
                    self._quat_manifold.setdefault(
                        name[: -len(suffix)], []).append(i)
                    self._plain_manifold[i] = False
        for i, name in enumerate(self.vel_output_names):
            for suffix in _ORI_TANGENT:
                if name.endswith(suffix):
                    self._quat_tangent.setdefault(
                        name[: -len(suffix)], []).append(i)
                    self._plain_tangent[i] = False

        for body, idx in self._quat_manifold.items():
            if len(idx) != 4:
                raise ValueError(
                    f"'{body}' needs ori_x/y/z/w in pos_output_names, got {len(idx)}")
            if len(self._quat_tangent.get(body, [])) != 3:
                raise ValueError(
                    f"'{body}' needs ori_x/y/z in vel_output_names")
        n_plain_m = int(self._plain_manifold.sum())
        n_plain_t = int(self._plain_tangent.sum())
        if n_plain_m != n_plain_t:
            raise ValueError(
                "non-quaternion channels must appear in the same order in both "
                f"name lists (manifold {n_plain_m} vs tangent {n_plain_t})")
        # Slice order = declaration order (w, x, y, z as emitted by the cfg
        # assembly) = the wxyz layout quat_box_minus expects. sorted() keeps
        # appearance order because a quat's four entries are consecutive.
        self._quat_manifold = {k: sorted(v) for k, v in self._quat_manifold.items()}

    def _build_subgroup_index(self) -> None:
        """Per-family eta indices, for logging WHICH part of `y` is off. Their
        diagnostic; computed with an identity P so the numbers read as plain
        squared errors, not Lyapunov values.

        POSITION slots and VELOCITY slots go to SEPARATE groups (theirs:
        pelvis_pos vs pelvis_lin_vel, joint_pos vs joint_vel, ...) — the
        per-family reward kernels give the two different sigmas, and Q
        calibration needs the two error scales separately."""
        groups: dict[str, list[int]] = {}
        for i, name in enumerate(self.vel_output_names):
            head = name.split(":")[0]
            if ":ori_" in name:
                pos_key, vel_key = f"{head}_ori", f"{head}_ang_vel"
            elif head == "joint":
                pos_key, vel_key = "joint_pos", "joint_vel"
            else:
                pos_key, vel_key = f"{head}_pos", f"{head}_lin_vel"
            groups.setdefault(pos_key, []).append(2 * i)
            groups.setdefault(vel_key, []).append(2 * i + 1)
        self._subgroups = {
            k: torch.tensor(v, dtype=torch.long, device=self.device)
            for k, v in groups.items()
        }

    # ------------------------------------------------------------- evaluate
    def y_error(self, y_act: torch.Tensor, y_nom: torch.Tensor) -> torch.Tensor:
        """Manifold-aware position error -> tangent space, shape [B, n_channels]."""
        err = torch.zeros(y_act.shape[0], self.n_channels, device=y_act.device)
        err[:, self._plain_tangent] = (y_act[:, self._plain_manifold]
                                       - y_nom[:, self._plain_manifold])
        for body, m_idx in self._quat_manifold.items():
            t_idx = self._quat_tangent[body]
            err[:, t_idx] = quat_box_minus(y_act[:, m_idx], y_nom[:, m_idx])
        return err

    def compute(
        self,
        y_act: torch.Tensor,
        y_nom: torch.Tensor,
        dy_act: torch.Tensor,
        dy_nom: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the history one step and return (V, Vdot), each [B].

        Call EXACTLY once per env step — the finite difference carries state.
        """
        eta = torch.zeros(y_act.shape[0], self.n_states, device=y_act.device)
        eta[:, 0::2] = self.y_error(y_act, y_nom)
        eta[:, 1::2] = dy_act - dy_nom
        v = torch.einsum("bi,ij,bj->b", eta, self.P, eta)

        self.v_subgroups = {
            name: eta[:, idx].square().sum(dim=-1)      # identity P on purpose
            for name, idx in self._subgroups.items()
        }

        self._v_hist = torch.roll(self._v_hist, shifts=1, dims=1)
        self._v_hist[:, 0] = v.detach()
        self._n_hist = torch.clamp(self._n_hist + 1, max=3)

        dt = self.sim_dt
        three = (3.0 * self._v_hist[:, 0] - 4.0 * self._v_hist[:, 1]
                 + self._v_hist[:, 2]) / (2.0 * dt)
        two = (self._v_hist[:, 0] - self._v_hist[:, 1]) / dt
        vdot = torch.where(self._n_hist >= 3, three,
                           torch.where(self._n_hist == 2, two,
                                       torch.zeros_like(two)))
        # A reference switch or a state write makes the difference meaningless.
        vdot = torch.where(vdot.abs() > self.vdot_clamp,
                           torch.zeros_like(vdot), vdot)
        return v, vdot

    def invalidate(self, env_ids: torch.Tensor | None = None) -> None:
        """Drop the finite-difference history so Vdot resumes from scratch.

        Call on env reset AND on any discontinuous replacement of the
        reference (for us: the chain switch). Their code only ever resets on
        episode reset because their reference is a periodic clock that is never
        swapped mid-episode.
        """
        if env_ids is None:
            self._v_hist.zero_()
            self._n_hist.zero_()
        else:
            self._v_hist[env_ids] = 0.0
            self._n_hist[env_ids] = 0

    # ---------------------------------------------------------- normalizers
    def sigma_v(self, eta_max: float, eps: float = 1e-6) -> float:
        """Worst-case V over the declared operating range: V <= mu_max ||eta||^2."""
        return self.lambda_max * eta_max ** 2 + eps

    def sigma_vdot(self, eta_max: float, eta_dot_max: float, lam: float,
                   eps: float = 1e-6) -> float:
        """Worst-case (Vdot + lam V): Vdot = 2 eta^T P d(eta) <= 2||P|| |eta| |d eta|."""
        return (2.0 * self.norm_P * eta_max * eta_dot_max
                + lam * self.lambda_max * eta_max ** 2 + eps)
