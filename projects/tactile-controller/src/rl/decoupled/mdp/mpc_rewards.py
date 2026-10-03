"""MPC-plan tracking rewards for the joint_mpc task, ported from
mpc_online_tracking/mdp/mpc_command.py and re-targeted to
:class:`~rl.decoupled.mdp.commands.step_mpc.StepMPCCommand`.

These score the LOWER body against the QP's press-consistent plan. Because
``StepMPCCommand`` builds its ``x_ref`` from the step clip's own future root
states, the plan coincides with the imitation reference when no hand force is
commanded — these terms only pull AWAY from the clip where the QP says the
body must accommodate a commanded press (CoM shift, GRF redistribution).

``mpc_foot_placement`` is deliberately NOT ported: the MPC's footholds ARE
the step command's targets here, already priced at σ0.05 by
``track_feet_pos`` (the MPC copy would be a looser σ0.2 duplicate).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mjlab.sensor import ContactSensor

from .commands.locomotion_mpc import LocoMPCCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.reward_manager import RewardTermCfg


def mpc_grf_tracking(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
    grf_sensor_name: str = "feet_ground_contact",
) -> Tensor:
    """GRF tracking error vs. MPC reference — use with weight = 0.0."""
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, LocoMPCCommand):
        return torch.zeros(env.num_envs, device=env.device)

    grf_ref: Tensor = term._grf_ref

    sensor: ContactSensor = env.scene[grf_sensor_name]
    if sensor.data.force is None:
        return torch.zeros(env.num_envs, device=env.device)

    force = sensor.data.force
    # mjlab >= 1.5 contact sensors report the force applied BY the primary ON
    # the secondary (feet pressing the terrain: Fz < 0 in stance — measured
    # 2026-07-21). The MPC plans the ground reaction ON the robot, so negate.
    grf_actual = -force.reshape(env.num_envs, 6)

    rms_err = (grf_actual - grf_ref).pow(2).mean(dim=-1).sqrt()
    return -rms_err


def mpc_grf_press_tracking(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
    grf_sensor_name: str = "feet_ground_contact",
    std: float = 80.0,
) -> Tensor:
    """Press-gated GRF exp kernel — pays the legs for reproducing the QP's
    press-consistent feet plan (measured 2026-08-11: a 20 N press pins the QP
    hand slot and the feet counter it ~1:1 in fx + a CoP shift; the always-on
    ``mpc_grf_tracking`` at 0.002·(−RMS) priced that bracing at metric grade,
    so it was sold). Free-passes 1 outside a live hand gate: the term prices
    ONLY the bracing delta, leaving the no-press GRF economics untouched.
    ``mpc_grf_tracking`` is this kernel's unbounded L1 pair — keep both.

    ``std`` is MEASURED, not chosen: the resumed vel baseline runs at GRF-rms
    p10/p50/p90 = 32/79/166 N vs the QP plan (knee cfg, 2026-08-12 probe) —
    σ=80 puts the p50 at ≈ exp(−1) ≈ 0.37 mid-slope; the first σ=25 draft
    was gradient-dead everywhere (kernel ≤ 0.002 at p10)."""
    term = env.command_manager.get_term(command_name)
    gate = getattr(term, "_gate", None)
    if term is None or gate is None or not isinstance(term, LocoMPCCommand):
        return torch.ones(env.num_envs, device=env.device)

    sensor: ContactSensor = env.scene[grf_sensor_name]
    if sensor.data.force is None:
        return torch.ones(env.num_envs, device=env.device)
    grf_actual = -sensor.data.force.reshape(env.num_envs, 6)

    rms_err = (grf_actual - term._grf_ref).pow(2).mean(dim=-1).sqrt()
    r = torch.exp(-((rms_err / std) ** 2))
    pressing = gate.sum(dim=-1) > 0.5
    return torch.where(pressing, r, torch.ones_like(r))


class MpcAngMomTracking:
    """Soft MPC-guided angular-momentum regulation reward."""

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRlEnv") -> None:
        self._env = env
        self._k_prev: Tensor = torch.zeros(env.num_envs, 3, device=env.device)

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        command_name: str = "loco_mpc",
        w_k: float = 1.0,
        w_kdot: float = 0.01,
        q_k: tuple[float, float, float] = (1.0, 1.0, 0.5),
        q_kdot: tuple[float, float, float] = (0.1, 0.1, 0.05),
        lookahead_fracs: tuple[float, ...] | None = None,
        lookahead_weights: tuple[float, ...] | None = None,
    ) -> Tensor:
        """Angular-momentum tracking loss (≤ 0)."""
        term = env.command_manager.get_term(command_name)
        if term is None or not isinstance(term, LocoMPCCommand):
            return torch.zeros(env.num_envs, device=env.device)

        av = term._robot.data.root_link_ang_vel_w
        k_t = av @ term._I_approx

        dt = env.step_dt
        k_dot_t = (k_t - self._k_prev) / dt
        self._k_prev = k_t.detach()

        Q_k  = torch.tensor(q_k,    device=env.device, dtype=torch.float32)
        Q_dk = torch.tensor(q_kdot, device=env.device, dtype=torch.float32)

        if lookahead_fracs is None:
            k_mpc     = term._k_mpc_target
            k_dot_mpc = term._k_dot_mpc_target
            ek  = k_t   - k_mpc
            edk = k_dot_t - k_dot_mpc
            loss_k    = (ek.pow(2)  * Q_k ).sum(dim=-1)
            loss_kdot = (edk.pow(2) * Q_dk).sum(dim=-1)
            return -(w_k * loss_k + w_kdot * loss_kdot)

        N = term.cfg.mpc_horizon
        k_traj    = term._k_traj
        kdot_traj = term._kdot_traj

        if lookahead_weights is None:
            n = len(lookahead_fracs)
            lookahead_weights = tuple(1.0 / n for _ in range(n))
        if len(lookahead_weights) != len(lookahead_fracs):
            raise ValueError(
                "lookahead_weights must match lookahead_fracs length "
                f"({len(lookahead_weights)} vs {len(lookahead_fracs)})"
            )

        total = torch.zeros(env.num_envs, device=env.device)
        for frac, weight in zip(lookahead_fracs, lookahead_weights):
            t = frac * (N - 1)
            idx = min(max(int(t), 0), N - 2)
            alpha = min(max(t - idx, 0.0), 1.0)
            k_ref  = (1 - alpha) * k_traj[:, idx]    + alpha * k_traj[:, idx + 1]
            kd_ref = (1 - alpha) * kdot_traj[:, idx] + alpha * kdot_traj[:, idx + 1]
            ek  = k_t   - k_ref
            edk = k_dot_t - kd_ref
            loss_k    = (ek.pow(2)  * Q_k ).sum(dim=-1)
            loss_kdot = (edk.pow(2) * Q_dk).sum(dim=-1)
            total = total + weight * (w_k * loss_k + w_kdot * loss_kdot)

        return -total

    def reset(self, env_ids: Tensor) -> None:
        """Clear finite-difference history for reset environments."""
        self._k_prev[env_ids] = 0.0

def mpc_com_tracking(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
    w_pos: float = 1.0,
    w_vel: float = 0.5,
    lookahead_fracs: tuple[float, ...] | None = None,
    lookahead_weights: tuple[float, ...] | None = None,
) -> Tensor:
    """Reward for tracking the MPC-predicted CoM position and velocity."""
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, LocoMPCCommand):
        return torch.zeros(env.num_envs, device=env.device)

    c_cur = term._robot.data.root_link_pos_w
    v_cur = term._robot.data.root_link_lin_vel_w

    if lookahead_fracs is None:
        e_pos = c_cur - term._com_mpc_target
        e_vel = v_cur - term._com_vel_mpc_target
        return torch.exp(
            -w_pos * e_pos.pow(2).sum(dim=-1)
            -w_vel * e_vel.pow(2).sum(dim=-1)
        )

    N = term.cfg.mpc_horizon
    com_traj = term._com_traj
    vel_traj = term._vel_traj

    if lookahead_weights is None:
        k = len(lookahead_fracs)
        lookahead_weights = tuple(1.0 / k for _ in range(k))
    if len(lookahead_weights) != len(lookahead_fracs):
        raise ValueError(
            "lookahead_weights must match lookahead_fracs length "
            f"({len(lookahead_weights)} vs {len(lookahead_fracs)})"
        )

    total = torch.zeros(env.num_envs, device=env.device)
    for frac, weight in zip(lookahead_fracs, lookahead_weights):
        t = frac * (N - 1)
        idx = min(max(int(t), 0), N - 2)
        alpha = min(max(t - idx, 0.0), 1.0)
        c_ref = (1 - alpha) * com_traj[:, idx] + alpha * com_traj[:, idx + 1]
        v_ref = (1 - alpha) * vel_traj[:, idx] + alpha * vel_traj[:, idx + 1]
        e_pos = c_cur - c_ref
        e_vel = v_cur - v_ref
        total = total + weight * (
            w_pos * e_pos.pow(2).sum(dim=-1)
            + w_vel * e_vel.pow(2).sum(dim=-1)
        )

    return torch.exp(-total)

def mpc_com_vel_tracking(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
    w_vel: float = 1.0,
    lookahead_fracs: tuple[float, ...] | None = (0.0, 0.5, 1.0),
    lookahead_weights: tuple[float, ...] | None = None,
) -> Tensor:
    """Reward for tracking the MPC-predicted CoM **velocity** trajectory."""
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, LocoMPCCommand):
        return torch.zeros(env.num_envs, device=env.device)

    v_cur = term._robot.data.root_link_lin_vel_w

    if lookahead_fracs is None:
        e_vel = v_cur - term._com_vel_mpc_target
        return torch.exp(-w_vel * e_vel.pow(2).sum(dim=-1))

    N = term.cfg.mpc_horizon
    vel_traj = term._vel_traj

    if lookahead_weights is None:
        k = len(lookahead_fracs)
        lookahead_weights = tuple(1.0 / k for _ in range(k))
    if len(lookahead_weights) != len(lookahead_fracs):
        raise ValueError(
            "lookahead_weights must match lookahead_fracs length "
            f"({len(lookahead_weights)} vs {len(lookahead_fracs)})"
        )

    total = torch.zeros(env.num_envs, device=env.device)
    for frac, weight in zip(lookahead_fracs, lookahead_weights):
        t = frac * (N - 1)
        idx = min(max(int(t), 0), N - 2)
        alpha = min(max(t - idx, 0.0), 1.0)
        v_ref = (1 - alpha) * vel_traj[:, idx] + alpha * vel_traj[:, idx + 1]
        e_vel = v_cur - v_ref
        total = total + weight * w_vel * e_vel.pow(2).sum(dim=-1)

    return torch.exp(-total)

def mpc_ang_vel_tracking(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
    w_ang: float = 2.0,
    lookahead_fracs: tuple[float, ...] | None = (0.0, 0.5, 1.0),
    lookahead_weights: tuple[float, ...] | None = None,
) -> Tensor:
    """Reward for tracking the MPC-predicted body **angular velocity** trajectory."""
    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, LocoMPCCommand):
        return torch.zeros(env.num_envs, device=env.device)

    omega_cur = term._robot.data.root_link_ang_vel_w

    I_inv = torch.linalg.inv(term._I_approx)

    if lookahead_fracs is None:
        omega_mpc = term._k_mpc_target @ I_inv
        e = omega_cur - omega_mpc
        return torch.exp(-w_ang * e.pow(2).sum(dim=-1))

    N = term.cfg.mpc_horizon
    k_traj = term._k_traj

    if lookahead_weights is None:
        k_n = len(lookahead_fracs)
        lookahead_weights = tuple(1.0 / k_n for _ in range(k_n))
    if len(lookahead_weights) != len(lookahead_fracs):
        raise ValueError(
            "lookahead_weights must match lookahead_fracs length "
            f"({len(lookahead_weights)} vs {len(lookahead_fracs)})"
        )

    total = torch.zeros(env.num_envs, device=env.device)
    for frac, weight in zip(lookahead_fracs, lookahead_weights):
        t = frac * (N - 1)
        idx = min(max(int(t), 0), N - 2)
        alpha = min(max(t - idx, 0.0), 1.0)
        k_ref = (1 - alpha) * k_traj[:, idx] + alpha * k_traj[:, idx + 1]
        omega_ref = k_ref @ I_inv
        e = omega_cur - omega_ref
        total = total + weight * w_ang * e.pow(2).sum(dim=-1)

    return torch.exp(-total)

def mpc_foot_placement_tracking(
    env: "ManagerBasedRlEnv",
    command_name: str = "loco_mpc",
    sigma: float = 0.2,
    command_threshold: float = 0.1,
) -> Tensor:
    """Reward swing feet for tracking MPC-predicted landing positions."""
    import math as _math

    term = env.command_manager.get_term(command_name)
    if term is None or not isinstance(term, LocoMPCCommand):
        return torch.zeros(env.num_envs, device=env.device)

    B = env.num_envs
    device = env.device

    gait_phase = env.command_manager.get_term("gait").phase
    threshold = _math.cos(_math.pi * term.cfg.duty_factor)
    lf_in_swing = (gait_phase + _math.pi).sin() >= threshold
    rf_in_swing = gait_phase.sin() >= threshold

    site_pos = term._robot.data.site_pos_w[:, term._site_ids, :]
    r_lf = site_pos[:, term._lf_site_local_idx, :2]
    r_rf = site_pos[:, term._rf_site_local_idx, :2]

    lf_err = (r_lf - term._lf_landing_target[:, :2]).pow(2).sum(dim=-1)
    rf_err = (r_rf - term._rf_landing_target[:, :2]).pow(2).sum(dim=-1)

    lf_reward = torch.exp(-lf_err / (sigma ** 2))
    rf_reward = torch.exp(-rf_err / (sigma ** 2))

    lf_active = lf_in_swing & term._lf_landing_valid
    rf_active = rf_in_swing & term._rf_landing_valid
    n_active = lf_active.float() + rf_active.float()

    total = lf_active.float() * lf_reward + rf_active.float() * rf_reward
    reward = torch.where(n_active > 0, total / n_active,
                         torch.ones(B, device=device))

    vel_cmd = env.command_manager.get_command(term.cfg.vel_cmd_name)
    if vel_cmd is not None:
        cmd_mag = vel_cmd[:, :2].norm(dim=1) + vel_cmd[:, 2].abs()
        reward = torch.where(cmd_mag > command_threshold, reward,
                             torch.ones(B, device=device))

    return reward

def mpc_feet_gait(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    offset: list[float],
    threshold: float,
    command_name: str = "twist",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Phase-locked gait reward.

    The clock comes from the "gait" command term, so there is no `period` here
    — the period lives once, on that term. `threshold` MUST be
    `cos(pi * duty_factor)` of the MPC's contact schedule (see
    `GAIT_THRESHOLD` in the task's command_cfgs), otherwise the reward asks for
    a different stance window than the MPC plans for.
    """
    phi_base = env.command_manager.get_term("gait").phase  # (B,)

    offsets_t = torch.tensor(offset, device=env.device, dtype=torch.float32)
    phi = (
        phi_base.unsqueeze(1)
        + 2.0 * math.pi * offsets_t.unsqueeze(0)
    )
    desired_contact = (phi.sin() < threshold).float()

    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None, (
        f"ContactSensor '{sensor_name}' has no 'found' data"
    )
    actual_contact = (sensor.data.found > 0).float()

    reward = torch.mean(
        desired_contact * actual_contact
        + (1.0 - desired_contact) * (1.0 - actual_contact),
        dim=-1,
    )

    cmd = env.command_manager.get_command(command_name)
    if cmd is not None:
        cmd_mag = cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()
        reward *= (cmd_mag > command_threshold).float()

    return reward