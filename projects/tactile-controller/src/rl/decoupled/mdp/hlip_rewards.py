"""Rewards on the H-LIP CoM-proxy reference of :class:`FootHoldHLIPCommand`.

All read the command term's per-step state (stance-site frame); nothing here
is a tuning constant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from .commands.foot_hold_hlip import FootHoldHLIPCommand


def _hlip(env: ManagerBasedRlEnv, command_name: str) -> FootHoldHLIPCommand:
    return env.command_manager.get_term(command_name)


def hlip_com_pos(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    """exp(-|p - p_ref|^2 / std^2) on the proxy position, stance frame. [B]"""
    t = _hlip(env, command_name)
    err2 = (t.meas_pos - t.ref_pos).square().sum(-1)
    return torch.exp(-err2 / std ** 2)


def hlip_com_pos_l2(env: ManagerBasedRlEnv, command_name: str, clamp: float) -> torch.Tensor:
    """|p - p_ref|^2 clamped at ``clamp`` m^2 (negative weight). [B]"""
    t = _hlip(env, command_name)
    return (t.meas_pos - t.ref_pos).square().sum(-1).clamp(max=clamp)


def hlip_com_vel(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    """exp(-|v - v_ref|^2 / std^2) on the proxy velocity, stance frame. [B]"""
    t = _hlip(env, command_name)
    err2 = (t.meas_vel - t.ref_vel).square().sum(-1)
    return torch.exp(-err2 / std ** 2)


def hlip_com_vel_l2(env: ManagerBasedRlEnv, command_name: str, clamp: float) -> torch.Tensor:
    """|v - v_ref|^2 clamped at ``clamp`` (m/s)^2 (negative weight). [B]"""
    t = _hlip(env, command_name)
    return (t.meas_vel - t.ref_vel).square().sum(-1).clamp(max=clamp)


def orbital_energy_ssp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    """One-sided exp(-max(E_ref - E, 0)^2 / std^2) while the swing is active,
    1 outside the SSP. E in m^2/s^2. [B]"""
    t = _hlip(env, command_name)
    short = (t.E_ref - t.E_meas).clamp(min=0.0)
    r = torch.exp(-short.square() / std ** 2)
    return torch.where(t.ssp_active, r, torch.ones_like(r))


def orbital_energy_ssp_l1(env: ManagerBasedRlEnv, command_name: str, clamp: float) -> torch.Tensor:
    """max(E_ref - E, 0) clamped at ``clamp`` while the swing is active
    (negative weight); 0 outside the SSP. [B]"""
    t = _hlip(env, command_name)
    short = (t.E_ref - t.E_meas).clamp(min=0.0, max=clamp)
    return torch.where(t.ssp_active, short, torch.zeros_like(short))


def liftoff_energy_l1(env: ManagerBasedRlEnv, command_name: str, clamp: float) -> torch.Tensor:
    """max(E_ref - E, 0) clamped at ``clamp`` on the lift-off step only
    (negative weight). Charges a wind-up that did not deliver the orbit's
    energy before the swing foot left the ground. [B]"""
    t = _hlip(env, command_name)
    short = (t.E_ref - t.E_meas).clamp(min=0.0, max=clamp)
    return torch.where(t.liftoff_event, short, torch.zeros_like(short))
