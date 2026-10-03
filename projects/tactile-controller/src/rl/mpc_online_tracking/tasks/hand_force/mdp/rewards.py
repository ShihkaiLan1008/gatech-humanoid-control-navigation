"""Force-contact rewards (stage 2). SUM over the two hand command terms, like
every stage-1 hand term, so per-hand weights match.

PRIORITY ORDER (user decision 2026-07-28): **signal continuity FIRST,
magnitude accuracy second.** The tactile deliverable is an UNBROKEN force
signal — chattering contact is worse than a steady press at the wrong
magnitude. The term set implements that ordering:

* ``hand_contact_hold`` (PRIMARY): saturating ramp of the normal force
  against a LOW floor (3 N) — full marks for "a usable signal exists",
  FLAT above the floor (no pressure to press harder). Charges dropout
  DURATION.
* ``HandForceRate``: charges dropout FREQUENCY — a break-remake cycle is two
  large |Δf| events. hold + rate together make chatter strictly unprofitable.
* ``hand_force_tracking`` (σ 15) + ``hand_force_l1`` (small weight): the
  DEMOTED accuracy shaping. σ is wide on purpose: a steady 15 N press under a
  30 N command must score meaningfully (0.37), not like no contact (the old
  σ8 gave 0.03 — punishing steadiness almost as hard as absence, the exact
  inversion of the objective).
* ``contact_alignment``: binary make-contact stepping stone (grace-free —
  touching during transit earns immediately).

House rules still honored: NEVER square an unbounded contact force (exp is
bounded, the penalty is a CLAMPED norm); exp + penalty per quantity (run-1
lesson — the clamped L1 carries the far-field gradient the kernels lack).

Branch semantics (per hand):
* mask=1: tracking err = ‖f_meas − f_cmd_w‖ (no contact ⇒ err = ‖f_cmd‖ —
  not pressing IS the error, which also makes hovering unprofitable).
* mask=0: err = ‖f_meas‖ — identically 0 with parked cubes (free hands earn
  full kernels); future-proofs the fixed-wall stage where off-window touch is
  possible and wrong. ``hand_contact_hold`` is likewise 1 when nothing is
  commanded (f_cmd below floor: unmasked, ramp edges).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.reward_manager import RewardTermCfg

    from .commands import HandForceEePoseCommand


def _force_errors(
    env: ManagerBasedRlEnv, command_names: tuple[str, str]
) -> torch.Tensor:
    """(B, n_hands) masked-branch force-vector error norm [N]."""
    errs = []
    for name in command_names:
        term: HandForceEePoseCommand = env.command_manager.get_term(name)
        # Phantom windows take the idle branch: their press error would be
        # ‖f_cmd‖ regardless of behavior — unachievable, pure reward noise.
        masked = (term.contact_mask > 0.5) & (term.phantom < 0.5)
        f_meas = term.measured_force_w()
        err_press = (f_meas - term.force_cmd_w).norm(dim=-1)
        err_idle = f_meas.norm(dim=-1)
        errs.append(torch.where(masked, err_press, err_idle))
    return torch.stack(errs, dim=-1)


def hand_force_tracking(
    env: ManagerBasedRlEnv, std: float, command_names: tuple[str, str]
) -> torch.Tensor:
    """exp(-(err/std)^2) per hand, SUM over hands. In [0, 2]."""
    err = _force_errors(env, command_names)
    return torch.exp(-((err / std) ** 2)).sum(-1)


def hand_force_l1(
    env: ManagerBasedRlEnv, clamp: float, command_names: tuple[str, str]
) -> torch.Tensor:
    """Clamped force-error norm, SUM over hands [N]. Negative weight."""
    err = _force_errors(env, command_names)
    return err.clamp(max=clamp).sum(-1)


def hand_contact_hold(
    env: ManagerBasedRlEnv,
    command_names: tuple[str, str],
    floor: float = 3.0,
    grace_s: float = 1.0,
) -> torch.Tensor:
    """PRIMARY continuity term: saturating normal-force ramp vs a low floor.

    Per hand: clamp(f_n / floor, 0, 1) where f_n = (f_meas · p̂)⁺ is the
    measured press component along the commanded axis. Full marks whenever a
    ≥floor signal exists; FLAT above the floor (magnitude belongs to the
    tracking terms). Free passes (reward 1): commanded magnitude below the
    floor (unmasked, ramp-in start, ramp-out release) and the first
    ``grace_s`` of a window (reach transit — the target just moved to a new
    spot). SUM over hands, in [0, 2]."""
    out = None
    for name in command_names:
        term: HandForceEePoseCommand = env.command_manager.get_term(name)
        f_n = (
            (term.measured_force_w() * term.press_dir_w).sum(-1).clamp(min=0.0)
        )
        r = (f_n / floor).clamp(0.0, 1.0)
        free = (
            (term.force_cmd_mag < floor)
            | (term._contact_age < grace_s)
            | (term.phantom > 0.5)  # nothing there to hold — free pass
        )
        r = torch.where(free, torch.ones_like(r), r)
        out = r if out is None else out + r
    return out


def contact_alignment(
    env: ManagerBasedRlEnv, command_names: tuple[str, str]
) -> torch.Tensor:
    """mask=1 → measured contact (0/1); mask=0 → 1. SUM over hands, [0, 2].

    The make-contact shaping term: force tracking alone is gradient-poor
    before touch (any not-touching pose scores the same ‖f_cmd‖ error)."""
    out = None
    for name in command_names:
        term: HandForceEePoseCommand = env.command_manager.get_term(name)
        masked = (term.contact_mask > 0.5) & (term.phantom < 0.5)
        contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for sensor in term._sensors:
            found = sensor.data.found
            if found is not None:
                contact |= (found > 0).any(dim=-1)
        contact = contact.float()
        r = torch.where(masked, contact, torch.ones_like(contact))
        out = r if out is None else out + r
    return out


class HandForceRate:
    """Clamped per-step measured-force change, SUM over hands [N]. Negative
    weight — tactile-hardware protection: the policy must modulate force
    smoothly (command-side ramps make smooth tracking feasible)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        self._prev = torch.zeros(env.num_envs, 2, 3, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_names: tuple[str, str],
        clamp: float = 20.0,
    ) -> torch.Tensor:
        f = torch.stack(
            [
                env.command_manager.get_term(n).measured_force_w()
                for n in command_names
            ],
            dim=1,
        )
        rate = (f - self._prev).norm(dim=-1).clamp(max=clamp).sum(-1)
        self._prev = f.detach()
        return rate

    def reset(self, env_ids: torch.Tensor) -> None:
        self._prev[env_ids] = 0.0
