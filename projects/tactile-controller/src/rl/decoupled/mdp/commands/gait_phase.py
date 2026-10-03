"""Gait-clock command: a free-running phase that freezes to stance when standing.

Port of the ``mpc_online_tracking`` observation-side clock (``mdp/phase.py``)
into a proper :class:`CommandTerm`.  Same behaviour, but the state is OWNED by
a manager term instead of being stashed on ``env`` as ``env._gait_phase``:

* the clock is integrated exactly ONCE per env step (the observation version
  advanced on every obs-group evaluation, which is why it needed a separate
  side-effect-free ``phase_readout`` for the critic group);
* reset uses the manager's ``reset(env_ids)`` hook instead of inferring an
  episode boundary from ``episode_length_buf`` going backwards;
* the integration step is the ``dt`` the command manager passes in, so the
  reset-path call (``compute(dt=0.0)``) no longer advances the clock;
* consumers read it through the manager: ``get_command("gait")`` for the
  (sin, cos) observation, ``get_term("gait").phase`` for the raw radians.

DECLARATION ORDER: put this term AFTER the twist term in the commands cfg.
``CommandManager`` iterates terms in cfg-dict insertion order, so declaring it
first would make it read a one-step-stale twist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

TWO_PI = 2.0 * math.pi


class GaitPhaseCommand(CommandTerm):
    """Free gait clock, gated on the twist command's magnitude.

    While the commanded twist is above ``command_threshold`` the phase runs at
    ``2*pi/period``.  Below it the phase decays geometrically back to 0 (the
    touchdown boundary), so a standing robot sits at a fixed point of the clock
    instead of being told to keep stepping.  The first swing side on each
    walk-out is picked from the lateral command sign.
    """

    cfg: GaitPhaseCommandCfg

    def __init__(self, cfg: GaitPhaseCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        B = self.num_envs
        self._phase = torch.zeros(B, device=self.device)
        self._was_standing = torch.ones(B, dtype=torch.bool, device=self.device)
        self._command = torch.zeros(B, 2, device=self.device)
        self._dt = env.step_dt

    # -- properties ---------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        """(B, 2) = (sin phi, cos phi)."""
        return self._command

    @property
    def phase(self) -> torch.Tensor:
        """(B,) gait phase in RADIANS, wrapped to [0, 2*pi).

        This is what ``cd_mpc.contact_schedule.make_walking_schedule`` expects
        for its ``gait_phase`` argument, and what a phase-locked contact reward
        should read.
        """
        return self._phase

    @property
    def period(self) -> float:
        return self.cfg.period

    @property
    def moving(self) -> torch.Tensor:
        """(B,) bool: the clock advanced this tick (source above threshold /
        not holding). Valid after this term's update in the same tick."""
        return ~self._was_standing

    # -- lifecycle ----------------------------------------------------------

    def compute(self, dt: float) -> None:
        # Stash dt: the base class does not forward it to _update_command, and
        # the reset path calls compute(dt=0.0), which must not advance.
        self._dt = dt
        super().compute(dt)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Episode reset (the resampling timer never fires)."""
        self._phase[env_ids] = 0.0
        self._was_standing[env_ids] = True

    def _update_command(self) -> None:
        if self.cfg.command_name == "foothold":
            # Foothold line: the clock runs unless the foothold command holds.
            # The source term is declared AFTER this clock, so the flag is one
            # step stale — hold transitions only happen at touchdown
            # boundaries, where the stale read is exact.
            src = self._env.command_manager.get_term(self.cfg.command_name)
            moving = ~src.holding
            just_started = moving & self._was_standing
            if just_started.any():
                # Walk-out restarts at the boundary the SOURCE picked ({0, pi})
                # so the first swing side honors its committed queue head.
                self._phase[just_started] = src.walk_out_phase[just_started]
            self._was_standing = ~moving
        else:
            cmd = self._env.command_manager.get_command(self.cfg.command_name)
            cmd_mag = cmd[:, :2].norm(dim=1) + cmd[:, 2].abs()
            moving = cmd_mag > self.cfg.command_threshold

            # Walk-out: start the cycle on the side we are stepping towards.
            just_started = moving & self._was_standing
            if just_started.any():
                vy = cmd[just_started, 1]
                self._phase[just_started] = torch.where(
                    vy < 0.0,
                    torch.full_like(vy, math.pi),
                    torch.zeros_like(vy),
                )
            self._was_standing = ~moving

        dphi = TWO_PI * self._dt / self.cfg.period
        # Wrap before decaying: the phase must stay in [0, 2*pi) or the
        # standing decay multiplies an unbounded accumulated angle and jumps
        # the clock instead of easing it back to the boundary.
        self._phase = torch.where(
            moving,
            (self._phase + dphi) % TWO_PI,
            self._phase * self.cfg.stance_decay,
        )

        self._command[:, 0] = self._phase.sin()
        self._command[:, 1] = self._phase.cos()

    def _update_metrics(self) -> None:
        pass


@dataclass(kw_only=True)
class GaitPhaseCommandCfg(CommandTermCfg):
    """Config for :class:`GaitPhaseCommand`."""

    # Clock term: the base resampling timer must never fire mid-episode; the
    # only per-episode work happens in reset -> _resample_command.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    period: float = 0.8
    """Full gait cycle T_cycle in seconds (two touchdowns)."""

    command_name: Literal["twist", "foothold"] = "twist"
    """Twist command term this clock is gated on.  Declare THIS term after it."""

    command_threshold: float = 0.1
    """|v_xy| + |wz| below this counts as a standing command."""

    stance_decay: float = 0.95
    """Per-step geometric decay of the phase towards 0 while standing."""

    def build(self, env: ManagerBasedRlEnv) -> GaitPhaseCommand:
        return GaitPhaseCommand(self, env)
