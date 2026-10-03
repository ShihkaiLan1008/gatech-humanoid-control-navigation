"""Stage-2 observation functions: the frozen stage-1 slots go LIVE.

Layout contract (the reason stage-1 checkpoints RESUME): these functions fill
the exact dims/positions of hand_pose's placeholders —
``hand_stage2_zeros`` (8) → real per-hand [mask(1), force_cmd_b(3)/40] and
``hand_ext_force`` (6) → sensed per-hand contact force. Same shapes, same
term order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from .commands import HandForceEePoseCommand

_FORCE_SCALE = 40.0  # command force range upper bound — O(1) conditioning


def hand_contact_force_command(
    env: ManagerBasedRlEnv, command_names: tuple[str, str]
) -> torch.Tensor:
    """(B, 8): per hand [contact_mask(1), force_cmd_b(3)/40], L then R.

    The force command is the hand-on-surface force in each term's sample
    frame — the same frame as its pose command obs. Fills the
    ``hand_stage2_zeros`` slot of the stage-1 layout."""
    out = []
    for name in command_names:
        term: HandForceEePoseCommand = env.command_manager.get_term(name)
        out.append(term.contact_mask.unsqueeze(-1))
        out.append(term.force_cmd_b / _FORCE_SCALE)
    return torch.cat(out, dim=-1)


def hand_ext_force_meas(
    env: ManagerBasedRlEnv, command_names: tuple[str, str]
) -> torch.Tensor:
    """(B, 6): per hand SENSED external force ON THE WRIST /40, L then R.

    The cube sensors report the force BY the wrist ON the cube; the external
    force on the wrist is the reaction (negated) — same semantics as the
    stage-1 ``hand_ext_force`` slot ("external force on the wrist"), now
    measured instead of frozen zero. Critic-only."""
    out = []
    for name in command_names:
        term: HandForceEePoseCommand = env.command_manager.get_term(name)
        out.append(-term.measured_force_w() / _FORCE_SCALE)
    return torch.cat(out, dim=-1)
