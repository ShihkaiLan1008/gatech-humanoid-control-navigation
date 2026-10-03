"""H-LIP-fed centroidal MPC with the hand-force decision slots.

:class:`LocoHlipManipMPCCommand` composes the two existing layers by
inheritance: the schedule and references of :class:`LocoHlipMPCCommand`
(committed H-LIP footholds, orbit CoM ``x_ref``) and the hand layer of
:class:`LocoManipMPCCommand` (18-D ``cd`` QP, command-gated hand slots,
cone-AABB bounds, ``u_ref`` hand rows = -f_cmd).  Requires ``mpc_impl="cd"``.
Declare it AFTER ``foot_plan`` and both EE terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .locomotion_hlip_mpc import LocoHlipMPCCommand, LocoHlipMPCCommandCfg
from .locomotion_manip_mpc import LocoManipMPCCommand, LocoManipMPCCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class LocoHlipManipMPCCommand(LocoManipMPCCommand, LocoHlipMPCCommand):
    """H-LIP schedule + orbit ``x_ref`` (``LocoHlipMPCCommand``) under the
    loco-manip QP and hand gating of ``LocoManipMPCCommand``."""

    cfg: LocoHlipManipMPCCommandCfg


@dataclass(kw_only=True)
class LocoHlipManipMPCCommandCfg(LocoManipMPCCommandCfg, LocoHlipMPCCommandCfg):
    """Config for :class:`LocoHlipManipMPCCommand`: the union of both parents' fields."""

    mpc_impl: str = "cd"

    def build(self, env: ManagerBasedRlEnv) -> LocoHlipManipMPCCommand:
        return LocoHlipManipMPCCommand(self, env)
