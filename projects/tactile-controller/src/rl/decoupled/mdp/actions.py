"""Action terms shared across the decoupled tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["FrozenJointPositionAction", "FrozenJointPositionActionCfg"]


class FrozenJointPositionAction(JointPositionAction):
    """A joint-position PD hold with ZERO action dimension.

    The term owns its actuators exactly like `JointPositionAction` — position
    target written every step, encoder-bias DR included — but contributes
    nothing to the policy's action space: the policy never sees or controls
    these joints.

    This is the lower-body-only training stand-in for the upper policy: the
    arms ride a PD hold at the nominal pose as a real ACTION TERM (the
    ActionManager seam the joint tasks' `upper_body_joint_pos` fills), instead
    of the old hack of a command term writing actuator targets at reset.

    GRAVITY COMPENSATION (``gravity_comp``, default on): a plain PD hold does
    NOT sit at its target — it settles at the gravity equilibrium, measured
    0.17 rad of arm sag at the G1's arm gains, i.e. the hands droop well below
    the nominal pose. An IK action would not fix this (it would emit the same
    joint targets — the nominal pose IS FK of the defaults — and stretch the
    same PD). Instead the hold target is biased by the model's own bias force:

        q_target = q_default + qfrc_bias(q) / kp

    whose PD equilibrium is exactly q_default (kp·(q_target − q0) = τ_gravity).
    kp is the per-env actuator gain read from the live warp model, so the
    `pd_gains` DR is respected; qfrc_bias is gravity + Coriolis at the current
    state, so the hold is the quasi-static inverse dynamics of the arm. The
    correction is clamped (``max_comp``) so impact/fall transients cannot fling
    the target.
    """

    def __init__(self, cfg: FrozenJointPositionActionCfg, env: ManagerBasedRlEnv):
        # the base __init__ sizes its buffers via the action_dim property —
        # report the true dim until it finishes, 0 to the manager afterwards
        self._frozen = False
        super().__init__(cfg=cfg, env=env)
        self._env = env
        self._kp: torch.Tensor | None = None    # lazy: DR runs after build
        self._processed_actions = (
            torch.zeros_like(self._raw_actions) * self._scale + self._offset
        )
        self._frozen = True

    @property
    def action_dim(self) -> int:
        return 0 if self._frozen else self._action_dim

    def _gain(self) -> torch.Tensor:
        """Per-env kp for the held joints, [nworld or 1, n]. Read lazily from
        the warp model so the startup `pd_gains` DR is already applied."""
        if self._kp is None:
            local, _ = self._entity.find_actuators(
                list(self._target_names), preserve_order=True)
            ctrl_ids = self._entity.indexing.ctrl_ids[
                torch.tensor(local, dtype=torch.long,
                             device=self._entity.indexing.ctrl_ids.device)]
            gainprm = self._env.sim.model.actuator_gainprm  # torch view
            if gainprm.ndim == 2:                # not DR-expanded: [nu, prm]
                gainprm = gainprm[None]
            self._kp = gainprm[:, ctrl_ids.to(gainprm.device), 0].clamp(min=1.0)
        return self._kp

    def process_actions(self, actions: torch.Tensor) -> None:
        # receives the manager's empty [B, 0] slice; the hold target is the
        # (possibly per-env) offset, recomputed in case offset was re-tensored
        target = torch.zeros_like(self._raw_actions) * self._scale + self._offset
        if self.cfg.gravity_comp:
            bias = self._entity.data._joint_dof_field("qfrc_bias")
            comp = bias[:, self._target_ids] / self._gain()
            target = target + comp.clamp(-self.cfg.max_comp, self.cfg.max_comp)
        self._processed_actions = target


@dataclass(kw_only=True)
class FrozenJointPositionActionCfg(JointPositionActionCfg):
    """Cfg for the zero-dim nominal PD hold."""

    gravity_comp: bool = True
    """Bias the hold target by qfrc_bias/kp so the PD equilibrium is the
    nominal pose instead of the gravity-sagged one."""

    max_comp: float = 0.3
    """Clamp [rad] on the per-joint gravity-comp target bias."""

    def build(self, env: ManagerBasedRlEnv) -> FrozenJointPositionAction:
        return FrozenJointPositionAction(self, env)
