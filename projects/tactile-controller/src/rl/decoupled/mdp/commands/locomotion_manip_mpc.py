"""Loco-manipulation MPC command on the VELOCITY line (joint_vel_force_mpc).

:class:`LocoManipMPCCommand` extends the proven :class:`LocoMPCCommand`
(uniform-twist x_ref, measured-velocity c_bar, gait-clock walking schedule)
with the 6 hand-force control slots of the merged ``cd`` QP — the schedule,
references and foot machinery are inherited UNCHANGED; only the hand layer is
added, carried verbatim from the measured hand_force recipe:

* QP hand gate = commanded contact mask ∧ debounced measured surface contact
  (``contact_debounce_steps`` at the policy rate). Phantom windows never gate
  on (no surface → no measured contact) — the QP plans no phantom bracing.
* ``u_ref`` hand slots = −f_cmd_w — u enters the momentum equation exactly
  like the feet GRF (force ON THE ROBOT), so hand slots carry the
  surface→robot reaction. The lab original (mpc-rl) passes ZERO here (hand
  force = free slack, never a tracked command); the +118 N phantom-thrust
  measurement belongs to the box-SIGN fix in the vendored loco_manip_mpc.py,
  not to any u_ref convention.
* ``R_f_hand = 1e-1`` — at 1e-2 the QP re-optimizes the hand as a balance
  actuator and force tracking dies (measured on the source stack).
* Per-env world-frame cone-AABB force bounds, COMMAND-SCALED
  (``f_hand_headroom``·f_cmd, capped): a command-independent box hands the QP
  a phantom tangential-without-pressing channel (measured 13 N vertical
  crutch at zero command).
* Hand contact points = the tactile APEX sites (dome apex = command frame).

Requires ``mpc_impl="cd"`` — the vendored centroidal_mpc has no hand slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mjlab.managers.scene_entity_config import SceneEntityCfg

from rl.decoupled.cd_mpc import cd as cd_mpc

from .locomotion_mpc import LocoMPCCommand, LocoMPCCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactSensor

    from .ee_contact import EeContactTargetCommand


def _cone_aabb(press_dir_w: Tensor, mu: float, cap: Tensor) -> tuple[Tensor, Tensor]:
    """World-axis AABB of the reaction-force cone, truncated at ``cap``.

    ``cap`` is PER-ENV and COMMAND-SCALED by the caller (f_cmd·headroom, not a
    global max). Returns (lb, ub), each (B, 3)."""
    p = press_dir_w
    s = (1.0 - p.pow(2)).clamp(min=0.0).sqrt()
    c = cap.unsqueeze(-1)
    c1 = -c * (p - mu * s)
    c2 = -c * (p + mu * s)
    lb = torch.minimum(c1, c2).clamp(max=0.0)
    ub = torch.maximum(c1, c2).clamp(min=0.0)
    return lb, ub


class LocoManipMPCCommand(LocoMPCCommand):
    """Velocity-line centroidal MPC with hand-force decision slots."""

    cfg: "LocoManipMPCCommandCfg"

    def __init__(self, cfg: "LocoManipMPCCommandCfg", env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)
        B = env.num_envs
        device = env.device

        hand_site_cfg = SceneEntityCfg("robot", site_names=cfg.hand_site_names)
        hand_site_cfg.resolve(env.scene)
        self._hand_site_ids: list[int] = hand_site_cfg.site_ids
        _, _hand_names = self._robot.find_sites(
            cfg.hand_site_names, preserve_order=False
        )
        self._lh_local_idx = next(
            i for i, n in enumerate(_hand_names) if "left" in n.lower()
        )
        self._rh_local_idx = next(
            i for i, n in enumerate(_hand_names) if "right" in n.lower()
        )

        def _sensor_list(names) -> list["ContactSensor"]:
            if cfg.gate_mode != "sensor":
                return []
            if isinstance(names, str):
                names = (names,)
            return [env.scene[n] for n in (names or ())]

        self._lhand_sensors = _sensor_list(cfg.lhand_sensor_name)
        self._rhand_sensors = _sensor_list(cfg.rhand_sensor_name)

        self._last_contact_step = torch.full(
            (B, 2), -(10**6), dtype=torch.long, device=device
        )
        self._gate = torch.zeros(B, 2, device=device)
        self._hand_cmd_w = torch.zeros(B, 6, device=device)
        self._hand_u_star = torch.zeros(B, 6, device=device)

    def _build_mpc(self, cfg: "LocoManipMPCCommandCfg", env: "ManagerBasedRlEnv") -> None:
        if cfg.mpc_impl != "cd":
            raise ValueError(
                "LocoManipMPCCommand requires mpc_impl='cd' — the vendored "
                "centroidal_mpc has no hand slots."
            )
        self._mpc = cd_mpc.CentroidalMPC(
            cd_mpc.MPCConfig(
                N=cfg.mpc_horizon,
                dt=cfg.mpc_dt,
                mass=cfg.mass,
                locomanipulation=True,
                mu_hand=cfg.mu_hand,
                f_hand_max=cfg.f_hand_max,
                R_f_hand=cfg.R_f_hand,
            ),
            device=env.device,
        )
        self._mpc_input_cls = cd_mpc.MPCInput
        self._extra_mpc_kwargs = {}  # hand kwargs are built per solve

    # ── hand layer ──────────────────────────────────────────────────────────

    def _ee_terms(self) -> tuple["EeContactTargetCommand", "EeContactTargetCommand"]:
        mgr = self._env.command_manager
        return (
            mgr.get_term(self.cfg.ee_command_names[0]),
            mgr.get_term(self.cfg.ee_command_names[1]),
        )

    def _per_step_update(self) -> None:
        """Debounce bookkeeping must run at the POLICY rate, not the solve rate."""
        for h, sensors in ((0, self._lhand_sensors), (1, self._rhand_sensors)):
            raw = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for sensor in sensors:
                found = sensor.data.found
                if found is not None:
                    raw |= (found > 0).any(dim=-1)
            self._last_contact_step[:, h] = torch.where(
                raw,
                torch.full_like(self._last_contact_step[:, h], self._step_count),
                self._last_contact_step[:, h],
            )

    def _debounced_contact(self) -> Tensor:
        """(B, 2) float: measured surface contact with dropout debounce."""
        return (
            (self._step_count - self._last_contact_step)
            <= self.cfg.contact_debounce_steps
        ).float()

    def _build_mpc_input(self, *, x0: Tensor, schedule, x_ref: Tensor, c_bar: Tensor):
        cfg = self.cfg
        B = self.num_envs
        N = cfg.mpc_horizon
        device = self.device

        hand_pos = self._robot.data.site_pos_w[:, self._hand_site_ids, :]
        r_lh = hand_pos[:, self._lh_local_idx, :]
        r_rh = hand_pos[:, self._rh_local_idx, :]

        lh_term, rh_term = self._ee_terms()
        mask_cmd = torch.stack(
            [lh_term.contact_mask, rh_term.contact_mask], dim=-1
        )
        if cfg.gate_mode == "command":
            # No contact sensors (virtual-spring task): gate on the command's
            # own state — window live ∧ not phantom (whole-window semantics).
            live = torch.stack(
                [(t.phantom < 0.5).float() for t in (lh_term, rh_term)], dim=-1
            )
            hand_contact = mask_cmd * live
        else:
            hand_contact = mask_cmd * self._debounced_contact()
        self._gate = hand_contact.detach().clone()

        f_cmd_lh_w = lh_term.force_cmd_w  # force ON SURFACE
        f_cmd_rh_w = rh_term.force_cmd_w
        self._hand_cmd_w = torch.cat([f_cmd_lh_w, f_cmd_rh_w], dim=-1).detach()

        cap_lh = (cfg.f_hand_headroom * lh_term.force_cmd_mag).clamp(max=cfg.f_hand_max)
        cap_rh = (cfg.f_hand_headroom * rh_term.force_cmd_mag).clamp(max=cfg.f_hand_max)
        lb_lh, ub_lh = _cone_aabb(lh_term.press_dir_w, cfg.mu_hand, cap_lh)
        lb_rh, ub_rh = _cone_aabb(rh_term.press_dir_w, cfg.mu_hand, cap_rh)
        hand_force_lb = torch.cat([lb_lh, lb_rh], dim=-1)
        hand_force_ub = torch.cat([ub_lh, ub_rh], dim=-1)

        u_ref = torch.zeros(B, N, 18, device=device)
        u_ref[:, :, 12:15] = -f_cmd_lh_w.unsqueeze(1)
        u_ref[:, :, 15:18] = -f_cmd_rh_w.unsqueeze(1)

        return cd_mpc.MPCInput(
            x0=x0,
            schedule=schedule,
            x_ref=x_ref,
            u_ref=u_ref,
            u_prev=self._u_prev,
            other_body_force=torch.zeros(B, 3, device=device),
            other_body_torque=torch.zeros(B, 3, device=device),
            c_bar=c_bar,
            r_LH=r_lh,
            r_RH=r_rh,
            hand_contact=hand_contact,
            hand_force_lb=hand_force_lb,
            hand_force_ub=hand_force_ub,
        )

    def _post_solve(self, mpc_out) -> None:
        self._hand_u_star = mpc_out.u_star[:, 12:18].detach()

    def _resample_command(self, env_ids: Tensor) -> None:
        super()._resample_command(env_ids)
        self._last_contact_step[env_ids] = -(10**6)
        self._gate[env_ids] = 0.0
        self._hand_cmd_w[env_ids] = 0.0
        self._hand_u_star[env_ids] = 0.0


@dataclass(kw_only=True)
class LocoManipMPCCommandCfg(LocoMPCCommandCfg):
    """Configuration for :class:`LocoManipMPCCommand`."""

    mpc_impl: str = "cd"

    ee_command_names: tuple[str, str] = ("left_ee_cmd", "right_ee_cmd")
    """The two :class:`EeContactTargetCommand` terms carrying mask + force."""

    hand_site_names: tuple[str, str] = ("left_tactile_site", "right_tactile_site")
    """Hand contact points fed to the QP = the tactile APEX sites."""

    gate_mode: str = "sensor"
    """QP hand gate source: "sensor" = mask ∧ debounced measured contact (the
    cube task); "command" = mask ∧ not phantom (the virtual-spring task,
    which has no contact sensors)."""

    lhand_sensor_name: str | tuple[str, ...] = "lhand_cube_contact"
    rhand_sensor_name: str | tuple[str, ...] = "rhand_cube_contact"

    mu_hand: float = 0.6
    f_hand_max: float = 50.0
    f_hand_headroom: float = 1.25
    R_f_hand: float = 1e-1
    """At 1e-2 the QP re-optimizes the hand as a balance actuator (measured on
    the source stack — hand force tracking dies)."""

    contact_debounce_steps: int = 5
    """Measured-contact dropout tolerance in POLICY steps (0.1 s at 50 Hz)."""

    def build(self, env: "ManagerBasedRlEnv") -> LocoManipMPCCommand:
        return LocoManipMPCCommand(self, env)
