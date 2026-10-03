"""PLAY-ONLY: operator-placed tactile-tip goal PLUS operator-driven contact
mask / normal force, realized by the virtual spring.

:class:`ManualEeTargetCommand` (gizmo goal, no pool) extended with
:class:`EeSpringContactCommand` (the vforce plant): a checkbox and a force
slider replace the sampled contact windows; the spring riding the target,
the force ramp-in and the mask/force accessors the actor obs and the rewards
read are the trained term's, so a VForce checkpoint plays here strict=True.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .ee_spring import (
    _CMD_FORCE_VIS,
    _MEAS_FORCE_VIS,
    EeSpringContactCommand,
    EeSpringContactCommandCfg,
)
from .manual_ee_target import ManualEeTargetCommand, ManualEeTargetCommandCfg

if TYPE_CHECKING:
    import viser

    from mjlab.envs import ManagerBasedRlEnv


class ManualEeSpringContactCommand(ManualEeTargetCommand, EeSpringContactCommand):
    """Gizmo-placed tip goal + operator contact mask and NORMAL force [N]."""

    cfg: "ManualEeSpringContactCommandCfg"

    def __init__(self, cfg: "ManualEeSpringContactCommandCfg", env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self._gui_contact: "viser.GuiCheckboxHandle | None" = None
        self._gui_force: "viser.GuiSliderHandle | None" = None

    # ── operator surface (headless-safe) ────────────────────────────────────

    def set_contact(self, env_idx: int, on: bool, force_n: float) -> None:
        """Contact mask + normal force command [N] (along the goal's +x) for
        one env.  Turning the mask on restarts the force ramp-in."""
        live = bool(on) and float(force_n) > 0.0
        if live and float(self.contact_mask[env_idx]) < 0.5:
            self._contact_age[env_idx] = 0.0
        self.contact_mask[env_idx] = 1.0 if live else 0.0
        self.force_mag[env_idx] = float(force_n) if live else 0.0
        self.phantom[env_idx] = 0.0
        self._tan_u[env_idx] = 0.0

    def _update_command(self) -> None:
        if self._gui_contact is not None and self._gui_force is not None:
            self.set_contact(
                self._viewed_env(), bool(self._gui_contact.value), float(self._gui_force.value)
            )
        super()._update_command()

    # ── viser GUI: the gizmo plus a contact checkbox and a force slider ─────

    def create_gui(
        self,
        name: str,
        server: "viser.ViserServer",
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        super().create_gui(name, server, get_env_idx, on_change, request_action)
        with server.gui.add_folder(f"{name} (force)"):
            self._gui_contact = server.gui.add_checkbox("contact", initial_value=False)
            self._gui_force = server.gui.add_slider(
                "force [N]", min=0.0, max=self.cfg.force_slider_max, step=0.5,
                initial_value=min(10.0, self.cfg.force_slider_max),
            )

    def _debug_vis_impl(self, visualizer) -> None:
        """The reference frame plus, while the mask is on, the commanded force
        (at the target) and the spring force (at the tip)."""
        super()._debug_vis_impl(visualizer)
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        masked = (self.contact_mask > 0.5).cpu().numpy()
        tgt = self.target_pose_w()[0].cpu().numpy()
        fcmd = self.force_cmd_w.cpu().numpy()
        fmeas = self.measured_force_w().cpu().numpy()
        tip = self.measured_ee_pose_w()[0].cpu().numpy()
        tag = self.cfg.body_name
        scale = 0.01  # m per N
        for b in env_indices:
            if not masked[b]:
                continue
            for kind, start, f, color in (
                ("cmd", tgt[b], fcmd[b], _CMD_FORCE_VIS),
                ("spring", tip[b], fmeas[b], _MEAS_FORCE_VIS),
            ):
                if float(np.linalg.norm(f)) > 1.0:
                    visualizer.add_arrow(
                        start=start, end=start + scale * f, color=color, width=0.010,
                        label=f"spring_force_{kind}_{tag}_{b}",
                    )


@dataclass(kw_only=True)
class ManualEeSpringContactCommandCfg(ManualEeTargetCommandCfg, EeSpringContactCommandCfg):
    """PLAY-ONLY: the spring term's plant and accessors, operator-owned goal,
    mask and force (no sampled windows)."""

    contact_prob: float = 0.0
    phantom_prob: float = 0.0
    force_slider_max: float = 40.0
    """Upper end of the viser force slider [N]."""

    def build(self, env: ManagerBasedRlEnv) -> ManualEeSpringContactCommand:
        return ManualEeSpringContactCommand(self, env)
