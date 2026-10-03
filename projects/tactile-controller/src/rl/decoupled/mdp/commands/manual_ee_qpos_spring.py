"""PLAY-ONLY: `EEQposeSpringCommand` with the operator owning the tip goal
(viser gizmo, no dataset draw), the contact mask, the normal force and a
`virtual force` switch — off = the mask and force command stay live for the
observations but no spring force is applied (the plant's phantom mode).
Command, accessors and debug vis are the trained term's, so an arm -VForce
checkpoint plays here strict=True.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

from mjlab.utils.lab_api.math import quat_apply, quat_mul, yaw_quat

from .ee_qpos_spring import EEQposeSpringCommand, EEQposeSpringCommandCfg
from .manual_ee_target import _q_apply, _q_conj, _q_mul

if TYPE_CHECKING:
    import viser

    from mjlab.envs import ManagerBasedRlEnv

_DRAG_TIMEOUT = 5.0


class ManualEEQposeSpringCommand(EEQposeSpringCommand):
    """Gizmo-placed tip goal + operator contact mask, normal force [N] and the
    virtual-force switch. A fresh episode's goal is the tip pose the hand
    starts at; `ref_joint_pos` is the start posture.

    Attributes:
        virtual_force: 1 while the spring is applied on a live mask, 0 = command
            only (phantom). Shape (B,).
    """

    cfg: ManualEEQposeSpringCommandCfg

    def __init__(self, cfg: ManualEEQposeSpringCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self.virtual_force = torch.ones(self.num_envs, device=self.device)
        # written by the viser thread, consumed on the sim thread: (pos, quat) in the yaw-aligned frame
        self._pending: tuple[np.ndarray, np.ndarray] | None = None
        self._root = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))  # root pos, YAW quat (world)
        self._gizmo: viser.TransformControlsHandle | None = None
        self._get_env_idx: Callable[[], int] | None = None
        self._dragging = False
        self._gui_contact: viser.GuiCheckboxHandle | None = None
        self._gui_force: viser.GuiSliderHandle | None = None
        self._gui_virtual: viser.GuiCheckboxHandle | None = None

    # -- operator surface (headless-safe) ---------------------------------------

    def set_goal_yaw(self, env_idx: int, pos: torch.Tensor, quat: torch.Tensor) -> None:
        """Jump one env's goal to (`pos` (3,), `quat` wxyz (4,)), yaw-aligned pelvis frame."""
        self.pose_yaw[env_idx, :3] = pos
        self.pose_yaw[env_idx, 3:] = quat

    def set_contact(self, env_idx: int, on: bool, force_n: float, virtual: bool = True) -> None:
        """Contact mask + normal force command [N] along the goal's +x for one
        env; `virtual` False keeps the command but applies no spring force.
        Turning the mask on restarts the force ramp-in."""
        live = bool(on) and float(force_n) > 0.0
        if live and float(self.contact_mask[env_idx]) < 0.5:
            self._contact_age[env_idx] = 0.0
        self.contact_mask[env_idx] = 1.0 if live else 0.0
        self.force_mag[env_idx] = float(force_n) if live else 0.0
        self.virtual_force[env_idx] = 1.0 if virtual else 0.0
        self.phantom[env_idx] = 0.0 if virtual else 1.0
        self._tan_u[env_idx] = 0.0

    # -- lifecycle -----------------------------------------------------------------

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)  # window bookkeeping; the draw below is overwritten
        root_pos = self.robot.data.root_link_pos_w[env_ids]
        yaw_inv = yaw_quat(self.robot.data.root_link_quat_w[env_ids])
        yaw_inv = torch.cat([yaw_inv[:, :1], -yaw_inv[:, 1:]], dim=-1)
        tip_p = self.robot.data.body_link_pos_w[env_ids, self.body_idx]
        tip_q = self.robot.data.body_link_quat_w[env_ids, self.body_idx]
        self.pose_yaw[env_ids, :3] = quat_apply(yaw_inv, tip_p - root_pos)
        q = quat_mul(yaw_inv, tip_q)
        self.pose_yaw[env_ids, 3:] = torch.where(q[:, :1] < 0.0, -q, q)
        self.ref_joint_pos[env_ids] = self.robot.data.joint_pos[env_ids][:, self.data.joint_ids]

    def _update_command(self) -> None:
        if self._pending is not None:
            pos, quat = self._pending
            self._pending = None
            self.set_goal_yaw(
                self._viewed_env(),
                torch.as_tensor(pos, device=self.device, dtype=torch.float32),
                torch.as_tensor(quat, device=self.device, dtype=torch.float32),
            )
        if self._gui_contact is not None and self._gui_force is not None:
            self.set_contact(
                self._viewed_env(), bool(self._gui_contact.value), float(self._gui_force.value),
                virtual=True if self._gui_virtual is None else bool(self._gui_virtual.value),
            )
        super()._update_command()

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self._sync_viewer()

    # -- viser: the gizmo plus contact / force / virtual-force controls -----------

    def create_gui(
        self,
        name: str,
        server: viser.ViserServer,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        del on_change, request_action
        self._get_env_idx = get_env_idx
        # under /fixed_bodies the gizmo's pose IS a sim-world pose (scene offset)
        self._gizmo = server.scene.add_transform_controls(
            f"/fixed_bodies/ee_cmd/{name}", scale=0.15, depth_test=False, disable_sliders=True)
        self._gizmo.on_update(self._on_gizmo)
        with server.gui.add_folder(f"{name} (force)"):
            self._gui_contact = server.gui.add_checkbox("contact", initial_value=False)
            self._gui_force = server.gui.add_slider(
                "force [N]", min=0.0, max=self.cfg.force_slider_max, step=0.5,
                initial_value=min(10.0, self.cfg.force_slider_max))
            self._gui_virtual = server.gui.add_checkbox("virtual force", initial_value=True)

    def _viewed_env(self) -> int:
        b = 0 if self._get_env_idx is None else self._get_env_idx()
        return max(0, min(int(b), self.num_envs - 1))

    def _on_gizmo(self, event) -> None:
        """Gizmo world pose -> yaw-aligned-frame goal, using the cached root."""
        self._dragging = event.phase != "end"
        root_pos, yaw_q = self._root
        p_w = np.asarray(event.target.position, dtype=np.float64)
        q_w = np.asarray(event.target.wxyz, dtype=np.float64)
        pos = _q_apply(_q_conj(yaw_q), p_w - root_pos)
        quat = _q_mul(_q_conj(yaw_q), q_w)
        quat /= max(np.linalg.norm(quat), 1e-9)
        self._pending = (pos, quat)

    def _sync_viewer(self) -> None:
        if self._gizmo is None:
            return
        b = self._viewed_env()
        self._root = (
            self.robot.data.root_link_pos_w[b].cpu().numpy().astype(np.float64),
            yaw_quat(self.robot.data.root_link_quat_w)[b].cpu().numpy().astype(np.float64),
        )
        if self._dragging:
            if time.time() - self._gizmo.update_timestamp < _DRAG_TIMEOUT:
                return
            self._dragging = False
        pos_w, quat_w = self.target_pose_w()
        self._gizmo.position = tuple(float(v) for v in pos_w[b].cpu().numpy())
        self._gizmo.wxyz = tuple(float(v) for v in quat_w[b].cpu().numpy())


@dataclass(kw_only=True)
class ManualEEQposeSpringCommandCfg(EEQposeSpringCommandCfg):
    """PLAY-ONLY: the spring term's plant and accessors, operator-owned goal,
    mask, force and virtual-force switch (no draws, no windows)."""

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    contact_prob: float = 0.0
    phantom_prob: float = 0.0
    force_slider_max: float = 40.0
    """Upper end of the viser force slider [N]."""

    def build(self, env: ManagerBasedRlEnv) -> ManualEEQposeSpringCommand:
        return ManualEEQposeSpringCommand(self, env)
