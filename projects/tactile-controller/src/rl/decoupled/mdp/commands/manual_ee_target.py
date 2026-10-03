"""PLAY-ONLY: operator-placed tactile-tip targets for the foot_mimic line.

Same [B, 7] robot-frame command, same reference views and debug frame as
:class:`EeTargetCommand`, so any checkpoint plays strict=True.  The pool draw
is replaced by a viser transform gizmo per hand: drag / rotate it and the
goal jumps there (no glide).  A fresh episode starts at the nominal tip pose.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from mjlab.utils.lab_api.math import matrix_from_quat

from .ee_tartget import EeTargetCommand, EeTargetCommandCfg

if TYPE_CHECKING:
    import viser

    from mjlab.envs import ManagerBasedRlEnv

_DRAG_TIMEOUT = 5.0
"""Releases a gizmo whose drag-end event was lost (client gone mid-drag)."""


def _q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _q_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _q_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _q_mul(_q_mul(q, np.array([0.0, *v])), _q_conj(q))[1:]


class ManualEeTargetCommand(EeTargetCommand):
    """Tactile-tip target assigned by the operator instead of the reach pool."""

    cfg: "ManualEeTargetCommandCfg"

    def __init__(self, cfg: "ManualEeTargetCommandCfg", env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        # Written by the viser thread, consumed on the sim thread: (pos_b, quat_b).
        self._pending: tuple[np.ndarray, np.ndarray] | None = None
        self._root = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        self._gizmo: "viser.TransformControlsHandle | None" = None
        self._get_env_idx: Callable[[], int] | None = None
        self._dragging = False

    # ── the operator replaces the pool ──────────────────────────────────────

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Fresh episodes start at the nominal tip; later resamples are ignored."""
        fresh = env_ids[self._fresh[env_ids]]
        if len(fresh) == 0:
            return
        n = len(fresh)
        pos = self._nominal_pos_b.expand(n, 3)
        quat = self._nominal_quat_b.expand(n, 4)
        self._set_goal(fresh, pos, quat)
        q0 = self._arm_q_default[fresh]
        self._anchor_start_q[fresh] = q0
        self._anchor_goal_q[fresh] = q0
        self.anchor_q[fresh] = q0
        self._fresh[fresh] = False

    def _set_goal(self, env_ids: torch.Tensor, pos_b: torch.Tensor, quat_b: torch.Tensor) -> None:
        """Jump the reference and its goal to ``pos_b``/``quat_b`` (robot frame)."""
        self._start_pos[env_ids] = pos_b
        self._goal_pos[env_ids] = pos_b
        self._start_quat[env_ids] = quat_b
        self._goal_quat[env_ids] = quat_b
        self.pose_command_b[env_ids, :3] = pos_b
        self.pose_command_b[env_ids, 3:] = quat_b
        self._T[env_ids] = 1.0
        self._t[env_ids] = 1.0

    def _update_command(self) -> None:
        if self._pending is not None:
            pos_b, quat_b = self._pending
            self._pending = None
            ids = torch.tensor([self._viewed_env()], device=self.device)
            self._set_goal(
                ids,
                torch.as_tensor(pos_b, device=self.device, dtype=torch.float32).unsqueeze(0),
                torch.as_tensor(quat_b, device=self.device, dtype=torch.float32).unsqueeze(0),
            )
        super()._update_command()

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self._sync_viewer()

    # ── viser gizmo ─────────────────────────────────────────────────────────

    def create_gui(
        self,
        name: str,
        server: "viser.ViserServer",
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        del on_change, request_action
        self._get_env_idx = get_env_idx
        # Under /fixed_bodies the gizmo's pose IS a sim-world pose (scene offset).
        self._gizmo = server.scene.add_transform_controls(
            f"/fixed_bodies/ee_cmd/{name}", scale=0.15, depth_test=False, disable_sliders=True
        )
        self._gizmo.on_update(self._on_gizmo)

    def _viewed_env(self) -> int:
        b = 0 if self._get_env_idx is None else self._get_env_idx()
        return max(0, min(int(b), self.num_envs - 1))

    def _on_gizmo(self, event) -> None:
        """Gizmo world pose -> robot-frame goal, using the cached anchor."""
        self._dragging = event.phase != "end"
        root_pos, root_q = self._root
        p_w = np.asarray(event.target.position, dtype=np.float64)
        q_w = np.asarray(event.target.wxyz, dtype=np.float64)
        pos_b = _q_apply(_q_conj(root_q), p_w - root_pos)
        quat_b = _q_mul(_q_conj(root_q), q_w)
        quat_b /= max(np.linalg.norm(quat_b), 1e-9)
        self._pending = (pos_b, quat_b)

    def _sync_viewer(self) -> None:
        if self._gizmo is None:
            return
        b = self._viewed_env()
        self._root = (
            self.robot.data.root_link_pos_w[b].cpu().numpy().astype(np.float64),
            self.anchor_quat()[b].cpu().numpy().astype(np.float64),
        )
        if self._dragging:
            if time.time() - self._gizmo.update_timestamp < _DRAG_TIMEOUT:
                return
            self._dragging = False
        pos_w, quat_w = self.target_pose_w()
        self._gizmo.position = tuple(float(v) for v in pos_w[b].cpu().numpy())
        self._gizmo.wxyz = tuple(float(v) for v in quat_w[b].cpu().numpy())

    def _debug_vis_impl(self, visualizer) -> None:
        """Only the reference frame (goal ≡ reference here)."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        pos_w, quat_w = self.target_pose_w()
        pos_np = pos_w.cpu().numpy()
        rot_np = matrix_from_quat(quat_w).cpu().numpy()
        for b in env_indices:
            visualizer.add_frame(
                position=pos_np[b], rotation_matrix=rot_np[b], scale=0.15,
                axis_radius=0.008, label=f"ee_ref_{self.cfg.body_name}_{b}",
            )


@dataclass(kw_only=True)
class ManualEeTargetCommandCfg(EeTargetCommandCfg):
    """PLAY-ONLY: the pool term's geometry and views, operator-assigned goals."""

    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: ManagerBasedRlEnv) -> ManualEeTargetCommand:
        return ManualEeTargetCommand(self, env)
