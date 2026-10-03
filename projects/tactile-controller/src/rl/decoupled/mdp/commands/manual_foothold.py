"""PLAY-ONLY: operator-assigned footholds for the foothold line.

A :class:`FootHoldCommand` whose sampler is replaced by a HUMAN: the step
machinery, the swing arc, the hold mode, the ``[B, 7]`` command layout and the
MPC / H-LIP timeline views are all the trained term's, so any checkpoint of a
task that declares ``foothold`` plays here strict=True; swap it into
``commands["foothold"]`` of that task's play cfg (foot_mpc, foot_hlip, ...).

Interaction (viser):

* two ground boxes = the TRAINED landing coverage box for each foot, drawn in
  its stance frame (the OTHER foot's measured pose), each with a draggable
  marker: drag in xy, twist about z.  A drag outside the box snaps back in.
* ``step [left | right]`` takes exactly one step with that foot, then stands.
* ``walk`` steps continuously, alternating feet, with the current placements.
* per-foot ``x / width / dyaw`` sliders and a shared ``apex`` slider.

The placement is stored RELATIVE to the stance foot (mirrored stance frame,
exactly the trained ``target_m``; ``dyaw`` = heading change of the step, left
turn positive for both feet), so it survives the robot moving.  While a step
is in flight its target is COMMITTED; a drag applies to the next step.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

from rl.g1_assets.g1_constants import FEET_Y_DIST

from .foot_hold import FootHoldCommand, FootHoldCommandCfg

if TYPE_CHECKING:
    import viser

    from mjlab.envs import ManagerBasedRlEnv

_FEET = ("left", "right")
_HOLD_FOREVER = 1.0e9
"""Stand duration [s] that only an operator request ends."""
_DRAG_TIMEOUT = 5.0
"""Safety net only: a drag ends on the gizmo's ``end`` phase; this releases a
marker whose release event was lost (client gone mid-drag)."""
_BOX_VIS = ((0.10, 0.80, 0.25, 0.20), (0.20, 0.45, 0.95, 0.20))
_TGT_VIS = ((0.10, 0.90, 0.20, 0.95), (0.25, 0.55, 1.00, 0.95))


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_to_wxyz(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _wxyz_to_yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class ManualFootHoldCommand(FootHoldCommand):
    """Footholds assigned by the operator instead of the sampler."""

    cfg: "ManualFootHoldCommandCfg"

    def __init__(self, cfg: "ManualFootHoldCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        sp = cfg.sampler
        # The drawn boxes ARE the trained clamps of the sampler.
        self.x_range = tuple(sp.x_clamp)
        self.w_range = (
            max(sp.w_min_base, FEET_Y_DIST + sp.dw_range[0] + sp.jitter_w[0]),
            FEET_Y_DIST + sp.dw_range[1] + sp.jitter_w[1],
        )
        self.yaw_range = tuple(sp.yaw_clamp)
        self.apex_range = tuple(sp.apex_range)

        # Operator state.  Plain numpy/python: viser callbacks run off the sim
        # thread and must never touch a CUDA tensor.
        # [foot][x, width, dyaw] in the mirrored stance frame.
        self._spec = np.array([[0.0, FEET_Y_DIST, 0.0]] * 2)
        self._apex_spec = 0.5 * (self.apex_range[0] + self.apex_range[1])
        self._walk = False
        self._steps_left = 0
        self._lead_foot: int | None = None
        self._anchor = np.zeros((2, 3))  # [foot][x, y, yaw] of its stance foot

        self._get_env_idx: Callable[[], int] | None = None
        self._gizmos: list["viser.TransformControlsHandle"] = []
        self._sliders: list[list[Any]] = []
        self._apex_slider: Any = None
        self._dragging = [False, False]
        self._syncing = False

        print(
            "[INFO] manual footholds: markers + step buttons live in the viser "
            "GUI (run play with --viewer viser)"
        )

    # ── the operator replaces the sampler ───────────────────────────────────

    def _operator_m(self, foot: torch.Tensor) -> torch.Tensor:
        """Mirrored-frame targets (n, 4) = [x, w, 0, s·dyaw] for ``foot`` (n,)."""
        spec = torch.as_tensor(self._spec, device=self.device, dtype=torch.float32)[foot]
        s = torch.where(foot == 0, 1.0, -1.0)
        return torch.stack(
            [spec[:, 0], spec[:, 1], torch.zeros_like(spec[:, 0]), s * spec[:, 2]], dim=-1)

    def _plan_next(self, ids: torch.Tensor, foot: torch.Tensor, may_hold: bool) -> None:
        """The follower is the operator's placement; a stand follows unless walking."""
        del may_hold
        self._next_foot[ids] = foot
        self._next_m[ids] = self._operator_m(foot)
        self._next_apex[ids] = float(self._apex_spec)
        self._next_hold[ids] = not self._walk
        self._next_hold_time[ids] = _HOLD_FOREVER

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Reset comes up standing until the operator asks for a step."""
        super()._resample_command(env_ids)
        self._hold_left[env_ids] = _HOLD_FOREVER

    def _update_command(self) -> None:
        super()._update_command()
        if not (self._walk or self._steps_left > 0):
            return
        ids = self.standing.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) == 0:
            return
        if self._lead_foot is None:
            foot = 1 - self.swing[ids]
        else:
            foot = torch.full_like(self.swing[ids], int(self._lead_foot))
        self._plan_next(ids, foot, may_hold=False)
        self._issue_step(ids)
        self._update_arc()
        if self._steps_left > 0:
            self._steps_left -= 1
        self._lead_foot = None

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self._sync_viewer()

    # ── operator surface (headless-safe: plain attributes) ──────────────────

    def set_spec(self, foot: int, x: float, width: float, dyaw: float) -> None:
        """Place one foot (0 = left), clamped into the trained box.  ``width``
        is the ABSOLUTE lateral distance from the stance foot [m]; ``dyaw`` is
        the heading change of the step [rad], left turn positive."""
        self._spec[foot] = (
            float(np.clip(x, *self.x_range)),
            float(np.clip(width, *self.w_range)),
            float(np.clip(dyaw, *self.yaw_range)),
        )

    def set_apex(self, apex: float) -> None:
        """Swing-arc peak height above the lift-off site [m], clamped."""
        self._apex_spec = float(np.clip(apex, *self.apex_range))

    def set_walk(self, walk: bool) -> None:
        """Continuous alternating stepping with the current placements;
        switching off stops at the next landing."""
        self._walk = bool(walk)
        if not self._walk:
            self._next_hold[:] = True
            self._next_hold_time[:] = _HOLD_FOREVER

    def request_step(self, foot: int | None = None, n: int = 1) -> None:
        """Take ``n`` steps; ``foot`` (0 = left, 1 = right) picks which foot
        swings on the next one (None = the foot that did not swing last)."""
        if foot is not None:
            self._lead_foot = int(foot)
        self._steps_left += int(n)

    # ── geometry ────────────────────────────────────────────────────────────

    def _stance_of(self, b: int, foot: int) -> np.ndarray:
        """(x, y, yaw) of the foot ``foot`` steps FROM (the other foot)."""
        other = 1 - foot
        p = self.robot_feet_pos_w[b, other, 0:2].cpu().numpy()
        return np.array([p[0], p[1], float(self.robot_feet_yaw[b, other])])

    def _target_of(self, b: int, foot: int) -> np.ndarray:
        """(x, y, yaw) where ``foot`` lands: the committed target while it is
        swinging, else what the current placement would command."""
        if not bool(self.standing[b]) and int(self.swing[b]) == foot:
            p = self._target_pos_w[b, 0:2].cpu().numpy()
            return np.array([p[0], p[1], float(self._target_yaw_w[b])])
        ax, ay, ayaw = self._anchor[foot]
        x, w, dyaw = self._spec[foot]
        sgn = 1.0 if foot == 0 else -1.0
        c, s = math.cos(ayaw), math.sin(ayaw)
        return np.array([ax + c * x - s * sgn * w,
                         ay + s * x + c * sgn * w,
                         _wrap_pi(ayaw + dyaw)])

    # ── viser GUI ───────────────────────────────────────────────────────────

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

        with server.gui.add_folder(f"{name} (manual)"):
            walk = server.gui.add_checkbox("walk", initial_value=self._walk)

            @walk.on_update
            def _(_ev) -> None:
                self.set_walk(bool(walk.value))

            steps = server.gui.add_button_group("step", options=["left", "right"])

            @steps.on_click
            def _(event) -> None:
                self.request_step(0 if event.target.value == "left" else 1)

            reset_btn = server.gui.add_button("place in place")

            @reset_btn.on_click
            def _(_ev) -> None:
                for foot in (0, 1):
                    self._set_spec_gui(foot, 0.0, FEET_Y_DIST, 0.0)

            self._apex_slider = server.gui.add_slider(
                "apex [m]", min=self.apex_range[0], max=self.apex_range[1],
                step=0.005, initial_value=float(self._apex_spec),
            )
            self._apex_slider.on_update(lambda _ev: self.set_apex(float(self._apex_slider.value)))

            for foot, side in enumerate(_FEET):
                with server.gui.add_folder(side):
                    handles = [
                        server.gui.add_slider(
                            "x [m]", min=self.x_range[0], max=self.x_range[1],
                            step=0.005, initial_value=float(self._spec[foot, 0]),
                        ),
                        server.gui.add_slider(
                            "width [m]", min=self.w_range[0], max=self.w_range[1],
                            step=0.005, initial_value=float(self._spec[foot, 1]),
                        ),
                        server.gui.add_slider(
                            "dyaw [deg]",
                            min=math.degrees(self.yaw_range[0]),
                            max=math.degrees(self.yaw_range[1]),
                            step=1.0,
                            initial_value=math.degrees(float(self._spec[foot, 2])),
                        ),
                    ]
                    self._sliders.append(handles)
                    for h in handles:
                        h.on_update(lambda _ev, f=foot: self._on_slider(f))

        # Markers under /fixed_bodies: that frame carries the tracking camera's
        # scene offset, so a marker's pose IS a sim-world pose.
        for foot, side in enumerate(_FEET):
            giz = server.scene.add_transform_controls(
                f"/fixed_bodies/foothold_cmd/{side}",
                scale=0.22,
                depth_test=False,
                disable_sliders=True,
                translation_limits=((-1e3, 1e3), (-1e3, 1e3), (0.0, 0.0)),
                rotation_limits=((0.0, 0.0), (0.0, 0.0), (-1e3, 1e3)),
            )
            giz.on_update(lambda ev, f=foot: self._on_gizmo(f, ev))
            self._gizmos.append(giz)

    def _set_spec_gui(self, foot: int, x: float, w: float, dyaw: float) -> None:
        """set_spec + write the clamped values back into the sliders.

        GUARDED: viser runs a slider's callback synchronously on assignment, so
        without the flag the first write-back re-reads the two still-stale
        sliders and reverts those channels.
        """
        self.set_spec(foot, x, w, dyaw)
        if not self._sliders:
            return
        self._syncing = True
        try:
            s = self._sliders[foot]
            s[0].value = float(self._spec[foot, 0])
            s[1].value = float(self._spec[foot, 1])
            s[2].value = math.degrees(float(self._spec[foot, 2]))
        finally:
            self._syncing = False

    def _on_slider(self, foot: int) -> None:
        if self._syncing:  # our own write-back, not the operator
            return
        s = self._sliders[foot]
        self.set_spec(
            foot, float(s[0].value), float(s[1].value), math.radians(s[2].value)
        )

    def _on_gizmo(self, foot: int, event) -> None:
        """Marker world pose -> placement in the mirrored stance frame."""
        self._dragging[foot] = event.phase != "end"
        ax, ay, ayaw = self._anchor[foot]
        px, py = float(event.target.position[0]), float(event.target.position[1])
        c, s = math.cos(ayaw), math.sin(ayaw)
        dx = c * (px - ax) + s * (py - ay)
        dy = -s * (px - ax) + c * (py - ay)
        sgn = 1.0 if foot == 0 else -1.0
        self._set_spec_gui(
            foot, dx, sgn * dy, _wrap_pi(_wxyz_to_yaw(event.target.wxyz) - ayaw),
        )

    def _viewed_env(self) -> int:
        b = 0 if self._get_env_idx is None else self._get_env_idx()
        return max(0, min(int(b), self.num_envs - 1))

    def _sync_viewer(self) -> None:
        """Cache the stance frames for the callbacks and re-place the markers
        (skipped for a marker the operator is dragging)."""
        if not self._gizmos:
            return
        b = self._viewed_env()
        now = time.time()
        for foot, giz in enumerate(self._gizmos):
            self._anchor[foot] = self._stance_of(b, foot)
            if self._dragging[foot]:
                if now - giz.update_timestamp < _DRAG_TIMEOUT:
                    continue  # do not fight the operator's drag
                self._dragging[foot] = False  # lost the release event
            tx, ty, tyaw = self._target_of(b, foot)
            giz.position = (float(tx), float(ty), 0.0)
            giz.wxyz = _yaw_to_wxyz(float(tyaw))

    # ── debug vis: coverage boxes + the two placements ──────────────────────

    def _debug_vis_impl(self, visualizer) -> None:
        super()._debug_vis_impl(visualizer)  # stance/target footprints + arc
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        x_c = 0.5 * (self.x_range[0] + self.x_range[1])
        x_h = 0.5 * (self.x_range[1] - self.x_range[0])
        w_c = 0.5 * (self.w_range[0] + self.w_range[1])
        w_h = 0.5 * (self.w_range[1] - self.w_range[0])
        for b in env_indices:
            for foot in (0, 1):
                ax, ay, ayaw = self._stance_of(b, foot)
                mat = _rot_z(float(ayaw))
                sgn = 1.0 if foot == 0 else -1.0
                off = mat @ np.array([x_c, sgn * w_c, 0.0])
                visualizer.add_box(
                    center=np.array([ax + off[0], ay + off[1], 0.004]),
                    size=np.array([x_h, w_h, 0.004]),
                    mat=mat,
                    color=_BOX_VIS[foot],
                    label=f"foothold_box_{foot}_{b}",
                )
                tx, ty, tyaw = self._target_of(b, foot)
                start = np.array([tx, ty, 0.02])
                end = start + 0.15 * np.array([np.cos(tyaw), np.sin(tyaw), 0.0])
                visualizer.add_arrow(
                    start=start, end=end, color=_TGT_VIS[foot], width=0.012,
                    label=f"foothold_tgt_{foot}_{b}",
                )


@dataclass(kw_only=True)
class ManualFootHoldCommandCfg(FootHoldCommandCfg):
    """PLAY-ONLY: the trained term's geometry and timing, operator targets."""

    def build(self, env: ManagerBasedRlEnv) -> ManualFootHoldCommand:
        return ManualFootHoldCommand(self, env)
