"""PLAY-ONLY: operator-assigned foot placement for the foot_mimic line.

Third source for the same ``step_cmd`` interface the lower policy was trained
on (clips -> :class:`FootPlacementCommand`, Raibert planner ->
:class:`VelocityFootstepCommand`, and here: a HUMAN).  Only the target changes;
the step machinery, the reset teleport, the synthesized reference views the
critic/rewards read, and the [B,7] command layout are the planner's, so any
joint_foot_mimic checkpoint plays here strict=True.

Interaction (viser):

* two ground boxes = the TRAINED landing coverage box for each foot, drawn in
  its stance frame (the OTHER foot's measured pose), each with a draggable
  marker: drag in xy, twist about z.  A drag outside the box snaps back in.
  (``coverage boxes`` checkbox hides them; snapshot captures never draw them.)
* debug vis = foot_hold-style bead footprints under the reference feet (red
  target while a step is in flight, orange otherwise), the lift-off -> target
  swing arc and a bead at the measured swing foot.
* ``step [left | right]`` takes exactly one step with that foot, then stands.
* ``walk`` steps continuously, alternating feet, with the current placements.
* per-foot ``x / width / dyaw`` sliders for exact values.

The placement is stored RELATIVE to the stance foot (mirrored stance frame,
exactly the trained ``target_m``), so it survives the robot moving: the marker
keeps its spot inside the box while the box walks along with the feet.  While a
step is in flight its target is COMMITTED — the marker shows what was sent, and
a drag applies to the next step.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

from .raibert_footstep import VelocityFootstepCommand, VelocityFootstepCommandCfg

if TYPE_CHECKING:
    import viser

    from mjlab.envs import ManagerBasedRlEnv

_FEET = ("left", "right")
_DRAG_TIMEOUT = 5.0
"""Safety net only: a drag ends on the gizmo's ``end`` phase; this releases a
marker whose release event was lost (client gone mid-drag)."""

_BOX_VIS = ((0.10, 0.80, 0.25, 0.20), (0.20, 0.45, 0.95, 0.20))
_SOLE_STANCE_VIS = (1.00, 0.65, 0.00, 1.0)
_SOLE_TARGET_VIS = (0.90, 0.10, 0.10, 1.0)
_ARC_VIS = (0.05, 0.20, 0.90, 1.0)
_ARC_NOW_VIS = (0.70, 0.20, 0.95, 1.0)
_FOOT_HALF = (0.09, 0.03)
_FOOT_X_OFF = 0.04
"""Foot-site support rectangle (foot_hold's footprint), site = ankle + 0.04 x."""
_FOOT_BEADS = 20
_ARC_APEX = 0.065
_ARC_SEGMENTS = 16


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_to_wxyz(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _wxyz_to_yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _bead_ring(n: int, hx: float, hy: float) -> np.ndarray:
    """(n, 2) points equally spaced along the perimeter of the rectangle with
    half-extents (hx, hy), local frame."""
    corners = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy], [hx, hy]])
    seg = corners[1:] - corners[:-1]
    seg_len = np.linalg.norm(seg, axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    dist = np.linspace(0.0, cum[-1], n + 1)[:-1]
    j = np.clip(np.searchsorted(cum, dist, side="right") - 1, 0, 3)
    f = ((dist - cum[j]) / seg_len[j])[:, None]
    return corners[j] + f * seg[j]


def _x_aligned_rotation(x: np.ndarray) -> np.ndarray:
    """(3, 3) rotation whose local x axis is the unit vector ``x``."""
    up = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    y = np.cross(up, x)
    y /= max(np.linalg.norm(y), 1e-9)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)


def _rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class ManualFootstepCommand(VelocityFootstepCommand):
    """Foot placement assigned by the operator instead of a twist planner."""

    cfg: "ManualFootstepCommandCfg"

    def __init__(self, cfg: "ManualFootstepCommandCfg", env: ManagerBasedRlEnv):
        assert cfg.fixed_twist == (0.0, 0.0, 0.0), (
            "the manual term owns stepping — the twist must stay zero so the "
            "planner's reset comes up standing"
        )
        super().__init__(cfg, env)

        # The drawn boxes ARE the clamp _start_step applies.
        self.x_range = cfg.landing_x_range
        self.w_range = (cfg.nominal_width + cfg.landing_dy_range[0],
                        cfg.nominal_width + cfg.landing_dy_range[1])
        self.yaw_range = cfg.landing_yaw_range

        # Operator state.  Plain numpy/python: viser callbacks run off the sim
        # thread and must never touch a CUDA tensor.
        # [foot][x, width, yaw_m] in the MIRRORED stance frame (+yaw = outward).
        self._spec = np.array([[0.0, cfg.nominal_width, 0.0]] * 2)
        self._walk = False
        self._steps_left = 0
        self._lead_foot: int | None = None
        self._anchor = np.zeros((2, 3))  # [foot][x, y, yaw] of its stance foot
        self._liftoff = torch.zeros(self.num_envs, 3, device=self.device)
        self._show_boxes = True
        self._figure_mode = False

        self._get_env_idx: Callable[[], int] | None = None
        self._gizmos: list["viser.TransformControlsHandle"] = []
        self._sliders: list[list[Any]] = []
        self._dragging = [False, False]
        self._syncing = False

        print(
            "[INFO] manual foot placement: markers + step buttons live in the "
            "viser GUI (run play with --viewer viser)"
        )

    # ── the operator replaces the planner ───────────────────────────────────

    def _plan_target(self, env_ids, swing, stance_yaw, s, T):
        """The operator's placement for whichever foot is swinging."""
        del stance_yaw, s, T
        spec = torch.as_tensor(
            self._spec, device=self.device, dtype=torch.float32
        )[swing]
        return spec[:, 0], spec[:, 1] - self.cfg.nominal_width, spec[:, 2]

    def _start_step(self, env_ids, swing, stance_pos, stance_yaw):
        idx = torch.arange(len(env_ids), device=self.device)
        self._liftoff[env_ids] = self.robot_feet_pos_w[env_ids][idx, swing]
        super()._start_step(env_ids, swing, stance_pos, stance_yaw)

    def _update_command(self):
        """Step only when the operator asks; otherwise hold the stand."""
        active = ~self.standing
        self.t[active] += 1
        boundary = (active & (self.t >= self.step_frames)) | self.standing
        if not bool(boundary.any()):
            return
        ids = boundary.nonzero(as_tuple=False).squeeze(-1)

        if self._walk or self._steps_left > 0:
            if self._steps_left > 0:
                self._steps_left -= 1
            side, self._lead_foot = self._lead_foot, None
            swing = (
                1 - self.swing[ids] if side is None
                else torch.full_like(self.swing[ids], int(side))
            )
            stance = 1 - swing
            idx = torch.arange(len(ids), device=self.device)
            self._start_step(
                ids, swing,
                self.robot_feet_pos_w[ids][idx, stance],
                self.robot_feet_yaw[ids][idx, stance],
            )
        else:
            enter = ids[~self.standing[ids]]
            if len(enter) > 0:
                self._enter_stand(enter)

    def _update_metrics(self):
        super()._update_metrics()
        self._sync_viewer()

    # ── operator surface (headless-safe: plain attributes) ──────────────────

    def set_spec(self, foot: int, x: float, width: float, dyaw: float) -> None:
        """Place one foot (0 = left), clamped into the trained box.  ``width``
        is the ABSOLUTE lateral distance from the stance foot; ``dyaw`` is
        mirrored (+ = toed outward)."""
        self._spec[foot] = (
            float(np.clip(x, *self.x_range)),
            float(np.clip(width, *self.w_range)),
            float(np.clip(dyaw, *self.yaw_range)),
        )

    def set_walk(self, walk: bool) -> None:
        """Continuous alternating stepping with the current placements."""
        self._walk = bool(walk)

    def set_figure_mode(self, on: bool) -> None:
        """Hide the operator-only coverage boxes (snapshot captures)."""
        self._figure_mode = bool(on)

    def request_step(self, foot: int | None = None, n: int = 1) -> None:
        """Take ``n`` steps; ``foot`` (0 = left, 1 = right) picks which foot
        swings on the next one (None = alternate)."""
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
            p = self.ref_feet_pos_w[b, foot, 0:2].cpu().numpy()
            return np.array([p[0], p[1], float(self.ref_feet_yaw[b, foot])])
        ax, ay, ayaw = self._anchor[foot]
        x, w, yaw_m = self._spec[foot]
        sgn = 1.0 if foot == 0 else -1.0
        c, s = math.cos(ayaw), math.sin(ayaw)
        return np.array([ax + c * x - s * sgn * w,
                         ay + s * x + c * sgn * w,
                         _wrap_pi(ayaw + sgn * yaw_m)])

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
                self._walk = bool(walk.value)

            steps = server.gui.add_button_group("step", options=["left", "right"])

            @steps.on_click
            def _(event) -> None:
                self.request_step(0 if event.target.value == "left" else 1)

            reset_btn = server.gui.add_button("place in place")
            boxes = server.gui.add_checkbox("coverage boxes", initial_value=self._show_boxes)

            @boxes.on_update
            def _(_ev) -> None:
                self._show_boxes = bool(boxes.value)

            @reset_btn.on_click
            def _(_ev) -> None:
                for foot in (0, 1):
                    self._set_spec_gui(foot, 0.0, self.cfg.nominal_width, 0.0)

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
                f"/fixed_bodies/footstep_cmd/{side}",
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
        sliders and reverts those channels (a diagonal drag would keep only x).
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
            foot, dx, sgn * dy,
            sgn * _wrap_pi(_wxyz_to_yaw(event.target.wxyz) - ayaw),
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

    # ── debug vis: soles, swing arc, coverage boxes ─────────────────────────

    def _debug_vis_impl(self, visualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        feet_pos = self._ref_feet_pos.cpu().numpy()
        feet_yaw = self._ref_feet_yaw.cpu().numpy()
        swing = self.swing.cpu().numpy()
        standing = self.standing.cpu().numpy()
        liftoff = self._liftoff.cpu().numpy()
        now = self.robot_feet_pos_w.cpu().numpy()
        for b in env_indices:
            for foot in (0, 1):
                is_target = (not standing[b]) and foot == swing[b]
                self._draw_sole(
                    visualizer, feet_pos[b, foot], feet_yaw[b, foot],
                    _SOLE_TARGET_VIS if is_target else _SOLE_STANCE_VIS,
                    f"sole_{foot}_{b}",
                )
            if not standing[b]:
                self._draw_arc(visualizer, liftoff[b], feet_pos[b, swing[b]], b)
                visualizer.add_sphere(
                    center=now[b, swing[b]], radius=0.02, color=_ARC_NOW_VIS,
                    label=f"swing_now_{b}",
                )
            if self._show_boxes and not self._figure_mode:
                self._draw_boxes(visualizer, b)

    @staticmethod
    def _draw_sole(visualizer, pos, yaw, color, tag: str) -> None:
        """foot_hold-style footprint: centre bead + a ring of beads on the
        foot-site support rectangle, at ground level."""
        mat = _rot_z(float(yaw))
        c = np.array([pos[0], pos[1], 0.0]) + mat @ np.array([_FOOT_X_OFF, 0.0, 0.0])
        visualizer.add_sphere(
            center=c + np.array([0.0, 0.0, 0.02]), radius=0.025, color=color, label=f"{tag}_c",
        )
        for i, (lx, ly) in enumerate(_bead_ring(_FOOT_BEADS, *_FOOT_HALF)):
            visualizer.add_sphere(
                center=c + mat @ np.array([lx, ly, 0.01]), radius=0.008, color=color,
                label=f"{tag}_{i}",
            )

    @staticmethod
    def _draw_arc(visualizer, p0, p1, b: int) -> None:
        """Zero-end-velocity cubic in xy, sin² bump in z, lift-off -> target,
        drawn as foot_hold's box segments."""
        u = np.linspace(0.0, 1.0, _ARC_SEGMENTS + 1)
        h = u * u * (3.0 - 2.0 * u)
        pts = p0[None, :] + (p1 - p0)[None, :] * h[:, None]
        pts[:, 2] += _ARC_APEX * np.sin(np.pi * u) ** 2
        for k in range(_ARC_SEGMENTS):
            d = pts[k + 1] - pts[k]
            length = float(np.linalg.norm(d))
            if length < 1e-6:
                continue
            visualizer.add_box(
                center=0.5 * (pts[k] + pts[k + 1]),
                size=np.array([0.5 * length + 0.004, 0.008, 0.008]),
                mat=_x_aligned_rotation(d / length),
                color=_ARC_VIS, label=f"swing_arc_{b}_{k}",
            )

    def _draw_boxes(self, visualizer, b: int) -> None:
        x_c = 0.5 * (self.x_range[0] + self.x_range[1])
        x_h = 0.5 * (self.x_range[1] - self.x_range[0])
        w_c = 0.5 * (self.w_range[0] + self.w_range[1])
        w_h = 0.5 * (self.w_range[1] - self.w_range[0])
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
                label=f"footstep_box_{foot}_{b}",
            )


@dataclass(kw_only=True)
class ManualFootstepCommandCfg(VelocityFootstepCommandCfg):
    """PLAY-ONLY: the planner's geometry and reset, operator-assigned targets."""

    fixed_twist: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    """Zero and unused: stepping is the operator's buttons.  Keeps the
    inherited reset standing and the reference twist views honest."""

    twist_resample_s: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> ManualFootstepCommand:
        return ManualFootstepCommand(self, env)
