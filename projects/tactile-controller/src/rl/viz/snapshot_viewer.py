"""Viser play viewer with a "Snapshot" panel for paper-figure captures.

Every capture renders the CURRENT browser camera through MuJoCo's offscreen
renderer at an arbitrary resolution (command / sensor debug geometry included),
writes ``<out_dir>/<prefix>_<stamp>[_<k>].png`` plus a ``.json`` with the camera
in both the viser frame and the MuJoCo free-camera parametrization. Requires a
GL backend for ``mujoco.Renderer`` (headless: ``MUJOCO_GL=egl`` before import).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np
import torch
from PIL import Image

from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig
from mjlab.viewer.viser.viewer import UpdateReason


class SnapshotViserViewer(ViserPlayViewer):
    """``ViserPlayViewer`` plus a Snapshot GUI folder.

    Args:
        out_dir: Directory that receives the PNG/JSON pairs (created on first use).
        Remaining arguments are forwarded to ``ViserPlayViewer``.
    """

    def __init__(self, *args: Any, out_dir: str | Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._out_dir = Path(out_dir)
        self._renderer: OffscreenRenderer | None = None
        self._renderer_size: tuple[int, int] | None = None
        self._burst_remaining = 0
        self._burst_every = 1
        self._burst_tag = ""
        self._burst_step0 = 0
        self._burst_total = 0

    def setup(self) -> None:
        super().setup()
        gui = self._server.gui
        with gui.add_folder("Snapshot"):
            self._snap_prefix = gui.add_text("Prefix", initial_value="shot")
            self._snap_width = gui.add_number("Width", initial_value=3840, min=64, max=16384, step=1)
            self._snap_height = gui.add_number("Height", initial_value=2160, min=64, max=16384, step=1)
            self._snap_debug = gui.add_checkbox(
                "Debug geometry", initial_value=True, hint="Draw command/sensor debug vis into the render."
            )
            self._snap_flat = gui.add_checkbox(
                "Flat lighting", initial_value=True,
                hint="Headlight only: no cast shadows, no specular, no floor reflection.",
            )
            self._snap_alpha = gui.add_checkbox(
                "Transparent sky", initial_value=True, hint="Alpha 0 where nothing is rendered (depth at far plane)."
            )
            self._snap_viser_png = gui.add_checkbox(
                "Also save viser view", initial_value=False, hint="Browser render at the same size (WYSIWYG)."
            )
            capture = gui.add_button("Capture")
            self._burst_frames = gui.add_number("Burst frames", initial_value=8, min=1, max=1000, step=1)
            self._burst_steps = gui.add_number("Burst every N steps", initial_value=10, min=1, max=1000, step=1)
            burst = gui.add_button("Burst (resumes if paused)")
            default_pose = gui.add_button("Default pose + pause")
            self._cam_file = gui.add_text("Camera json", initial_value="")
            load_cam = gui.add_button("Apply camera json")

            @capture.on_click
            def _(_) -> None:
                self.request_action("CUSTOM", {"type": "snapshot"})

            @burst.on_click
            def _(_) -> None:
                self.request_action(
                    "CUSTOM",
                    {"type": "snapshot_burst", "frames": int(self._burst_frames.value), "every": int(self._burst_steps.value)},
                )

            @default_pose.on_click
            def _(_) -> None:
                self.request_action("CUSTOM", {"type": "default_pose"})

            @load_cam.on_click
            def _(_) -> None:
                self._apply_camera_json(self._cam_file.value)

    def _handle_custom_action(self, action: ViewerAction, payload: Optional[Any]) -> bool:
        if isinstance(payload, dict):
            kind = payload.get("type")
            if kind == "snapshot":
                self._capture(self._stamp())
                return True
            if kind == "snapshot_burst":
                self._burst_remaining = int(payload["frames"])
                self._burst_total = self._burst_remaining
                self._burst_every = max(1, int(payload["every"]))
                self._burst_tag = self._stamp()
                self._burst_step0 = self._step_count
                if self._is_paused:
                    self.resume()
                    self._sync_ui_state()
                return True
            if kind == "default_pose":
                self._set_default_pose()
                return True
        return super()._handle_custom_action(action, payload)

    def _execute_step(self) -> bool:
        ok = super()._execute_step()
        if ok and self._burst_remaining > 0:
            if (self._step_count - self._burst_step0) % self._burst_every == 0:
                k = self._burst_total - self._burst_remaining
                self._capture(f"{self._burst_tag}_{k:03d}")
                self._burst_remaining -= 1
                if self._burst_remaining == 0:
                    self.log(f"[INFO] Burst done -> {self._out_dir}")
        return ok

    # Capture.

    @staticmethod
    def _stamp() -> str:
        return time.strftime("%Y%m%d_%H%M%S")

    def _client_camera(self):
        clients = self._server.get_clients()
        if not clients:
            self.log("[WARN] Snapshot: no browser client connected")
            return None
        return clients[max(clients)].camera

    def _camera_record(self, cam) -> dict[str, Any]:
        """Viser camera -> world-frame MuJoCo free-camera parameters (roll dropped)."""
        offset = np.asarray(self._scene._scene_offset, dtype=np.float64)
        pos_w = np.asarray(cam.position, dtype=np.float64) - offset
        look_w = np.asarray(cam.look_at, dtype=np.float64) - offset
        fwd = look_w - pos_w
        dist = float(np.linalg.norm(fwd))
        fwd = fwd / max(dist, 1e-9)
        return {
            "viser": {
                "position": np.asarray(cam.position).tolist(),
                "look_at": np.asarray(cam.look_at).tolist(),
                "wxyz": np.asarray(cam.wxyz).tolist(),
                "up_direction": np.asarray(cam.up_direction).tolist(),
                "fov": float(cam.fov),
                "scene_offset": offset.tolist(),
            },
            "world": {"position": pos_w.tolist(), "look_at": look_w.tolist()},
            "mujoco": {
                "lookat": look_w.tolist(),
                "distance": dist,
                "azimuth": math.degrees(math.atan2(fwd[1], fwd[0])),
                "elevation": math.degrees(math.asin(float(np.clip(fwd[2], -1.0, 1.0)))),
                "fovy": math.degrees(float(cam.fov)),
            },
        }

    def _get_renderer(self, width: int, height: int) -> OffscreenRenderer:
        env = self.env.unwrapped
        if self._renderer is not None and self._renderer_size != (width, height):
            self._renderer.close()
            self._renderer = None
        if self._renderer is None:
            cfg = dataclasses.replace(
                env.cfg.viewer,
                width=width,
                height=height,
                env_idx=self._scene.env_idx,
                max_extra_envs=0,
                origin_type=ViewerConfig.OriginType.WORLD,
            )
            self._renderer = OffscreenRenderer(
                model=env.sim.mj_model,
                cfg=cfg,
                scene=env.scene,
                sim_model=env.sim.model,
                expanded_fields=env.sim.expanded_fields,
            )
            self._renderer.initialize()
            self._renderer_size = (width, height)
        return self._renderer

    def _capture(self, name: str) -> None:
        cam = self._client_camera()
        if cam is None:
            return
        width, height = int(self._snap_width.value), int(self._snap_height.value)
        rec = self._camera_record(cam)
        env = self.env.unwrapped
        mjcam = mujoco.MjvCamera()
        mjcam.type = mujoco.mjtCamera.mjCAMERA_FREE
        mjcam.lookat[:] = rec["mujoco"]["lookat"]
        mjcam.distance = rec["mujoco"]["distance"]
        mjcam.azimuth = rec["mujoco"]["azimuth"]
        mjcam.elevation = rec["mujoco"]["elevation"]
        env.sim.mj_model.vis.global_.fovy = rec["mujoco"]["fovy"]

        self._out_dir.mkdir(parents=True, exist_ok=True)
        stem = self._out_dir / f"{self._snap_prefix.value}_{name}"
        with self._sim_lock:
            renderer = self._get_renderer(width, height)
            renderer._cfg = dataclasses.replace(renderer._cfg, env_idx=self._scene.env_idx)
            callback = env.update_visualizers if self._snap_debug.value else None
            # lights, reflectance and site colours are baked into the mjv scene
            # at update time, so the style must wrap the update as well
            with self._render_style(env.sim.mj_model, flat=bool(self._snap_flat.value)), \
                    self._figure_mode(env):
                renderer.update(env.sim.data, debug_vis_callback=callback, camera=mjcam)
                img = renderer.render()
                if self._snap_alpha.value:
                    img = self._with_sky_alpha(renderer, img)
        Image.fromarray(img).save(stem.with_suffix(".png"))
        rec["step"] = int(self._step_count)
        rec["env_idx"] = int(self._scene.env_idx)
        stem.with_suffix(".json").write_text(json.dumps(rec, indent=2))
        if self._snap_viser_png.value:
            try:
                rgba = cam.get_render(height=height, width=width, transport_format="png")
                Image.fromarray(rgba).save(stem.with_name(stem.name + "_viser.png"))
            except Exception as e:  # noqa: BLE001
                self.log(f"[WARN] viser render failed: {e}")
        self.log(f"[INFO] Snapshot {stem.name}.png ({width}x{height})")

    @staticmethod
    @contextlib.contextmanager
    def _figure_mode(env):
        """Command terms exposing ``set_figure_mode(bool)`` hide operator-only
        geometry (coverage boxes) for the duration of the capture."""
        terms = [t for t in env.command_manager._terms.values() if hasattr(t, "set_figure_mode")]
        for t in terms:
            t.set_figure_mode(True)
        try:
            yield
        finally:
            for t in terms:
                t.set_figure_mode(False)

    @staticmethod
    @contextlib.contextmanager
    def _render_style(model: mujoco.MjModel, flat: bool):
        """Hide sites; with ``flat`` also drop cast shadows, specular and the
        floor reflection and lift the headlight ambient."""
        saved = (
            model.light_castshadow.copy(), model.mat_reflectance.copy(), model.site_rgba.copy(),
            model.vis.headlight.ambient.copy(), model.vis.headlight.diffuse.copy(),
            model.vis.headlight.specular.copy(), int(model.vis.headlight.active),
        )
        try:
            model.site_rgba[:, 3] = 0.0
            if flat:
                model.light_castshadow[:] = 0
                model.mat_reflectance[:] = 0.0
                model.vis.headlight.active = 1
                model.vis.headlight.ambient[:] = 0.50
                model.vis.headlight.diffuse[:] = 0.40
                model.vis.headlight.specular[:] = 0.0
            yield
        finally:
            (model.light_castshadow[:], model.mat_reflectance[:], model.site_rgba[:],
             model.vis.headlight.ambient[:], model.vis.headlight.diffuse[:],
             model.vis.headlight.specular[:], model.vis.headlight.active) = saved

    @staticmethod
    def _with_sky_alpha(renderer: OffscreenRenderer, rgb: np.ndarray) -> np.ndarray:
        """RGBA copy of ``rgb`` with alpha 0 on far-plane (background) pixels."""
        r = renderer.renderer
        r.enable_depth_rendering()
        try:
            depth = r.render()
        finally:
            r.disable_depth_rendering()
        zfar = float(renderer._model.vis.map.zfar * renderer._model.stat.extent)
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
        rgba[depth >= 0.99 * zfar, 3] = 0
        return rgba

    def _apply_camera_json(self, path: str) -> None:
        """Restore a saved WORLD camera pose onto the client (viser = world + current offset)."""
        cam = self._client_camera()
        if cam is None or not path:
            return
        try:
            rec = json.loads(Path(path).read_text())
        except Exception as e:  # noqa: BLE001
            self.log(f"[WARN] camera json: {e}")
            return
        offset = np.asarray(self._scene._scene_offset, dtype=np.float64)
        cam.up_direction = np.asarray(rec["viser"]["up_direction"])
        cam.fov = float(rec["viser"]["fov"])
        cam.position = np.asarray(rec["world"]["position"]) + offset
        cam.look_at = np.asarray(rec["world"]["look_at"]) + offset

    # Default pose.

    def _set_default_pose(self) -> None:
        """Pause and place the selected env's robot at its default keyframe, zero velocity."""
        env = self.env.unwrapped
        robot = env.scene["robot"]
        ids = torch.tensor([self._scene.env_idx], dtype=torch.int64, device=env.device)
        with self._sim_lock:
            root = robot.data.default_root_state[ids].clone()
            root[:, :3] += env.scene.env_origins[ids]
            root[:, 7:] = 0.0
            robot.write_root_state_to_sim(root, ids)
            qpos = robot.data.default_joint_pos[ids].clone()
            robot.write_joint_state_to_sim(qpos, torch.zeros_like(qpos), env_ids=ids)
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.sim.sense()
        if not self._is_paused:
            self.pause()
        self._pending_update_reasons.add(UpdateReason.ACTION)
        self._sync_ui_state()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()
