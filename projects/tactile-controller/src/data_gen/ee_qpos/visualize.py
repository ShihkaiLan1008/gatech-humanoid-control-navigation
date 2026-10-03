"""Interactive IK viewer: drag a target per hand, watch the arm solve live,
tune the cost weights with sliders and read the diagnostics.

    uv run python -m data_gen.ee_qpos.visualize [--cfg PATH] [--port 8080]

The gizmo's full pose is the target (position + orientation, roll included, as
in `generate`). Each hand is solved with the other arm at rest, the display
composes both solutions. A dragged target warm-starts from the current
posture; a jumped target (Random / Reset buttons) is solved from the current
posture and from rest, keeping the better solution.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import viser
from viser.extras import ViserUrdf

from .cost import BLOCKS, THETA_NAMES, CostCfg, GateCfg, HandSolver, accept, reason_names
from .generate import Range, pyr_from_quat, quat_axes, quat_from_pyr
from .model import CFG_PATH, SIDES, ArmModel, RobotCfg, load_cfg, load_urdf

SLIDER_RANGE = {
    "pos": (0.0, 200.0, 1.0), "ori": (0.0, 100.0, 0.5), "limit": (0.0, 500.0, 5.0),
    "limit_margin": (0.0, 0.3, 0.005), "elbow": (0.0, 500.0, 5.0), "elbow_max": (0.0, 2.09, 0.01),
    "rest": (0.0, 5.0, 0.05), "coll": (0.0, 500.0, 5.0), "coll_activation": (0.0, 0.05, 0.001),
    "swivel": (0.0, 50.0, 0.5), "swivel_far": (-1.0, 3.14, 0.01), "swivel_near": (-1.0, 3.14, 0.01),
    "swivel_r_far": (0.05, 0.60, 0.005), "swivel_r_near": (0.05, 0.60, 0.005), "swivel_band": (0.0, 1.57, 0.01),
}
"""Slider (min, max, step) per `THETA_NAMES` entry."""
COLORS = {"left": (60, 140, 255), "right": (255, 120, 45)}


def _deg(rad: float) -> str:
    return f"{int(round(np.degrees(rad)))}°"


class App:
    """State shared between the viser callbacks (writers) and the main loop
    (the only thread that runs JAX)."""

    def __init__(self, y: dict, port: int):
        self.model = ArmModel.load(RobotCfg.from_dict(y["robot"]))
        self.cost = CostCfg.from_dict(y.get("cost"))
        self.gates = GateCfg.from_dict(y.get("gates"))
        self.ranges = {s: Range.from_cfg(s, y, self.model.shoulder(s)) for s in SIDES}
        self.theta = self.cost.theta()
        self.q = {s: self.model.rest7(s).copy() for s in SIDES}
        self.target = {s: self.model.nominal_target(s) for s in SIDES}
        self.pending = {s: True for s in SIDES}
        self.multistart = {s: False for s in SIDES}
        self.lock = threading.Lock()
        self.rng = np.random.default_rng(0)

        self.solvers = {s: HandSolver(self.model, s, self.cost, batch=1) for s in SIDES}
        for s in SIDES:  # compile before serving
            pos, quat = self.target[s]
            self.solvers[s].solve(pos[None], quat[None], self.q[s][None], self.theta)
            self.solvers[s].diagnostics(self.q[s][None], pos[None], quat[None], self.theta)
        arm_pairs = [
            i for i, (a, b) in enumerate(self.model.pair_names)
            if a.split("_")[0] in SIDES and b.split("_")[0] in SIDES and a.split("_")[0] != b.split("_")[0]
        ]
        self._arm_pairs = jnp.asarray(arm_pairs, dtype=jnp.int32)
        self._cross = jax.jit(self._cross_clearance)
        self._cross(jnp.asarray(self.model.full_both(jnp.asarray(self.q["left"]), jnp.asarray(self.q["right"]))))

        self.server = viser.ViserServer(port=port)
        self._build_scene()
        self._build_gui()
        self._update_robot()

    def _cross_clearance(self, q29: jax.Array) -> jax.Array:
        d = self.model.coll.compute_self_collision_distance(self.model.robot, q29) - jnp.asarray(self.model.pair_floor)
        return d[self._arm_pairs].min()

    # -- scene --------------------------------------------------------------

    def _build_scene(self) -> None:
        sv = self.server
        sv.scene.set_up_direction("+z")
        sv.scene.add_frame("/pelvis", axes_length=0.2, axes_radius=0.005)
        sv.scene.add_grid("/grid", width=3.0, height=3.0, position=(0.0, 0.0, -0.793))
        urdf = load_urdf(self.model.cfg.urdf, meshes=True)
        self.urdf_vis = ViserUrdf(sv, urdf, root_node_name="/robot", mesh_color_override=(0.72, 0.72, 0.74))
        self.gizmo, self.box, self.arrows, self.shell = {}, {}, {}, {}
        for s in SIDES:
            pos, quat = self.target[s]
            self.gizmo[s] = sv.scene.add_transform_controls(
                f"/target/{s}", scale=0.15, depth_test=False, disable_sliders=True,
                position=tuple(float(v) for v in pos), wxyz=tuple(float(v) for v in quat),
            )
            self.gizmo[s].on_update(lambda ev, s=s: self._on_gizmo(s, ev))
            r = self.ranges[s]
            centre = ((r.x_min + r.x_max) / 2, r.sign * (r.y_gap + r.y_max) / 2, (r.z_lo + r.z_hi) / 2)
            self.box[s] = sv.scene.add_box(
                f"/range/{s}/box", color=COLORS[s], wireframe=True,
                dimensions=(r.x_max - r.x_min, r.y_max - r.y_gap, r.z_hi - r.z_lo),
                position=centre, visible=False,
            )
            self.shell[s] = {k: self._cylinder(s, k, rad, True) for k, rad in (("min", r.r_min), ("max", r.r_max))}
            self.arrows[s] = sv.scene.add_line_segments(
                f"/arrow/{s}", points=np.zeros((2, 2, 3), np.float32),
                colors=np.array([[COLORS[s], COLORS[s]], [(40, 40, 40), (40, 40, 40)]], np.uint8),
                line_width=4.0,
            )
        self.capsule_mesh = None

    def _cylinder(self, side: str, kind: str, radius: float, visible: bool):
        """Wireframe vertical cylinder of the region shell about that arm's
        shoulder axis, spanning the z band: two rings plus vertical lines."""
        r = self.ranges[side]
        cx, cy = float(r.center[0]), float(r.center[1])
        th = np.linspace(0.0, 2 * np.pi, 73)
        ring = np.stack([cx + radius * np.cos(th), cy + radius * np.sin(th)], axis=1)
        segs = []
        for zz in (r.z_lo, r.z_hi):
            pts = np.concatenate([ring, np.full((ring.shape[0], 1), zz)], axis=1)
            segs.append(np.stack([pts[:-1], pts[1:]], axis=1))
        th16 = np.linspace(0.0, 2 * np.pi, 17)[:-1]
        top = np.stack([cx + radius * np.cos(th16), cy + radius * np.sin(th16), np.full(th16.shape, r.z_hi)], axis=1)
        bot = top.copy()
        bot[:, 2] = r.z_lo
        segs.append(np.stack([bot, top], axis=1))
        points = np.concatenate(segs).astype(np.float32)
        colors = np.broadcast_to(np.array(COLORS[side], np.uint8), (points.shape[0], 2, 3)).copy()
        return self.server.scene.add_line_segments(
            f"/range/{side}/r_{kind}", points=points, colors=colors, line_width=2.0, visible=visible,
        )

    def _set_r(self, side: str) -> None:
        """Apply the r_min / r_max sliders: region, shells, then re-solve."""
        r_min, r_max = self.r_sliders[side]["min"].value, self.r_sliders[side]["max"].value
        self.ranges[side] = replace(self.ranges[side], r_min=r_min, r_max=r_max)
        for k, rad in (("min", r_min), ("max", r_max)):
            self.shell[side][k].remove()
            self.shell[side][k] = self._cylinder(side, k, rad, self.show_shell.value)
        self._mark_all()

    def _on_gizmo(self, side: str, event) -> None:
        pos = np.asarray(event.target.position, dtype=np.float32)
        wxyz = np.asarray(event.target.wxyz, dtype=np.float32)
        with self.lock:
            self.target[side] = (pos, wxyz / max(np.linalg.norm(wxyz), 1e-9))
            self.pending[side] = True

    def _set_gizmo(self, side: str, pos: np.ndarray, wxyz: np.ndarray) -> None:
        self.gizmo[side].position = tuple(float(v) for v in pos)
        self.gizmo[side].wxyz = tuple(float(v) for v in wxyz)
        with self.lock:
            self.target[side] = (np.asarray(pos, np.float32), np.asarray(wxyz, np.float32))
            self.pending[side] = True
            self.multistart[side] = True

    # -- gui ----------------------------------------------------------------

    def _build_gui(self) -> None:
        gui = self.server.gui
        self.sliders = {}
        self._syncing = False
        with gui.add_folder("Cost"):
            for i, name in enumerate(THETA_NAMES):
                lo, hi, step = SLIDER_RANGE[name]
                self.sliders[name] = gui.add_slider(name, lo, hi, step, float(self.theta[i]))
                self.sliders[name].on_update(lambda _: self._on_slider())
            reset_w = gui.add_button("Reset weights to yaml")

            @reset_w.on_click
            def _(_) -> None:
                self._syncing = True
                for i, name in enumerate(THETA_NAMES):
                    self.sliders[name].value = float(self.cost.theta()[i])
                self._syncing = False
                self._on_slider()

        with gui.add_folder("Actions"):
            b_reset = gui.add_button("Reset targets to nominal")
            b_random = gui.add_button("Random targets in range")
            b_rest = gui.add_button("Re-solve from rest")
            self.solve_both = gui.add_checkbox("solve both hands", True)

            @b_reset.on_click
            def _(_) -> None:
                for s in SIDES:
                    with self.lock:
                        self.q[s] = self.model.rest7(s).copy()
                    self._set_gizmo(s, *self.model.nominal_target(s))

            @b_random.on_click
            def _(_) -> None:
                for s in SIDES:
                    self._random_target(s)

            @b_rest.on_click
            def _(_) -> None:
                with self.lock:
                    for s in SIDES:
                        self.q[s] = self.model.rest7(s).copy()
                        self.pending[s] = True

        self.r_sliders = {}
        with gui.add_folder("Region r_min / r_max (horizontal m from the shoulder axis)"):
            for s in SIDES:
                r = self.ranges[s]
                self.r_sliders[s] = {
                    "min": gui.add_slider(f"{s} r_min", 0.05, 0.60, 0.005, r.r_min),
                    "max": gui.add_slider(f"{s} r_max", 0.10, 0.70, 0.005, r.r_max),
                }
                for h in self.r_sliders[s].values():
                    h.on_update(lambda _, s=s: self._set_r(s))

        with gui.add_folder("Show"):
            self.show_capsules = gui.add_checkbox("collision capsules", False)
            self.show_shell = gui.add_checkbox("r_min / r_max shells", True)
            self.show_box = gui.add_checkbox("range box", False)
            self.show_arrows = gui.add_checkbox("tool-x arrows", True)
            self.show_shell.on_update(lambda _: [setattr(h, "visible", self.show_shell.value) for s in SIDES for h in self.shell[s].values()])
            self.show_box.on_update(lambda _: [setattr(self.box[s], "visible", self.show_box.value) for s in SIDES])
            self.show_arrows.on_update(lambda _: [setattr(self.arrows[s], "visible", self.show_arrows.value) for s in SIDES])
            self.show_capsules.on_update(lambda _: self._mark_all())

        self.readout = {}
        for s in SIDES:
            with gui.add_folder(f"{s} hand"):
                self.readout[s] = gui.add_markdown("waiting")
        with gui.add_folder("Both"):
            self.readout["both"] = gui.add_markdown("")

    def _random_target(self, side: str) -> None:
        """Move that hand's gizmo to a uniform draw strictly inside the region."""
        pos, pyr = self.ranges[side].sample(self.rng, 1)
        self._set_gizmo(side, pos[0], quat_from_pyr(pyr)[0])

    def _on_slider(self) -> None:
        if self._syncing:
            return
        with self.lock:
            self.theta = np.array([self.sliders[n].value for n in THETA_NAMES], dtype=np.float32)
        self._mark_all()

    def _mark_all(self) -> None:
        with self.lock:
            for s in SIDES:
                self.pending[s] = True

    # -- main loop ------------------------------------------------------------

    def _update_robot(self) -> None:
        q29 = np.asarray(self.model.full_both(jnp.asarray(self.q["left"]), jnp.asarray(self.q["right"])))
        self.urdf_vis.update_cfg(q29)
        if self.show_capsules.value:
            mesh = self.model.capsules(q29)
            self.capsule_mesh = self.server.scene.add_mesh_simple(
                "/capsules", np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32),
                color=(255, 200, 60), opacity=0.35, side="double",
            )
        elif self.capsule_mesh is not None:
            self.capsule_mesh.remove()
            self.capsule_mesh = None
        cross = float(self._cross(jnp.asarray(q29)))
        self.readout["both"].content = f"cross-arm clearance **{cross * 1e3:+.1f} mm** (both arms at their solutions)"

    def _solve(self, side: str) -> None:
        """Solve `side` for its current target and refresh its readout and arrows.

        Starts from the current posture; when `multistart` is set (a jumped target)
        also from rest, keeping the accepted solution with the lower residual norm."""
        with self.lock:
            pos, quat = self.target[side]
            theta = self.theta.copy()
            starts = [("current", self.q[side].copy())]
            if self.multistart[side]:
                starts.append(("rest", self.model.rest7(side).copy()))
                self.multistart[side] = False
        solver = self.solvers[side]
        t0 = time.perf_counter()
        best = None
        for start, q0 in starts:
            q, it = solver.solve(pos[None], quat[None], q0[None], theta)
            diag = solver.diagnostics(q, pos[None], quat[None], theta)
            ok, reason = accept(diag, self.gates, self.cost)
            key = (not bool(ok[0]), float(np.linalg.norm(diag.res_norm[0])))
            if best is None or key < best[0]:
                best = (key, start, q, it, diag, ok, reason)
        _, start, q, it, diag, ok, reason = best
        ms = (time.perf_counter() - t0) * 1e3
        with self.lock:
            self.q[side] = q[0]
        pair = self.model.pair_names[int(diag.min_pair[0])]
        blocks = "  ".join(f"{b} {v:.2f}" for b, v in zip(BLOCKS, diag.res_norm[0]))
        verdict = "**OK**" if ok[0] else f"**FAIL** ({', '.join(reason_names(int(reason[0])))})"
        if len(starts) > 1:
            verdict += f" · best start: {start}"
        rg = self.ranges[side]
        r_sh = float(np.linalg.norm(pos[:2] - rg.center[:2]))
        pyr = pyr_from_quat(quat.astype(np.float64))
        inside = bool(rg.contains(pos[None].astype(np.float64), pyr[None], tol=0.05)[0])
        pitch, yaw, roll = (f"{int(round(v))}°" for v in pyr)
        self.readout[side].content = (
            f"{verdict} · {ms:.1f} ms · {int(it[0])} iters  \n"
            f"target horizontal r from the shoulder axis **{r_sh:.3f} m** · pitch {pitch} yaw {yaw} roll {roll} · "
            f"{'inside the region' if inside else '**OUTSIDE the region**'}  \n"
            f"pos err **{diag.pos_err[0] * 1e3:.2f} mm** · ori err **{diag.ori_err[0]:.2f}°**  \n"
            f"clearance **{diag.min_clear[0] * 1e3:+.1f} mm** ({pair[0]} × {pair[1]})  \n"
            f"elbow **{_deg(diag.elbow[0])}** ({diag.elbow[0]:.2f} rad) · swivel ψ **{_deg(diag.swivel[0])}** "
            f"(target {_deg(diag.swivel_ref[0])}) · limit margin {_deg(diag.limit_margin[0])} · manip {diag.manip[0]:.4f}  \n"
            f"residuals: {blocks}"
        )
        tip, d, ex = diag.tip_pos[0], quat_axes(quat)[0], quat_axes(diag.tip_quat[0])[0]
        self.arrows[side].points = np.stack(
            [np.stack([pos, pos + 0.12 * d]), np.stack([tip, tip + 0.12 * ex])]
        ).astype(np.float32)

    def run(self) -> None:
        print(f"viser serving on http://localhost:{self.server.get_port()}  (Ctrl-C to stop)")
        try:
            while True:
                did = False
                for s in SIDES:
                    with self.lock:
                        go = self.pending[s]
                        self.pending[s] = False
                    if go and (self.solve_both.value or s == "left"):
                        self._solve(s)
                        did = True
                if did:
                    self._update_robot()
                time.sleep(0.005)
        except KeyboardInterrupt:
            print("\nstopped")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cfg", default=str(CFG_PATH), help="pipeline yaml")
    ap.add_argument("--port", type=int, default=8080, help="viser port")
    args = ap.parse_args()
    App(load_cfg(args.cfg), args.port).run()


if __name__ == "__main__":
    main()
