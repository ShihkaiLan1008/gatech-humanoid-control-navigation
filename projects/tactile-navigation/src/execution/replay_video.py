"""Render a demo video by REPLAYING a recorded episode -- never by re-running it.

Rendering during an episode was measured changing the physics trajectory
(``render_mode`` changes environment construction, so the same seed diverges).
Every validated episode therefore runs without rendering and records its full
generalized coordinates at video rate (``states.npz``); this module builds a
render environment for the same scene, writes each recorded state into it,
calls ``forward`` (kinematics only -- the policy and the physics never run),
and composites the frame with a synchronized telemetry panel:

    floorplan + planned route + goal        robot trajectory so far
    active pose hypotheses (mass-sized)     selected tactile station
    mode strip (WALK/STABILIZE/PROBE/...)   all-hand force vs 12/25 N
    route decisions and belief updates      hypothesis masses

    # one episode
    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
      uv run --project tactile_controller python -m execution.replay_video \
        --episode out/whole_house/<ep_dir> --scene <scene> \
        --out <ep_dir>/demo.mp4

    # side-by-side comparison (two finished demo.mp4 files)
    python -m execution.replay_video --hstack a.mp4 b.mp4 --out pair.mp4 \
        --labels belief_aware no_touch
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

MODE_NAMES = {"LOCOMOTION": "WALK", "STABILIZE": "STABILIZE", "PROBE": "PROBE",
              "TURN": "TURN", "ARRIVED": "ARRIVE", "FAILED": "FAILED"}
MODE_COLOR = {"WALK": "#2b8cbe", "STABILIZE": "#a6bddb", "PROBE": "#e34a33",
              "TURN": "#31a354", "ARRIVE": "#756bb1", "FAILED": "#636363",
              "TOUCH": "#e6873c", "LANDMARK_PROBE": "#c51b8a"}


class Telemetry:
    """Everything the overlay needs, indexed by time."""

    def __init__(self, ep: dict, scene):
        tr = ep["tick_trace"]
        self.t = np.asarray([r["t"] for r in tr])
        self.xy_w = np.asarray([r["xy"] for r in tr])
        self.mode = [MODE_NAMES.get(r["mode"], r["mode"]) for r in tr]
        self.force = np.asarray([max(r.get("force_l", 0.0), r.get("force_r", 0.0))
                                 for r in tr])
        self.hyp = ep.get("hypothesis_trace", [])
        self.hyp_t = np.asarray([h["t"] for h in self.hyp]) if self.hyp else np.zeros(0)
        self.sigma = np.asarray([h["s_sigma"] for h in self.hyp]) if self.hyp else np.zeros(0)
        self.directives = ep.get("directives", [])
        self.dir_t = np.asarray([d["t"] for d in self.directives])
        self.probes = ep.get("probes", [])
        self.disambig = set(getattr(scene, "disambiguating_stations", ()) or ())
        self.scene = scene
        self.ep = ep

    def _demo_mode(self, raw: str, directive) -> str:
        """PROBE splits into TOUCH (maintenance) vs LANDMARK_PROBE
        (disambiguator station) for the contact-rich overlay."""
        if raw != "PROBE" or directive is None:
            return raw
        if str(directive.get("reason", "")).startswith("touch_step_maintenance"):
            return "TOUCH"
        if directive.get("station") in self.disambig:
            return "LANDMARK_PROBE"
        return raw

    def at(self, t: float) -> dict:
        i = int(np.clip(np.searchsorted(self.t, t + 1e-9) - 1, 0, len(self.t) - 1))
        j = int(np.clip(np.searchsorted(self.hyp_t, t + 1e-9) - 1, 0,
                        max(len(self.hyp) - 1, 0))) if self.hyp else -1
        k = int(np.clip(np.searchsorted(self.dir_t, t + 1e-9) - 1, 0,
                        max(len(self.directives) - 1, 0))) if self.directives else -1
        directive = self.directives[k] if k >= 0 else None
        return {
            "i": i, "j": j, "mode": self._demo_mode(self.mode[i], directive),
            "force": float(self.force[i]),
            "hyp": self.hyp[j] if j >= 0 else None,
            "directive": directive,
            "presses": [p for p in self.probes if p["t"] <= t],
        }


def _panel_figure(scene, height_px: int, width_px: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon as MplPoly

    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes([0.06, 0.42, 0.90, 0.545])
    axs = fig.add_axes([0.06, 0.225, 0.90, 0.115])
    axf = fig.add_axes([0.06, 0.045, 0.90, 0.125])

    for body in scene.plan.bodies:
        ax.add_patch(MplPoly(np.asarray(body.verts), closed=True,
                             facecolor="#bdbdbd", edgecolor="#636363", lw=0.8, zorder=1))
    for e in scene.plan.edges:
        a, b = np.asarray(e.a), np.asarray(e.b)
        ax.plot([a[0], b[0]], [a[1], b[1]], "--", color="#9ecae1", lw=1.4, zorder=2)
    g = np.asarray(scene.goal_map_xy)
    ax.add_patch(Circle(g, scene.goal_radius_m, facecolor="none",
                        edgecolor="#31a354", lw=1.6, zorder=3))
    ax.annotate("goal", g, textcoords="offset points", xytext=(6, 6),
                fontsize=8, color="#31a354")
    xs = [v[0] for b in scene.plan.bodies for v in np.asarray(b.verts)]
    ys = [v[1] for b in scene.plan.bodies for v in np.asarray(b.verts)]
    ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)

    art = {
        "path": ax.plot([], [], "-", color="#252525", lw=1.6, zorder=4)[0],
        "robot": ax.plot([], [], "o", color="#252525", ms=7, zorder=6)[0],
        "heading": ax.plot([], [], "-", color="#252525", lw=1.4, zorder=6)[0],
        "hyps": ax.scatter([], [], s=[], c="#e34a33", alpha=0.75, zorder=5),
        "station": ax.plot([], [], "*", color="#e34a33", ms=14, zorder=6,
                           mec="k", mew=0.5)[0],
        "press_c": ax.scatter([], [], s=20, c="#31a354", marker="o", zorder=5,
                              edgecolors="k", linewidths=0.4),
        "press_f": ax.scatter([], [], s=20, facecolors="none", marker="o",
                              zorder=5, edgecolors="#2b8cbe", linewidths=1.1),
        "press_x": ax.scatter([], [], s=10, c="#969696", marker="x", zorder=5),
        "turn_mark": ax.plot([], [], "D", color="none", ms=10, mec="#31a354",
                             mew=1.8, zorder=7)[0],
        "turn_txt": ax.text(0, 0, "", fontsize=7, color="#31a354",
                            fontweight="bold", ha="center", va="bottom"),
        "mode_txt": ax.text(0.01, 1.015, "", transform=ax.transAxes, fontsize=11,
                            fontweight="bold"),
        "mass_txt": ax.text(0.99, 1.015, "", transform=ax.transAxes, fontsize=9,
                            ha="right"),
        "ticker": fig.text(0.06, 0.395, "", fontsize=7.0, family="monospace",
                           va="top"),
    }
    axs.axhline(0.12, color="#756bb1", ls="--", lw=1.0)
    axs.text(0.995, 0.12, " commit gate", fontsize=6, color="#756bb1",
             va="bottom", ha="right", transform=axs.get_yaxis_transform())
    axs.set_ylim(0, 0.42)
    axs.tick_params(labelsize=7)
    axs.set_ylabel("sigma m", fontsize=7)
    art["sigma"] = axs.plot([], [], "-", color="#756bb1", lw=1.1)[0]
    art["sigma_now"] = axs.plot([], [], "o", color="#756bb1", ms=4)[0]
    art["axs"] = axs
    axf.axhline(25.0, color="crimson", ls="--", lw=1.0)
    axf.axhline(12.0, color="darkorange", ls=":", lw=1.0)
    axf.set_ylim(0, 30)
    axf.tick_params(labelsize=7)
    axf.set_ylabel("hand N", fontsize=7)
    art["force"] = axf.plot([], [], "-", color="#252525", lw=0.9)[0]
    art["force_now"] = axf.plot([], [], "o", color="#e34a33", ms=4)[0]
    art["axf"] = axf
    return fig, ax, art


def _draw_panel(fig, ax, art, tel: Telemetry, world_to_map, t: float) -> np.ndarray:
    st = tel.at(t)
    i = st["i"]
    path = np.asarray([world_to_map(xy) for xy in tel.xy_w[:i + 1:4]])
    if len(path):
        art["path"].set_data(path[:, 0], path[:, 1])
        pos = world_to_map(tel.xy_w[i])
        art["robot"].set_data([pos[0]], [pos[1]])
    hyp = st["hyp"]
    if hyp is not None:
        leg = tel.scene.legs[hyp["leg"]]
        a, u = np.asarray(leg["origin"]), np.asarray(leg["u"])
        pts = np.asarray([a + s * u for s in hyp["hyp_s"]])
        art["hyps"].set_offsets(pts)
        art["hyps"].set_sizes(60 + 240 * np.asarray(hyp["masses"]))
        art["mass_txt"].set_text(
            "hypotheses " + "/".join(f"{m:.2f}" for m in hyp["masses"])
            + f"   sigma {hyp['s_sigma']:.2f} m")
    d = st["directive"]
    if d and d.get("station"):
        stn = tel.scene.stations.get(d["station"])
        if stn is not None:
            art["station"].set_data([stn.xy[0]], [stn.xy[1]])
    else:
        art["station"].set_data([], [])
    presses = st["presses"]
    by_kind = {"contact": [], "swept_free": [], "invalid": []}
    for p in presses:
        stn = tel.scene.stations.get(p["station"])
        if stn is not None:
            kind = p.get("kind") if p.get("kind") in by_kind else "invalid"
            by_kind[kind].append(stn.xy + float(p["offset_m"]) * stn.travel)
    for key, artist in (("contact", "press_c"), ("swept_free", "press_f"),
                        ("invalid", "press_x")):
        pts = by_kind[key]
        art[artist].set_offsets(np.asarray(pts) if pts else np.empty((0, 2)))

    # where the CURRENT tracked mode will put the next turn: the robot walks
    # until its believed along-leg position reaches the corner, so the
    # physical turn point is the physical position plus the believed
    # remaining distance.  This marker JUMPS by the hypothesis separation
    # when landmark evidence flips the tracked mode -- the "selected branch
    # before/after" the demo shows.
    hyp = st["hyp"]
    if (hyp is not None and hyp.get("masses")
            and hyp["leg"] < len(tel.scene.legs) - 1):
        leg = tel.scene.legs[hyp["leg"]]
        a, u = np.asarray(leg["origin"], float), np.asarray(leg["u"], float)
        turn_s = float(leg["terminus"]) + tel.scene.standoff_m
        s_trk = float(hyp["hyp_s"][int(np.argmax(hyp["masses"]))])
        pos_map = np.asarray(world_to_map(tel.xy_w[i]), float)
        s_phys = float((pos_map - a) @ u)
        mark = a + (s_phys + (turn_s - s_trk)) * u
        art["turn_mark"].set_data([mark[0]], [mark[1]])
        art["turn_txt"].set_position((mark[0], mark[1] + 0.10))
        art["turn_txt"].set_text("planned turn")
    else:
        art["turn_mark"].set_data([], [])
        art["turn_txt"].set_text("")

    if len(tel.sigma):
        jj = max(st["j"], 0) + 1
        art["sigma"].set_data(tel.hyp_t[:jj], tel.sigma[:jj])
        art["sigma_now"].set_data([tel.hyp_t[jj - 1]], [tel.sigma[jj - 1]])
        art["axs"].set_xlim(0, max(float(t), 5.0))
    mode = st["mode"]
    art["mode_txt"].set_text(mode)
    art["mode_txt"].set_color(MODE_COLOR.get(mode, "#252525"))
    lines = []
    if d:
        lines.append(f"t={d['t']:6.1f}s  {'/'.join(d['allowed_decisions'])}"
                     f"  {d['reason'][:46]}")
    for p in presses[-2:]:
        lines.append(f"t={p['t']:6.1f}s  press {p['station']}"
                     f"{p['offset_m']:+.2f} -> {p['kind']}")
    art["ticker"].set_text("\n".join(lines))
    w0 = max(0, i - 1500)
    art["force"].set_data(tel.t[w0:i + 1], tel.force[w0:i + 1])
    art["force_now"].set_data([tel.t[i]], [tel.force[i]])
    art["axf"].set_xlim(tel.t[w0], max(tel.t[i], tel.t[w0] + 5.0))

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return buf.reshape(h, w, 4)[:, :, :3].copy()


def replay(episode_dir: Path, scene_name: str, out: Path, fps: float = 25.0,
           stride: int = 1) -> None:
    from .backends import BackendConfig
    from .mujoco_backend import MujocoBackend, MujocoConfig
    from .scenes import build_scene

    ep = json.loads((episode_dir / "episode.json").read_text())
    rec = np.load(episode_dir / "states.npz")
    ts, qs = rec["t"], rec["qpos"]
    scene = build_scene(scene_name)
    be = MujocoBackend(scene, BackendConfig(), np.random.default_rng(0),
                       MujocoConfig(seed=11 + int(ep.get("seed", 0))),
                       render=True, video_path=None)
    tel = Telemetry(ep, scene)

    enc = None
    fig = ax = art = None
    try:
        import torch
        for n in range(0, len(ts), stride):
            q = torch.from_numpy(qs[n]).to(be.device)
            be.env.sim.data.qpos[0] = q
            be.env.sim.forward()
            frame = be.env.render()
            rgb = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)[:, :, :3])
            if fig is None:
                h = rgb.shape[0]
                fig, ax, art = _panel_figure(scene, h, (int(h * 0.98) // 2) * 2)
            panel = _draw_panel(fig, ax, art, tel, be.world_point_to_map, float(ts[n]))
            if panel.shape[0] != rgb.shape[0]:
                pad = rgb.shape[0] - panel.shape[0]
                panel = np.pad(panel, ((0, pad), (0, 0), (0, 0)))[:rgb.shape[0]]
            composite = np.hstack([rgb, panel])
            # libx264/yuv420p demands even dimensions
            composite = composite[:composite.shape[0] // 2 * 2,
                                  :composite.shape[1] // 2 * 2]
            if enc is None:
                hh, ww = composite.shape[:2]
                enc = subprocess.Popen(
                    ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
                     "-pix_fmt", "rgb24", "-s", f"{ww}x{hh}",
                     "-r", str(fps / stride), "-i", "-", "-an",
                     "-c:v", "libx264", "-preset", "medium", "-crf", "21",
                     "-pix_fmt", "yuv420p", str(out)],
                    stdin=subprocess.PIPE)
            enc.stdin.write(composite.tobytes())
    finally:
        if enc is not None:
            enc.stdin.close()
            enc.wait()
        be.close()
    print(f"replayed {len(ts) // stride} frames -> {out}", flush=True)


def hstack(a: Path, b: Path, out: Path, labels=("belief_aware", "no_touch")) -> None:
    """Side-by-side comparison of two finished videos, shorter one end-padded."""
    def dur(p):
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration", "-of", "csv=p=0", str(p)],
                           capture_output=True, text=True)
        return float(r.stdout.strip())
    da, db = dur(a), dur(b)
    pad = abs(da - db) + 0.5
    fa = f"[0:v]tpad=stop_mode=clone:stop_duration={pad if da < db else 0:.2f}," \
         f"drawtext=text='{labels[0]}':x=10:y=10:fontsize=28:fontcolor=white:" \
         f"box=1:boxcolor=black@0.5[l]"
    fb = f"[1:v]tpad=stop_mode=clone:stop_duration={pad if db < da else 0:.2f}," \
         f"drawtext=text='{labels[1]}':x=10:y=10:fontsize=28:fontcolor=white:" \
         f"box=1:boxcolor=black@0.5[r]"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(a), "-i", str(b),
                    "-filter_complex", f"{fa};{fb};[l][r]hstack=inputs=2:shortest=0",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "22",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    print(f"side-by-side -> {out}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=Path)
    ap.add_argument("--scene")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--stride", type=int, default=1,
                    help="render every Nth recorded state (2 halves the runtime)")
    ap.add_argument("--hstack", nargs=2, type=Path,
                    help="compose two finished videos side by side instead")
    ap.add_argument("--labels", nargs=2, default=("belief_aware", "no_touch"))
    args = ap.parse_args(argv)
    if args.hstack:
        hstack(args.hstack[0], args.hstack[1], args.out, tuple(args.labels))
        return 0
    assert args.episode and args.scene, "--episode and --scene are required"
    replay(args.episode, args.scene, args.out, fps=args.fps, stride=args.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
