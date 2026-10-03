"""Offline replay for V2 in-room episodes: recorded states -> video.
Physics never re-runs; the render env is a MujocoBackendV2 (same injected
world), states are written and forwarded kinematically.

    PYTHONPATH=src:tactile_controller/src MUJOCO_GL=egl \
      uv run --project tactile_controller python -m lane_graph.replay_v2 \
        --episode out/tacnav_v2/mj_m2 --map m2_pillar_room \
        --out out/tacnav_v2/mj_m2/demo_v2mvp_full.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from execution.replay_video_v2 import (_draw_left, _draw_right,
                                       _left_panel, _right_panel)

from .scene2 import build_v2_scene


class TelemetryV2:
    """The V2-map telemetry: world->map from the spawn frame, probes located
    by recorded truth, no leg/turn-point machinery."""

    def __init__(self, ep: dict, scene):
        tr = ep["tick_trace"]
        self.t = np.asarray([r["t"] for r in tr])
        xy_w = np.asarray([r["xy"] for r in tr])
        base0 = xy_w[0].copy()
        yaw0 = float(tr[0]["yaw"])
        v = np.array([math.cos(yaw0), math.sin(yaw0)])
        u = np.array([v[1], -v[0]])
        self._B, self._u, self._v = base0, u, v
        self.path_map = np.c_[(xy_w - base0) @ u, (xy_w - base0) @ v]
        self.xy_w = xy_w
        self.yaw = np.asarray([r.get("yaw", 0.0) for r in tr])
        names = {"LOCOMOTION": "WALK", "STABILIZE": "STABILIZE",
                 "PROBE": "PROBE", "TURN": "TURN"}
        self.mode = [names.get(r["mode"], r["mode"]) for r in tr]
        self.hyp = ep["hypothesis_trace"]
        self.hyp_t = np.asarray([h["t"] for h in self.hyp])
        self.directives = ep.get("directives", [])
        self.dir_t = np.asarray([d["t"] for d in self.directives]) \
            if self.directives else np.zeros(0)
        self.scene, self.ep = scene, ep
        self.turn_pts, self.flips = [None] * len(self.hyp), []
        self.disambig = set()

        # smoothed viz trace: 10 Hz resample + 2.5 s moving average
        tu = np.arange(self.t[0], self.t[-1], 0.1)
        k, pad = 25, 12
        sm = []
        for c in range(2):
            r = np.interp(tu, self.t, self.path_map[:, c])
            sm.append(np.convolve(np.pad(r, pad, mode="edge"),
                                  np.ones(k) / k, "valid")[:len(tu)])
        self.smooth_t, self.smooth = tu, np.c_[sm[0], sm[1]]

        self.presses = []
        nv = ni = 0
        for p in ep["probes"]:
            valid = p["kind"] in ("contact", "swept_free")
            nv, ni = (nv + 1, ni) if valid else (nv, ni + 1)
            txy = np.asarray(p["truth_xy"], float)
            pm = np.array([(txy - base0) @ u, (txy - base0) @ v])
            fyaw = float(p.get("truth_yaw", yaw0)) - yaw0 + math.pi / 2.0
            fdir = np.array([math.cos(fyaw), math.sin(fyaw)])
            self.presses.append({
                "t": p["t"], "kind": p["kind"], "valid": valid,
                "num": nv if valid else ni,
                "force": p["peak_force_n"],
                "xy": pm + 0.30 * fdir,
                "facing": fdir, "travel": np.array([-fdir[1], fdir[0]])})
        self.n_valid, self.n_invalid = nv, ni
        for pr in self.presses:
            j0 = max(np.searchsorted(self.hyp_t, pr["t"] - 1e-9) - 1, 0)
            j1 = min(np.searchsorted(self.hyp_t, pr["t"] + 1e-9),
                     len(self.hyp) - 1)
            m0 = 1.0 - max(self.hyp[j0]["masses"] or [1.0])
            m1 = 1.0 - max(self.hyp[j1]["masses"] or [1.0])
            pr["contradicts"] = bool(pr["valid"] and (m0 - m1) >= 0.05)

        # ── belief vs truth, both ALREADY in the map frame ──────────────────
        #
        # `ExecutiveV3._snap` records `truth_xy` through `_truth_map()`, i.e.
        # already mapped -- unlike `tick_trace` and `probes`, which record WORLD
        # truth and therefore do need the transform above.  This block used to
        # apply world->map to the mapped sample a SECOND time and display the
        # x-component of the result as "believed vs TRUE (along-route)".
        #
        # Measured on the r1_gallery demo (scripts/audit_belief_viz.py): the
        # panel showed up to +12.10 m (median +8.38) on an episode whose belief
        # was never worse than 0.496 m and which ended `arrived` with
        # global_pos_err 0.098 m.  The double transform rotates the mapped
        # point about the spawn by (pi/2 - yaw0), so the artefact GROWS with
        # distance travelled and reads exactly like accumulating route drift.
        # It is replay-side bookkeeping only: nothing the executive decides
        # ever consulted it.
        et, ebel, etruth, emodes = [], [], [], []
        for h in self.hyp:
            if h.get("bel_xy") is None or h.get("truth_xy") is None:
                continue
            et.append(h["t"])
            ebel.append(np.asarray(h["bel_xy"], float))
            etruth.append(np.asarray(h["truth_xy"], float))
            emodes.append(self._modes(h))
        self.err_t = np.asarray(et)
        self.bel_map = np.asarray(ebel, float).reshape(-1, 2)
        self.truth_map = np.asarray(etruth, float).reshape(-1, 2)
        self.modes = emodes
        # the honest pose error, and the along-route component of the SAME
        # vector -- kept distinct because only the first is a pose error
        self.err_xy = (np.linalg.norm(self.bel_map - self.truth_map, axis=1)
                       if len(self.err_t) else np.zeros(0))
        self.err_v = ((self.bel_map[:, 0] - self.truth_map[:, 0])
                      if len(self.err_t) else np.zeros(0))

    @staticmethod
    def _modes(h) -> list[dict]:
        """Live PF modes as (xy, mass), heaviest first.

        `hyp_xy` is the full map-frame position of every mode; episodes
        recorded before it existed carry only `hyp_s` (the x component), and
        those fall back to the leading pose's y so the marker still lands on
        the right lane rather than at zero."""
        masses = list(h.get("masses") or [])
        xy = h.get("hyp_xy")
        xs = list(h.get("hyp_s") or [])
        y0 = float(np.asarray(h["bel_xy"], float)[1]) if h.get("bel_xy") else 0.0
        out = []
        for i, m in enumerate(masses):
            if xy and i < len(xy):
                q = np.asarray(xy[i], float)
            elif i < len(xs):
                q = np.array([float(xs[i]), y0])
            else:
                continue
            out.append({"xy": q, "mass": float(m)})
        out.sort(key=lambda d: -d["mass"])
        return out

    def at(self, t: float) -> dict:
        i = int(np.clip(np.searchsorted(self.t, t + 1e-9) - 1, 0,
                        len(self.t) - 1))
        j = int(np.clip(np.searchsorted(self.hyp_t, t + 1e-9) - 1, 0,
                        len(self.hyp) - 1))
        k = int(np.clip(np.searchsorted(self.dir_t, t + 1e-9) - 1, 0,
                        max(len(self.directives) - 1, 0))) \
            if len(self.dir_t) else -1
        mode = self.mode[i]
        return {"i": i, "j": j, "mode": mode, "hyp": self.hyp[j],
                "turn_pt": None,
                "directive": self.directives[k] if k >= 0 else None}


def render(episode_dir: Path, map_name: str, out: Path, stride: int = 1,
           fps: float = 25.0) -> None:
    import torch
    from execution.backends import BackendConfig
    from execution.mujoco_backend import MujocoConfig
    from .mj_backend2 import MujocoBackendV2

    ep = json.loads((episode_dir / "episode.json").read_text())
    rec = np.load(episode_dir / "states.npz")
    ts, qs = rec["t"], rec["qpos"]
    scene = build_v2_scene(map_name)
    be = MujocoBackendV2(scene, BackendConfig(), np.random.default_rng(0),
                         MujocoConfig(seed=11 + int(ep.get("seed", 0)),
                                      video_width=720, video_height=720,
                                      camera_distance=2.1,
                                      camera_elevation=-18.0),
                         render=True, video_path=None)
    tel = TelemetryV2(ep, scene)
    lfig, lart = _left_panel(scene, tel, 600, 720)
    rfig, rart = _right_panel(600, 720)
    cam = be.env._offline_renderer._cam \
        if hasattr(be.env, "_offline_renderer") else None
    az = None
    enc = None
    try:
        for n in range(0, len(ts), stride):
            t = float(ts[n])
            q = torch.from_numpy(qs[n]).to(be.device)
            be.env.sim.data.qpos[0] = q
            be.env.sim.forward()
            if cam is not None:
                i = tel.at(t)["i"]
                target = math.degrees(tel.yaw[i])
                az = target if az is None else az + float(
                    np.clip((target - az + 180.0) % 360.0 - 180.0, -4, 4))
                cam.azimuth = az
            rgb = np.ascontiguousarray(
                np.asarray(be.env.render(), np.uint8)[:, :, :3])
            left = _draw_left(lfig, lart, tel, t)
            right = _draw_right(rfig, rart, tel, t)
            hh = min(left.shape[0], rgb.shape[0], right.shape[0])
            comp = np.hstack([left[:hh], rgb[:hh], right[:hh]])
            comp = comp[:comp.shape[0] // 2 * 2, :comp.shape[1] // 2 * 2]
            if enc is None:
                H, W = comp.shape[:2]
                enc = subprocess.Popen(
                    ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
                     "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                     "-r", str(fps / stride), "-i", "-", "-an",
                     "-c:v", "libx264", "-preset", "medium", "-crf", "21",
                     "-pix_fmt", "yuv420p", str(out)],
                    stdin=subprocess.PIPE)
            enc.stdin.write(comp.tobytes())
    finally:
        if enc is not None:
            enc.stdin.close()
            enc.wait()
        be.close()
    print(f"v2mvp replayed {len(ts) // stride} frames -> {out}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args(argv)
    render(args.episode, args.map, args.out, stride=args.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
