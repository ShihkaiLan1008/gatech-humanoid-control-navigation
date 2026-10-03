#!/usr/bin/env python3
"""Replay dataset trajectories as video, with the belief beside the robot.

    scripts/render_dataset_trajectories.py --dataset ~/Desktop/room2_belief_dataset

Picks N deliberately DIFFERENT local segments -- contact-heavy, all-miss, the
corner vertex, a curved landmark, a belief that collapses -- and renders each
one as an MP4: MuJoCo on the left, the particle filter on the right.

This is a REPLAY, not a re-execution.  `states/*.npz` holds every world's full
generalized coordinates at 25 Hz, so the qpos is written straight into the
simulator and `sim.forward()` poses the model -- physics does not re-run, so
what you watch is the run that produced the data, frame for frame.

The map frame is rebuilt from the record rather than guessed: a segment's base
pose is stored in BOTH frames (`segment_meta.xy_map/yaw_map` and
`state_before.base_pose_world`), and those two determine `base0/u/v` exactly.
Guessing it instead is what put a demo replay 0.64 m and 173 deg off, with the
camera framing the far side of a wall.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

W_MJ, H_MJ = 720, 720
CAM_DIST_M = 2.1
CAM_ELEV_DEG = -18.0
CAM_AZ_OFFSET_DEG = 35.0     # off the shoulder, so the wall stays in frame


def _yaw_of(qpos) -> float:
    """World yaw from the base free joint's quaternion (w, x, y, z)."""
    w, x, y, z = (float(v) for v in qpos[3:7])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _pf_line(u) -> str:
    """The update summary on one line -- the raw dict overran the panel."""
    if not isinstance(u, dict):
        return str(u)
    keep = [("n_contacts", "hit"), ("n_free", "free"),
            ("n_outliers", "outlier"), ("resampled", "resampled"),
            ("ess_before", "ESS<"), ("ess_after", "ESS>")]
    bits = [f"{lab}={u[k]:g}" if isinstance(u.get(k), (int, float))
            else f"{lab}={u[k]}" for k, lab in keep if k in u]
    a = u.get("associations")
    if a:
        bits.append("assoc=" + ",".join(map(str, a))[:40])
    return "  ".join(bits)
PANEL_W = 860


def project_python() -> str:
    venv = ROOT / "tactile_controller" / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <dataset>/videos")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--segments", default=None,
                    help="comma-separated segment ids, instead of choosing")
    ap.add_argument("--stride", type=int, default=1,
                    help="render every Nth logged frame (logged at 25 Hz)")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap the touches rendered per segment")
    ap.add_argument("--table-only", action="store_true",
                    help="write the summary table, render nothing")
    ap.add_argument("--in-venv", action="store_true", help=argparse.SUPPRESS)
    return ap


# ── choosing five different trajectories ─────────────────────────────────────


def _profile(rows, vertices):
    """One segment's character, from the record alone."""
    import numpy as np
    rows = sorted(rows, key=lambda r: r["step"])
    m = rows[0]["segment_meta"]
    hits = [r for r in rows if r["observation"]["contacted"]]
    objs: dict = {}
    for r in hits:
        k = r["observation"]["contact_object"]
        objs[k] = objs.get(k, 0) + 1
    pts = [r["observation"]["contact_point_map"] for r in hits
           if r["observation"].get("contact_point_map")]
    dv = min((min(float(np.linalg.norm(np.asarray(p) - v)) for v in vertices)
              for p in pts), default=float("inf")) if vertices else float("inf")
    b0 = rows[0]["belief_before"]
    bn = rows[-1]["belief_after"] or b0
    t0 = b0["cov"][0][0] + b0["cov"][1][1]
    tn = bn["cov"][0][0] + bn["cov"][1][1]
    init = m.get("belief_init") or {}
    return {
        "segment": rows[0]["segment"], "split": rows[0]["split"],
        "env_id": rows[0].get("env_id"),
        "region": m["region"], "steps": len(rows),
        "contacts": len(hits), "no_contact": len(rows) - len(hits)
        - sum(1 for r in rows
              if r["observation"]["label"] == "invalid_attempt"),
        "invalid": sum(1 for r in rows
                       if r["observation"]["label"] == "invalid_attempt"),
        "base_xy_map": [round(float(v), 3) for v in m["xy_map"]],
        "base_yaw_deg": round(math.degrees(float(m["yaw_map"])), 1),
        "lateral_offset_m": m["lateral_offset_m"],
        "reach_standoff_m": m["reach_standoff_m"],
        "followed_body": m["followed_body"],
        "touched": objs,
        "closest_to_vertex_m": (round(dv, 3) if dv < float("inf") else None),
        "b0_sigma_xy_m": [b0["sigma_x_m"], b0["sigma_y_m"]],
        "b0_sigma_theta_deg": round(math.degrees(b0["sigma_theta_rad"]), 2),
        "b0_components": init.get("n_components", 1),
        "b0_source": (init.get("provenance") or {}).get("source"),
        "b0_offset_from_gt_m": init.get("offset_from_gt_m"),
        "pos_cov_trace_start_m2": round(t0, 5),
        "pos_cov_trace_end_m2": round(tn, 5),
        "trace_ratio": round(tn / max(t0, 1e-12), 4),
        "ess_frac_end": round(bn["ess"] / max(1, bn["n_particles"]), 3),
        "max_force_n": max((r["observation"]["max_force_n"] for r in rows),
                           default=0.0),
        "peak_true_force_n": max((r["observation"]["max_force_true_n"]
                                  for r in rows), default=0.0),
    }


def choose(profiles, n: int) -> list:
    """Five that are actually different, each for a stated reason."""
    picked, why = [], {}

    def take(cands, reason, key):
        for c in sorted(cands, key=key):
            if c["segment"] not in {p["segment"] for p in picked}:
                picked.append(c)
                why[c["segment"]] = reason
                return

    full = [p for p in profiles if p["steps"] >= 10]
    take([p for p in full if p["region"] == "vertex"
          and p["closest_to_vertex_m"] is not None],
         "the corner VERTEX itself -- the closest contact to it",
         lambda p: p["closest_to_vertex_m"])
    take([p for p in full if p["contacts"] == 0],
         "every touch a valid NO-CONTACT: the base stands out of reach",
         lambda p: -p["steps"])
    take([p for p in full if p["trace_ratio"] < 0.2 and p["contacts"] >= 5],
         "the belief COLLAPSES: touches resolve a wide prior",
         lambda p: p["trace_ratio"])
    take([p for p in full if "bulge_0" in p["touched"]],
         "a CURVED landmark (the goal pillar), not a flat wall",
         lambda p: -p["touched"].get("bulge_0", 0))
    take([p for p in full if p["b0_components"] > 1],
         "a MULTIMODAL b_0: the filter is several places at once",
         lambda p: -p["b0_components"])
    # fill with whatever is most unlike what is already picked
    while len(picked) < n and full:
        seen_r = {p["region"] for p in picked}
        rest = [p for p in full
                if p["segment"] not in {q["segment"] for q in picked}]
        if not rest:
            break
        rest.sort(key=lambda p: (p["region"] in seen_r,
                                 -abs(p["contacts"] - p["no_contact"])))
        picked.append(rest[0])
        why[rest[0]["segment"]] = "a further contrast in region and outcome mix"
    for p in picked:
        p["chosen_because"] = why.get(p["segment"], "")
    return picked[:n]


# ── the belief panel ─────────────────────────────────────────────────────────


def panel_figure(manifest, prof):
    """A figure reused frame to frame: the room, the route, and empty artists."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Polygon

    dpi = 100
    fig = plt.figure(figsize=(PANEL_W / dpi, H_MJ / dpi), dpi=dpi)
    ax = fig.add_axes([0.09, 0.32, 0.88, 0.60])
    for b in manifest["map"]["bodies"]:
        ax.add_patch(Polygon(np.asarray(b["verts"], float), closed=True,
                             facecolor="#d9d2c6", edgecolor="#8a8378", lw=1.0,
                             zorder=1))
    route = np.asarray(manifest["route"]["polyline_map"], float)
    ax.plot(route[:, 0], route[:, 1], "-", color="#9aa7b1", lw=1.6, zorder=2)
    art = {}
    art["particles"] = ax.scatter([], [], s=3.5, c="#1f77b4", alpha=0.45,
                                  zorder=5, label="particles $b_t$")
    art["mean"], = ax.plot([], [], "o", ms=7, mfc="none", mec="#1f77b4",
                           mew=2.0, zorder=7, label="belief mean")
    art["gt"], = ax.plot([], [], "k+", ms=13, mew=2.0, zorder=8,
                         label="ground truth")
    art["hits"], = ax.plot([], [], "*", color="#d62728", ms=9, zorder=9,
                           label="contacts so far")
    art["miss"], = ax.plot([], [], ".", color="#2ca02c", ms=4, alpha=0.6,
                           zorder=6, label="no-contact targets")
    art["tgt"], = ax.plot([], [], "x", color="#ff7f0e", ms=10, mew=2.0,
                          zorder=10, label="current target")
    lo = route.min(0) - 1.1
    hi = route.max(0) + 1.1
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_xlabel("x [m, scene frame]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_title(f"segment {prof['segment']}  ({prof['region']}, "
                 f"{prof['split']})\n{prof['chosen_because']}", fontsize=9)

    tx = fig.add_axes([0.06, 0.02, 0.92, 0.24])
    tx.axis("off")
    art["text"] = tx.text(0.0, 1.0, "", va="top", ha="left", fontsize=7.6,
                          family="monospace")
    art["ax"] = ax
    return fig, art


def panel_rgb(fig, art, *, particles, mean, gt, hits, miss, target, caption):
    import numpy as np
    art["particles"].set_offsets(particles if len(particles)
                                 else np.zeros((0, 2)))
    art["mean"].set_data([mean[0]], [mean[1]])
    art["gt"].set_data([gt[0]], [gt[1]])
    art["hits"].set_data(*(np.asarray(hits).T if len(hits) else ([], [])))
    art["miss"].set_data(*(np.asarray(miss).T if len(miss) else ([], [])))
    if target is None:
        art["tgt"].set_data([], [])
    else:
        art["tgt"].set_data([target[0]], [target[1]])
    art["text"].set_text(caption)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    return np.ascontiguousarray(buf)


# ── one segment ──────────────────────────────────────────────────────────────


def render_segment(dataset: Path, manifest, rows, prof, out_mp4: Path, *,
                   stride: int, fps: float, max_steps=None) -> bool:
    import numpy as np
    import torch

    from data_collect.records import load_particles, load_states
    from data_collect.rooms import build_scene
    from execution.backends import BackendConfig
    from execution.vforce_backend import VForceBackend, VForceConfig
    from global_nav.geometry import wrap_angle

    rows = sorted(rows, key=lambda r: r["step"])
    if max_steps:
        rows = rows[:max_steps]
    if any(r.get("state_log") is None for r in rows):
        print(f"[video] segment {prof['segment']}: no state log -- skipped",
              flush=True)
        return False

    m = rows[0]["segment_meta"]
    xy_map = np.asarray(m["xy_map"], float)
    yaw_map = float(m["yaw_map"])
    xy_w = np.asarray(rows[0]["state_before"]["base_pose_world"][:2], float)
    yaw_w = float(rows[0]["state_before"]["base_pose_world"][2])
    # the frame this segment's room was installed in, solved exactly from the
    # pose recorded in BOTH frames
    yaw0 = float(wrap_angle(yaw_w - (yaw_map - math.pi / 2.0)))
    v = np.array([math.cos(yaw0), math.sin(yaw0)])
    u = np.array([v[1], -v[0]])
    base0 = xy_w - (xy_map[0] * u + xy_map[1] * v)

    scene = build_scene(manifest["map"]["room"], manifest["map"]["reverse"])
    mj = VForceConfig(seed=11, episode_budget_s=1.0e5,
                      video_width=W_MJ, video_height=H_MJ,
                      camera_distance=CAM_DIST_M,
                      camera_elevation=CAM_ELEV_DEG)
    be = VForceBackend(scene, BackendConfig(), np.random.default_rng(0), mj,
                       render=True, video_path=None, view=False)
    be.base0, be.u, be.v, be.yaw0 = base0, u, v, yaw0
    be._install_world()

    # The backend already aims the camera at the robot; `replay_video_v2` shows
    # the only thing worth steering per frame is the azimuth, so that the
    # camera looks over the robot's shoulder at the wall it is reaching for.
    # Setting `cam.type`/`trackbodyid` by hand instead put the camera inside a
    # wall -- the ids are per-robot, not mjModel body ids.
    cam = getattr(getattr(be.env, "_offline_renderer", None), "_cam", None)
    az = None

    fig, art = panel_figure(manifest, prof)
    hits, miss = [], []
    enc = None
    try:
        for r in rows:
            sl = r["state_log"]
            qs, ts = load_states(dataset, sl["from"], sl["to"], sl["env_id"])
            if not len(qs):
                continue
            b0 = r["belief_before"]
            pts = load_particles(dataset, b0["particles_row"])
            o = r["observation"]
            tgt = r["action"]["target_map_xy"]
            gt = r["state_before"]["base_pose_map"]
            cap = (
                f"step {r['step']:2d}/{len(rows) - 1}   {o['label']:<12s}"
                f" term={o['termination_reason']}\n"
                f"action   target_body=({r['action']['target_body'][0]:.3f},"
                f"{r['action']['target_body'][1]:+.3f},"
                f"{r['action']['target_body'][2]:.3f})  hand={r['action']['hand']}"
                f"  bin={r['action']['action_bin']}\n"
                f"observed force max {o['max_force_n']:.1f} N sensed /"
                f" {o['max_force_true_n']:.1f} N true"
                + (f"   object={o['contact_object']}"
                   f"   t2c={o['time_to_contact_s']} s"
                   if o["contacted"] else "   (bounded free sweep)") + "\n"
                f"belief   trace {b0['cov'][0][0] + b0['cov'][1][1]:.5f} m2"
                f"   ESS {b0['ess']:.0f}/{b0['n_particles']}"
                f"   sigma ({b0['sigma_x_m']:.3f}, {b0['sigma_y_m']:.3f}) m,"
                f" {math.degrees(b0['sigma_theta_rad']):.1f} deg\n"
                f"PF       {_pf_line(o.get('pf_update'))}")
            right = panel_rgb(fig, art, particles=pts[:, :2],
                              mean=b0["mean"], gt=gt, hits=hits, miss=miss,
                              target=tgt, caption=cap)
            for n in range(0, len(qs), stride):
                q = np.asarray(qs[n], dtype=np.float32)
                be.env.sim.data.qpos[0] = torch.from_numpy(q).to(be.device)
                be.env.sim.forward()
                if cam is not None:
                    target = math.degrees(_yaw_of(q)) + CAM_AZ_OFFSET_DEG
                    if az is None:
                        az = target
                    err = (target - az + 180.0) % 360.0 - 180.0
                    az += float(np.clip(err, -4.0, 4.0))
                    cam.azimuth = az
                left = np.ascontiguousarray(
                    np.asarray(be.env.render(), dtype=np.uint8)[:, :, :3])
                comp = np.hstack([left, right])
                if enc is None:
                    h, w = comp.shape[:2]
                    out_mp4.parent.mkdir(parents=True, exist_ok=True)
                    enc = subprocess.Popen(
                        ["ffmpeg", "-loglevel", "error", "-y", "-f",
                         "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                         "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                         "-preset", "medium", "-crf", "21", "-pix_fmt",
                         "yuv420p", str(out_mp4)], stdin=subprocess.PIPE)
                enc.stdin.write(comp.tobytes())
            if o["contacted"] and o.get("contact_point_map"):
                hits.append(o["contact_point_map"])
            elif o["label"] == "no_contact":
                miss.append(tgt)
    finally:
        if enc is not None:
            enc.stdin.close()
            enc.wait()
        import matplotlib.pyplot as plt
        plt.close(fig)
        be.close()
    return out_mp4.exists()


# ── the summary table ────────────────────────────────────────────────────────

TABLE_COLS = [
    ("segment", "seg"), ("region", "region"), ("split", "split"),
    ("steps", "steps"), ("contacts", "hit"), ("no_contact", "miss"),
    ("invalid", "inv"), ("base_xy_map", "base xy [m]"),
    ("base_yaw_deg", "yaw"), ("lateral_offset_m", "d_perp"),
    ("reach_standoff_m", "reach"), ("followed_body", "face"),
    ("touched", "touched"), ("closest_to_vertex_m", "to vertex"),
    ("b0_components", "b0 modes"), ("b0_source", "b0 src"),
    ("b0_sigma_xy_m", "b0 sigma xy"), ("b0_sigma_theta_deg", "b0 sig_th"),
    ("b0_offset_from_gt_m", "mu0 to GT"),
    ("pos_cov_trace_start_m2", "trace in"),
    ("pos_cov_trace_end_m2", "trace out"), ("trace_ratio", "ratio"),
    ("ess_frac_end", "ESS/N"), ("peak_true_force_n", "peak true N"),
]


def write_table(picked, out_dir: Path) -> Path:
    rows = [f"| {' | '.join(h for _, h in TABLE_COLS)} |",
            f"|{'|'.join('---' for _ in TABLE_COLS)}|"]
    for p in picked:
        cells = []
        for k, _h in TABLE_COLS:
            v = p.get(k)
            if isinstance(v, dict):
                v = ", ".join(f"{a}x{b}" for a, b in v.items()) or "-"
            elif isinstance(v, list):
                v = "(" + ", ".join(f"{x:g}" for x in v) + ")"
            cells.append("-" if v is None else str(v))
        rows.append(f"| {' | '.join(cells)} |")
    body = ["# Five dataset trajectories", "",
            "Each row is one **local segment**: one base pose, one particle "
            "filter, 10-20 touches folded into it one at a time.  The videos "
            "beside this file are REPLAYS of the recorded `qpos` -- physics "
            "does not re-run.", ""]
    body += rows
    body += ["", "## Why these five", ""]
    for p in picked:
        body.append(f"* **segment {p['segment']}** ({p['region']}, "
                    f"{p['split']}) -- {p['chosen_because']}")
    body += ["", "## Columns", "",
             "| column | meaning |", "|---|---|",
             "| `d_perp` | lateral offset from the reference centreline [m] |",
             "| `reach` | distance a reach must cover, cast along the pose's "
             "own facing direction [m] |",
             "| `to vertex` | closest contact to the room's corner vertex [m]"
             " |",
             "| `b0 modes` | components in the initial belief; >1 is a "
             "multimodal prior taken from the logs |",
             "| `b0 src` | whether that covariance came from a logged particle"
             " set or from a run record's 2x2 block |",
             "| `trace in/out` | position-covariance trace at the segment's "
             "first and last belief [m^2] |",
             "| `ESS/N` | effective sample size over particle count at the "
             "end |",
             "| `peak true N` | true physical peak force; the sensed channel "
             "saturates at 20 N |"]
    p = out_dir / "TRAJECTORIES.md"
    p.write_text("\n".join(body) + "\n")
    (out_dir / "trajectories.json").write_text(json.dumps(picked, indent=1))
    return p


def run(a) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from data_collect.diagnostics import load

    out_dir = a.out or (a.dataset / "videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, rows = load(a.dataset)
    by_seg: dict = {}
    for r in rows:
        by_seg.setdefault(r["segment"], []).append(r)
    verts = [v["xy_map"] for v in
             (manifest.get("route", {}) or {}).get("corner_vertices", [])]
    profiles = [_profile(v, verts) for v in by_seg.values()]

    if a.segments:
        want = [int(x) for x in a.segments.split(",")]
        picked = [p for s in want for p in profiles if p["segment"] == s]
        for p in picked:
            p.setdefault("chosen_because", "requested explicitly")
    else:
        picked = choose(profiles, a.n)

    print(f"[video] {len(rows)} transitions, {len(by_seg)} segments; chose "
          + ", ".join(str(p["segment"]) for p in picked), flush=True)
    table = write_table(picked, out_dir)
    print(f"[video] wrote {table}", flush=True)
    if a.table_only:
        return 0

    for p in picked:
        mp4 = out_dir / f"segment_{p['segment']:05d}_{p['region']}.mp4"
        print(f"[video] rendering segment {p['segment']} "
              f"({p['region']}, {p['steps']} steps) -> {mp4.name}", flush=True)
        try:
            ok = render_segment(a.dataset, manifest, by_seg[p["segment"]], p,
                                mp4, stride=a.stride, fps=a.fps,
                                max_steps=a.max_steps)
        except Exception as exc:                # noqa: BLE001 - keep going
            import traceback
            traceback.print_exc()
            print(f"[video] segment {p['segment']} failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            ok = False
        print(f"[video] {'wrote' if ok else 'FAILED'} {mp4.name}"
              + (f"  ({mp4.stat().st_size / 1e6:.1f} MB)" if ok else ""),
              flush=True)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    if not a.in_venv:
        cmd = [project_python(), "-u", str(Path(__file__).resolve()),
               "--in-venv", *argv]
        env = dict(os.environ)
        env["PYTHONPATH"] = "src:tactile_controller/src"
        env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
        print("+", " ".join(cmd), flush=True)
        return subprocess.Popen(cmd, cwd=str(ROOT), env=env).wait()
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
