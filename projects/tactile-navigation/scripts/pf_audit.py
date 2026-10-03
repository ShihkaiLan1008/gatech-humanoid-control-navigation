"""Subtask 1 -- run an episode with the particle filter instrumented, and attribute
every large belief error to a cause.

    PYTHONPATH=src python3 scripts/pf_audit.py --backend kinematic \
        --scene ambiguous_near_edge --seed 0 --out out/pf_audit/ane_s0

Writes ``pf_updates.json`` (one record per tactile update, see pf_diagnostics.py)
and ``pf_audit.md`` (the per-update table and the episode verdict).

The audit is only meaningful because truth is fed to the RECORDER and to nothing
else: ``--assert-no-truth-leak`` re-runs the same seed with the executive's truth
sampler poisoned, and requires the planner's decisions to come out identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from execution.backends import BackendConfig, KinematicBackend          # noqa: E402
from execution.executive import ExecConfig, Executive                   # noqa: E402
from execution.scenes import build_scene                                # noqa: E402
from global_nav.pf_diagnostics import PFDiagnostics                     # noqa: E402


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def _install_recorder(mb, be, leg_fn):
    """Wire truth into the RECORDER from outside the pipeline.

    The executive and the belief are unmodified: the firewall test forbidding
    `truth_pose` in the belief-update path still passes and still means what it
    says.  A recorder the pipeline had to feed would make that test
    unenforceable.

    The recorder scores a POSE, so BOTH of its components must be carried into
    the belief's frame.  Mapping the position and leaving the yaw in world was
    the defect behind PF_AUDIT.md 4 -- see ``world_yaw_to_map``.
    """
    def _truth():
        xy, yaw = be.truth_pose()
        if hasattr(be, "world_point_to_map"):
            xy = be.world_point_to_map(np.asarray(xy, float))
        yaw = be.world_yaw_to_map(float(yaw))
        return np.asarray(xy, float), float(yaw)

    diag = PFDiagnostics(truth_fn=_truth,
                         context_fn=lambda: {"t_s": be.time_s, "leg": leg_fn()})
    mb.b.diagnostics = diag
    return diag


def run_v3_episode(map_name: str, seed: int, backend: str, particles: int):
    """The V3 lane-graph executive -- the one that produced the R2/R3 demos."""
    from lane_graph.executive3 import ExecutiveV3, ExecV3Config
    from lane_graph.graph3 import build_graph3
    from lane_graph.run_v3 import PRESS_TARGET_N, _v3_backend
    from lane_graph.scene3 import build_v3_scene

    scene = build_v3_scene(map_name)
    graph = build_graph3(scene)
    rng = np.random.default_rng(seed)
    be = (KinematicBackend(scene, BackendConfig(probe_force_n=PRESS_TARGET_N), rng)
          if backend == "kinematic" else _v3_backend(scene, rng, seed))
    try:
        ex = ExecutiveV3(scene, be, graph, ExecV3Config(seed=seed, n_particles=particles))
        diag = _install_recorder(ex.mb, be, lambda: 0)
        res = ex.run()
    finally:
        be.close()
    return res, diag


def run_episode(scene_name: str, seed: int, policy: str, backend: str,
                touch_step_follow: bool, particles: int):
    """One episode with diagnostics attached.  Returns (result, diagnostics)."""
    scene = build_scene(scene_name)
    cfg = ExecConfig(policy=policy, seed=seed, n_particles=particles,
                     touch_step_follow=touch_step_follow)
    if backend == "kinematic":
        be = KinematicBackend(scene, BackendConfig(), np.random.default_rng(seed))
    else:
        from execution.mujoco_backend import MujocoBackend, MujocoConfig
        be = MujocoBackend(scene, BackendConfig(), np.random.default_rng(seed),
                           MujocoConfig(seed=seed), render=False, video_path=None)
    ex = Executive(scene, be, cfg)
    diag = _install_recorder(ex.mb, be, lambda: ex.leg)
    try:
        res = ex.run()
    finally:
        if hasattr(be, "close"):
            be.close()
    return res, diag


def write_report(out: Path, res: dict, diag: PFDiagnostics, title: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "pf_updates.json").write_text(json.dumps(_jsonable(diag.to_json()), indent=2))
    (out / "episode.json").write_text(json.dumps(_jsonable(res), indent=2))
    s = diag.summary()
    (out / "pf_summary.json").write_text(json.dumps(_jsonable(s), indent=2))

    L = [f"# PF audit -- {title}", ""]
    L += [f"- outcome **{res.get('outcome')}**, wrong_turn `{res.get('wrong_turn')}`,"
          f" final belief error **{res.get('global_pos_err_m', float('nan')):.3f} m**",
          f"- {s['n_updates']} tactile updates, {s['n_resampled']} triggered a resample,"
          f" min ESS {s.get('min_ess_after', float('nan')):.0f}",
          f"- truth outside particle support on **{s['n_truth_outside_support']}**"
          f" of {s['n_updates']} updates",
          f"- updates that made the error WORSE: **{s['n_updates_that_worsened_error']}**",
          f"- median (truth - leading) log-likelihood margin:"
          f" **{s['median_ll_margin_truth_minus_leading']}** nats", ""]
    L += ["## Attribution", "", "| verdict | n |", "|---|---|"]
    for k, v in sorted(s["verdicts"].items(), key=lambda kv: -kv[1]):
        L.append(f"| `{k}` | {v} |")
    L += ["", "## Per-update", "",
          "| # | t (s) | station | obs | err before | err after | ESS after | resamp |"
          " truth-nn | ll truth | ll leading | margin | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in diag.records:
        L.append(
            f"| {r.index} | {r.t_s:.1f} | {r.station or '-'} |"
            f" {r.n_contacts}c/{r.n_free}f/{r.n_outliers}x |"
            f" {r.pos_err_before_m:.3f} | {r.pos_err_after_m:.3f} |"
            f" {r.ess_after:.0f} | {'Y' if r.resampled else '-'} |"
            f" {r.truth_nn_dist_before_m:.3f} |"
            f" {r.ll_truth.get('total', float('nan')):.2f} |"
            f" {r.ll_leading.get('total', float('nan')):.2f} |"
            f" {r.ll_margin_truth_minus_leading:+.2f} | `{r.verdict()}` |")
    (out / "pf_audit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:14]))
    print(f"\nwrote {out}/pf_audit.md, pf_updates.json, pf_summary.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="ambiguous_near_edge")
    ap.add_argument("--policy", default="belief_aware")
    ap.add_argument("--backend", default="kinematic", choices=("kinematic", "mujoco"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--particles", type=int, default=900)
    ap.add_argument("--touch-step-follow", action="store_true")
    ap.add_argument("--stack", default="batch3a", choices=("batch3a", "v3"),
                    help="batch3a = station executive; v3 = lane-graph touch-follow")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    if a.stack == "v3":
        res, diag = run_v3_episode(a.scene, a.seed, a.backend, a.particles)
        title = f"{a.scene} / V3 touch-follow / {a.backend} / s{a.seed}"
    else:
        res, diag = run_episode(a.scene, a.seed, a.policy, a.backend,
                                a.touch_step_follow, a.particles)
        title = f"{a.scene} / {a.policy} / {a.backend} / s{a.seed}"
    write_report(a.out, res, diag, title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
