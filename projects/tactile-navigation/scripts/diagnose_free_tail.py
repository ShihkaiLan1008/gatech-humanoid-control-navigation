"""Subtask 3 -- what a swept-free ray actually does to the particle cloud.

    PYTHONPATH=src python3 scripts/diagnose_free_tail.py --scene r1_gallery --seeds 8

`skip_uninformative_probes` declines presses where no live hypothesis puts a
surface within reach, on the argument (PF_AUDIT.md 7) that the resulting free
ray penetrates no hypothesised wall for any pose and is therefore exactly flat.
The 5-map benchmark then showed it changing the belief error on 100 of 100
episodes, and *hurting* r1_gallery.  Something that is nominally flat is not.

The candidate mechanism is TAIL PRUNING.  "Flat across the live hypotheses" is a
statement about a handful of cluster CENTRES.  `swept_free_loglik` is applied to
every PARTICLE with no outlier gate -- unlike `contact_loglik`, which refuses an
update no belief mass explains -- so particles in the tail whose own map does
intersect the swept volume are penalised up to `free_cap_nats` (4 nats, a factor
of 55 in weight) even when every mode agrees.  If the truth happens to sit in
that tail, the ray quietly removes the support the filter would have needed.

Records per free observation, on the live cloud, before the update is applied:

    per-particle free loglik      spread, min, fraction pinned at the cap
    per-mode loglik               the same quantity at each cluster centre
    truth-mode mass               weight within `mode_radius_m` of the truth
    truth-nearest survival        distance truth -> nearest particle, and the
                                  weight that particle carries, before/after
    ESS / resampling
    cloud spread                  before/after
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from execution.backends import BackendConfig, KinematicBackend    # noqa: E402
from global_nav import metrics                                    # noqa: E402
from global_nav.models import swept_free_loglik                   # noqa: E402
from lane_graph.executive3 import ExecutiveV3, ExecV3Config       # noqa: E402
from lane_graph.graph3 import build_graph3                        # noqa: E402
from lane_graph.run_v3 import PRESS_TARGET_N                      # noqa: E402
from lane_graph.scene3 import build_v3_scene                      # noqa: E402


def run(scene, graph, seed: int, **kw):
    """One episode, with every free update dissected."""
    be = KinematicBackend(scene, BackendConfig(probe_force_n=PRESS_TARGET_N),
                          np.random.default_rng(seed))
    ex = ExecutiveV3(scene, be, graph, ExecV3Config(seed=seed, **kw))
    b = ex.mb.b
    events, orig = [], b.update

    def traced(contacts, frees):
        pre = None
        if frees:
            truth = np.asarray(be.truth_pose()[0], float)
            if hasattr(be, "world_point_to_map"):
                truth = np.asarray(be.world_point_to_map(truth), float)
            w = b.weights.copy()
            d = np.linalg.norm(b.xy - truth[None, :], axis=1)
            k = int(np.argmin(d))
            lls = [swept_free_loglik(b.plan, b.xy, b.theta, b.reg, o,
                                     b.cfg.observation) for o in frees]
            ll = np.sum(lls, axis=0)
            modes = b.modes()
            pre = {
                "n_free": len(frees),
                "ll_min": float(ll.min()), "ll_max": float(ll.max()),
                "ll_spread": float(ll.max() - ll.min()),
                "frac_at_cap": float(np.mean(ll <= -b.cfg.observation.free_cap_nats
                                             + 1e-9)),
                "frac_penalised": float(np.mean(ll < -1e-9)),
                "w_ll_mean": float(np.sum(w * ll)),
                "mode_ll": [float(np.sum(ll[np.linalg.norm(
                    b.xy - np.asarray(m["xy"], float)[None, :], axis=1)
                    <= b.cfg.mode_radius_m]
                    * w[np.linalg.norm(b.xy - np.asarray(m["xy"], float)[None, :],
                                       axis=1) <= b.cfg.mode_radius_m])
                    / max(m["weight"], 1e-12)) for m in modes],
                "mode_mass": [float(m["weight"]) for m in modes],
                "truth_nn_d": float(d[k]),
                "truth_nn_ll": float(ll[k]),
                "truth_nn_w": float(w[k]),
                "truth_mass": float(w[d <= b.cfg.mode_radius_m].sum()),
                "ess": float(metrics.effective_sample_size(w)),
                "spread": float(np.sqrt(np.sum(w * np.sum(
                    (b.xy - (w @ b.xy)[None, :]) ** 2, axis=1)))),
                "t": float(be.time_s),
            }
        rep = orig(contacts, frees)
        if pre is not None:
            w2 = b.weights
            d2 = np.linalg.norm(b.xy - np.asarray(
                [pre["truth_x"], pre["truth_y"]] if "truth_x" in pre else [0, 0]
            )[None, :], axis=1) if False else None
            truth = np.asarray(be.truth_pose()[0], float)
            if hasattr(be, "world_point_to_map"):
                truth = np.asarray(be.world_point_to_map(truth), float)
            d2 = np.linalg.norm(b.xy - truth[None, :], axis=1)
            pre.update({
                "truth_mass_after": float(w2[d2 <= b.cfg.mode_radius_m].sum()),
                "truth_nn_d_after": float(d2.min()),
                "ess_after": float(metrics.effective_sample_size(w2)),
                "resampled": bool(rep.get("resampled", False)),
                "spread_after": float(np.sqrt(np.sum(w2 * np.sum(
                    (b.xy - (w2 @ b.xy)[None, :]) ** 2, axis=1)))),
            })
            events.append(pre)
        return rep

    b.update = traced
    try:
        res = ex.run()
    finally:
        be.close()
    return res, events


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="r1_gallery")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("out/free_tail"))
    a = ap.parse_args(argv)

    scene = build_v3_scene(a.scene)
    graph = build_graph3(scene)
    allev, rows = [], []
    for seed in range(a.seeds):
        res, ev = run(scene, graph, seed, skip_uninformative_probes=False)
        for e in ev:
            e["seed"] = seed
        allev.extend(ev)
        rows.append({"seed": seed, "truly": bool(res.get("arrived_truly")),
                     "err": float(res["global_pos_err_m"]),
                     "n_free": sum(1 for p in res["probes"]
                                   if p["kind"] == "swept_free")})

    print(f"# swept-free tail audit -- {a.scene}, {a.seeds} seeds, skip OFF")
    print(f"{len(allev)} free updates\n")
    print(f"{'seed':>4s} {'t':>7s} {'nf':>3s} {'ll_spread':>9s} {'%pen':>6s}"
          f" {'%cap':>6s} {'mode_ll':>22s} {'truth_m':>8s} {'->after':>8s}"
          f" {'nn_d':>6s} {'nn_ll':>7s} {'ESS':>6s} {'->':>6s} {'rs':>3s}")
    for e in allev:
        ml = ", ".join(f"{x:+.2f}" for x in e["mode_ll"][:3])
        print(f"{e['seed']:4d} {e['t']:7.1f} {e['n_free']:3d} {e['ll_spread']:9.3f}"
              f" {100 * e['frac_penalised']:6.1f} {100 * e['frac_at_cap']:6.1f}"
              f" {ml:>22s} {e['truth_mass']:8.3f} {e['truth_mass_after']:8.3f}"
              f" {e['truth_nn_d']:6.3f} {e['truth_nn_ll']:+7.3f}"
              f" {e['ess']:6.0f} {e['ess_after']:6.0f}"
              f" {'Y' if e['resampled'] else '-':>3s}")

    ev = allev
    if ev:
        dm = np.array([e["truth_mass_after"] - e["truth_mass"] for e in ev])
        sp = np.array([e["ll_spread"] for e in ev])
        pen = np.array([e["frac_penalised"] for e in ev])
        nnll = np.array([e["truth_nn_ll"] for e in ev])
        print("\nSUMMARY")
        print(f"  per-particle free loglik spread : median {np.median(sp):.3f} nats,"
              f" max {sp.max():.3f}")
        print(f"  particles penalised at all      : median {100 * np.median(pen):.1f}%")
        print(f"  loglik at the TRUTH-nearest particle: median {np.median(nnll):+.3f}"
              f" nats, worst {nnll.min():+.3f}")
        print(f"  truth-mode mass change          : median {np.median(dm):+.4f},"
              f" mean {dm.mean():+.4f}")
        print(f"  updates that REDUCED truth mass : "
              f"{int(np.sum(dm < -1e-6))} of {len(dm)}")
        flat = [e for e in ev if e["ll_spread"] < 1e-6]
        print(f"  updates that were TRULY flat (spread ~ 0): {len(flat)} of {len(ev)}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.scene}_free_tail.json").write_text(
        json.dumps({"events": allev, "episodes": rows}, indent=1))
    print(f"\nwrote {a.out}/{a.scene}_free_tail.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
