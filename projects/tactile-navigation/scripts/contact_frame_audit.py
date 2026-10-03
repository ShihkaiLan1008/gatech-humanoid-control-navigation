"""Audit the whole chain that turns a physical touch into a particle weight.

    PYTHONPATH=src python3 scripts/contact_frame_audit.py out/pf_audit/v3_r3_mj_fixed

`scripts/contact_frame_check.py` asks the same questions of a LIVE press and
therefore needs MuJoCo, a GPU and four minutes of start-up.  This one asks them
of a recorded episode, so the answer costs nothing and can be re-derived for
every audit that has ever been written.

The chain, and what each stage can get wrong:

    wrist link, world          `_probe_ee_w`      is the wrist origin the point
                                                  that TOUCHED?  (FACE_OFFSET
                                                  is 0.0, so the kiss model
                                                  says yes; a capsule says no)
      -> body frame            `_contact_body_frame`   base pose and yaw
      -> map frame             `se2_apply` in the PF   THE frame bug: fixed in
                                                       `world_yaw_to_map`
      -> nearest map surface   `nearest_surface`       registration / geometry

Reports, per contact: the reach |p_body|, the residual to the nearest mapped
surface, and the angle between the normal the observation asserts and the
normal the map has there.  A residual at the truth pose comparable to the
residual at the filter's own leading particle is the pass condition -- it means
the likelihood is not lying about where the robot is.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from global_nav.geometry import se2_apply, se2_rotate, wrap_angle   # noqa: E402
from global_nav.models import nearest_surface                      # noqa: E402

REACH_MAX_M = 0.45      # measured hand_force_v2 reach depth
N_BODY = (-1.0, 0.0)    # `_contact_body_frame` asserts this on every contact


def _scene_plan(out: Path, scene: str | None):
    """The Floorplan the belief scored against."""
    name = scene
    if name is None:
        md = (out / "pf_audit.md")
        if md.exists():
            head = md.read_text().splitlines()[0]
            name = head.split("--")[-1].split("/")[0].strip()
    from lane_graph.scene3 import build_v3_scene
    return build_v3_scene(name).plan, name


def _world_map_rotation(out: Path, records) -> float | None:
    """Recover pi/2 - yaw0 from the episode log, when it is there.

    `episode.json` logs each probe's truth pose in WORLD; `pf_updates.json` logs
    the same pose in MAP.  Two frames related by a rigid transform, sampled at
    the same instants, determine that transform exactly -- which is how the yaw
    bug was pinned in the first place, and what lets this script show the size
    of the error the fix removed.
    """
    ep = out / "episode.json"
    if not ep.exists():
        return None
    probes = {round(float(p["t"]), 2): p for p in
              json.loads(ep.read_text()).get("probes", [])}
    W, M = [], []
    for r in records:
        p = probes.get(round(float(r["t_s"]), 2))
        if p is not None and r.get("truth_xy"):
            W.append(p["truth_xy"])
            M.append(r["truth_xy"])
    if len(W) < 2:
        return None
    W, M = np.asarray(W, float), np.asarray(M, float)
    H = (W - W.mean(0)).T @ (M - M.mean(0))
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:                      # never, for a true rigid pair
        return None
    return float(math.atan2(R[1, 0], R[0, 0]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, help="a pf_audit output directory")
    ap.add_argument("--scene", default=None, help="override the map name")
    a = ap.parse_args(argv)

    records = json.loads((a.out / "pf_updates.json").read_text())
    plan, scene = _scene_plan(a.out, a.scene)
    alpha = _world_map_rotation(a.out, records)

    with_body = [r for r in records if r.get("contact_p_body")]
    print(f"# contact-frame audit -- {scene}  ({a.out})")
    print(f"{len(records)} updates, {len(with_body)} carrying body-frame contacts")
    if alpha is not None:
        print(f"world->map rotation recovered from the episode log: "
              f"{alpha:+.4f} rad ({math.degrees(alpha):+.1f} deg)")
    if not with_body:
        print("\nNo contact_p_body recorded -- this audit predates the "
              "instrumentation.  Re-run scripts/pf_audit.py.")
        return 1

    print()
    print(f"{'#':>2s} {'t (s)':>7s} {'|p_body|':>8s} {'resid@truth':>11s} "
          f"{'resid@lead':>10s} {'normal err':>10s} {'resid@world-yaw':>15s}")
    reach, res_t, res_l, dphi, res_bug = [], [], [], [], []
    for r in with_body:
        xy = np.asarray(r["truth_xy"], float).reshape(1, 2)
        th = np.array([float(r["truth_yaw"])])
        reg = np.zeros((1, 2))
        for pb in r["contact_p_body"]:
            p = np.asarray(pb, float)
            q = se2_apply(xy, th, p)
            d, n = nearest_surface(plan, q, reg)
            n_obs = se2_rotate(th, np.asarray(N_BODY, float))
            err = abs(float(wrap_angle(
                math.atan2(float(n_obs[0, 1]), float(n_obs[0, 0]))
                - math.atan2(float(n[0, 1]), float(n[0, 0])))))
            bug = float("nan")
            if alpha is not None:
                # what the recorder scored before the fix: same position, yaw
                # left in world (i.e. rotated back by alpha about the base)
                qb = se2_apply(xy, th - alpha, p)
                bug = abs(float(nearest_surface(plan, qb, reg)[0][0]))
            reach.append(float(np.linalg.norm(p)))
            res_t.append(abs(float(d[0])))
            res_l.append(float(r.get("contact_residual_leading_m", float("nan"))))
            dphi.append(err)
            res_bug.append(bug)
            print(f"{r['index']:>2d} {r['t_s']:>7.1f} {reach[-1]:>8.3f} "
                  f"{res_t[-1]:>11.3f} {res_l[-1]:>10.3f} "
                  f"{math.degrees(err):>9.1f}d {bug:>15.3f}")

    def med(v):
        v = [x for x in v if np.isfinite(x)]
        return float(np.median(v)) if v else float("nan")

    print()
    print(f"median reach      {med(reach):.3f} m   (max allowed {REACH_MAX_M})")
    print(f"median resid@truth {med(res_t):.3f} m")
    print(f"median resid@lead  {med(res_l):.3f} m")
    print(f"median normal err  {math.degrees(med(dphi)):.1f} deg")
    if np.isfinite(med(res_bug)):
        print(f"median resid the yaw bug manufactured: {med(res_bug):.3f} m")

    print()
    bad = [x for x in reach if x > REACH_MAX_M]
    print(f"[{'FAIL' if bad else 'PASS'}] every contact inside the measured reach"
          f" ({len(bad)} of {len(reach)} outside)")
    ok_res = med(res_t) <= max(3.0 * med(res_l), 0.10)
    print(f"[{'PASS' if ok_res else 'FAIL'}] the truth pose explains the contacts"
          f" about as well as the filter's own answer")
    return 0 if (not bad and ok_res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
