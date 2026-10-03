"""Subtask 2 -- is the hardcoded contact normal wrong, and is the wrist better?

    PYTHONPATH=src python3 scripts/audit_contact_normal.py out/normal_probe/r3_base \
        --map r3_tpinch

`_contact_body_frame` asserts the touched face's outward normal is exactly
`(-1, 0)` -- the base's -x -- on every contact.  `PF_AUDIT.md` 4 measured that
constant 6-14 deg out against the mapped surface (median 11.7 on r3, 8.5 on r2)
while `sigma_normal_rad` is 0.22 (12.6 deg), so it fits but spends the entire
normal budget on an assumption.

The wrist reports its own orientation.  Which of its axes points out of the palm
is a FRAME CONVENTION, and this verifies it instead of assuming it: every
candidate axis (+-x, +-y, +-z of the measured wrist frame, expressed in the
base's yaw frame) is scored against the normal of the surface the contact
actually landed on.  The winner is the convention; if none beats the constant,
the constant stays.

Both are then priced in the belief's own currency -- the normal term of
`contact_loglik`, `-0.5 (dphi / sigma_normal)^2` nats.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from global_nav.geometry import se2_rotate, wrap_angle          # noqa: E402
from global_nav.models import nearest_surface                   # noqa: E402

CANDIDATES = [("hardcoded", None, +1.0)] + [
    (f"wrist_{ax}{'+' if sg > 0 else '-'}", ax, sg)
    for ax in ("x", "y", "z") for sg in (+1.0, -1.0)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", type=Path)
    ap.add_argument("--map", required=True)
    ap.add_argument("--sigma-normal", type=float, default=0.22)
    a = ap.parse_args(argv)

    from lane_graph.scene3 import build_v3_scene

    ep = json.loads((a.episode / "episode.json").read_text())
    plan = build_v3_scene(a.map).plan
    tr = ep["tick_trace"]
    base0 = np.asarray(tr[0]["xy"], float)
    yaw0 = float(tr[0]["yaw"])
    v = np.array([math.cos(yaw0), math.sin(yaw0)])
    u = np.array([v[1], -v[0]])

    probes = [p for p in ep["probes"]
              if p["kind"] == "contact" and p.get("wrist_axes")]
    if not probes:
        print("no contacts carrying wrist_axes -- re-run with the instrumented "
              "backend (MujocoConfig records them on every press)")
        return 1

    print(f"# contact-normal audit -- {a.map}  ({a.episode})")
    print(f"{len(probes)} contacts with a measured wrist frame\n")

    rows = {name: [] for name, _, _ in CANDIDATES}
    for p in probes:
        txy = np.asarray(p["truth_xy"], float)
        xy_map = np.array([(txy - base0) @ u, (txy - base0) @ v])
        yaw_map = wrap_angle(float(p["truth_yaw"]) - yaw0 + math.pi / 2.0)
        th = np.array([yaw_map])
        # where the contact landed, and the normal of the surface it landed on
        pb = p.get("p_body")
        if pb is None:
            # p_body is not in the probe log; reconstruct from the reported
            # normal's own stance is impossible, so use the nearest surface to
            # the base's press axis at reach
            pb = (0.40, 0.0)
        q = (xy_map + se2_rotate(th, np.asarray(pb, float))[0]).reshape(1, 2)
        _d, n_surf = nearest_surface(plan, q, np.zeros((1, 2)))
        phi_surf = math.atan2(float(n_surf[0, 1]), float(n_surf[0, 0]))
        for name, ax, sg in CANDIDATES:
            nb = (np.array([-1.0, 0.0]) if ax is None
                  else sg * np.asarray(p["wrist_axes"][ax], float))
            nn = float(np.linalg.norm(nb))
            if nn < 1e-6:
                continue
            nb = nb / nn
            nm = se2_rotate(th, nb)[0]
            dphi = abs(float(wrap_angle(
                math.atan2(float(nm[1]), float(nm[0])) - phi_surf)))
            rows[name].append(dphi)

    print(f"{'candidate':>12s} {'median':>8s} {'mean':>8s} {'max':>8s}"
          f" {'nats lost':>10s}")
    best, best_med = None, math.inf
    for name, _, _ in CANDIDATES:
        e = np.asarray(rows[name], float)
        if not len(e):
            continue
        nats = float(np.mean(0.5 * (e / a.sigma_normal) ** 2))
        print(f"{name:>12s} {math.degrees(np.median(e)):7.1f}d"
              f" {math.degrees(e.mean()):7.1f}d {math.degrees(e.max()):7.1f}d"
              f" {nats:10.3f}")
        if np.median(e) < best_med:
            best, best_med = name, float(np.median(e))

    hard = np.asarray(rows["hardcoded"], float)
    print(f"\nbest candidate: {best}  (median {math.degrees(best_med):.1f} deg)")
    print(f"hardcoded     : median {math.degrees(np.median(hard)):.1f} deg,"
          f" {float(np.mean(0.5 * (hard / a.sigma_normal) ** 2)):.3f} nats lost"
          f" per contact")
    if best == "hardcoded":
        print("\nVERDICT: no measured wrist axis beats the constant.  The frame"
              "\nconvention is NOT verified -- keep (-1, 0).")
    else:
        e = np.asarray(rows[best], float)
        print(f"\nVERDICT: {best} tracks the surface better"
              f" ({math.degrees(np.median(e)):.1f} vs"
              f" {math.degrees(np.median(hard)):.1f} deg).  Set"
              f"\nMujocoConfig.use_measured_normal with that axis and ablate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
