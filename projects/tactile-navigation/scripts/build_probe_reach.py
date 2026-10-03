#!/usr/bin/env python3
"""Extract a yaw envelope from the accepted ee_qpos.npz training workspace.

Only accepted cell centres at z=0.10, pitch=roll=0 and within half a position
cell of the existing probe ray contribute. The maximum is the projected FK
tip, capped at its target projection. This is sampled IK support, not a
measurement of a locomotion policy's dynamic tracking limit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def build(dataset, margin_m=.03):
    if not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError("margin must be finite and nonnegative")
    dataset = Path(dataset)
    with dataset.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    with np.load(dataset, allow_pickle=False) as pool:
        meta = json.loads(str(pool["meta"]))
        spacing = float(meta["config"]["sampling"]["pos_spacing"])
        hands, excluded = {}, {}
        for hand, sign in (("left", 1), ("right", -1)):
            key = hand + "_tactile__"
            target, fk = pool[key+"target_pos"], pool[key+"pos"]
            pyr = pool[key+"target_pyr"]  # degrees: pitch, yaw, roll
            selected = (pool[key+"is_center"] & np.isclose(target[:, 2], .10)
                        & np.isclose(pyr[:, 0], 0) & np.isclose(pyr[:, 2], 0))
            target, fk, pyr = target[selected], fk[selected], pyr[selected]
            center = np.array([0., sign * .15])
            rows, dropped = [], []
            for yaw in pool[key+"grid_yaw"]:
                rad = np.deg2rad(float(yaw))
                u = np.array([np.cos(rad), np.sin(rad)])
                cross = (target[:, :2] - center) @ np.array([-u[1], u[0]])
                ids = np.flatnonzero(np.isclose(pyr[:, 1], yaw)
                                    & (np.abs(cross) <= spacing / 2 + 1e-6))
                if not ids.size:
                    dropped.append(dict(yaw_deg=float(yaw), reason="no samples on probe ray"))
                    continue
                radius = np.minimum((target[ids, :2] - center) @ u,
                                    (fk[ids, :2] - center) @ u)
                best = int(np.argmax(radius))
                r = np.floor(float(radius[best]) * 10000) / 10000
                if r - margin_m <= .30:
                    dropped.append(dict(yaw_deg=float(yaw), reason="no stroke beyond guard after margin"))
                    continue
                rows.append(dict(yaw_deg=float(yaw), max_radius_m=r,
                                 support_count=int(ids.size),
                                 support_tip_pelvis_m=np.round(fk[ids[best]], 5).tolist()))
            hands[hand], excluded[hand] = rows, dropped
    return dict(schema="tactile_tip_reach_v1", name="ee_qpos_v1_z010",
                geometry=dict(frame="pelvis", point="tactile_tip", center_y_m=.15,
                              height_m=.10, guard_radius_m=.30), margin_m=margin_m,
                source=dict(dataset="data/ee_qpos_v1/ee_qpos.npz", sha256=digest,
                            generator_git=meta["git"], generated_at=meta["created"],
                            kind="sampled_ik_workspace_not_policy_measurement",
                            pos_spacing_m=spacing, cross_ray_tolerance_m=spacing/2,
                            selection="accepted centres; z=0.10; pitch=roll=0; max projected FK tip capped at target"),
                interpolation="minimum_of_adjacent_knots_no_extrapolation",
                hands=hands, excluded=excluded)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--margin-m", type=float, default=.03)
    a = p.parse_args()
    result = build(a.dataset, a.margin_m)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(a.out)
