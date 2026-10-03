"""Write the reach-pool artifacts the trainer loads instead of building with cuRobo.

    uv run python -m data_gen.ee_target_curobo.reach_pool_export --seeds 0 1 2 3

One npz per seed under `data/ee_target/`, named by the
hash of the pool knobs and the arm posture (`EeTargetCommandCfg` defaults and
the robot cfg's arm joint defaults), so a term whose knobs differ never picks
up a stale file. Needs cuRobo (`uv sync --extra curobo`).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import fields

import torch

import assets
from rl.decoupled.mdp.commands.ee_tartget import (
    EeTargetCommandCfg,
    build_reach_pool,
    load_reach_pool,
    reach_pool_path,
    save_reach_pool,
)
from rl.g1_assets.g1_constants import get_g1_robot_cfg


def default_q_ref(robot_cfg) -> dict[str, float]:
    """Joint defaults the entity would report (`init_state.joint_pos` patterns
    over the model's joint names, unmatched joints 0), keyed by joint name."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(assets.asset(assets.UNITREE_TACTILE_XML))
    names = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    q = dict.fromkeys(names, 0.0)
    for pat, val in robot_cfg.init_state.joint_pos.items():
        for n in names:
            if re.fullmatch(pat, n):
                q[n] = float(val)
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--force", action="store_true", help="rewrite existing files")
    args = ap.parse_args()

    cfg = {f.name: f.default for f in fields(EeTargetCommandCfg)}
    key = tuple(cfg[k] for k in (
        "robot_yml", "ee_frames", "pool_size", "x_margin", "y_reach", "y_margin",
        "z_max", "z_min", "torso_shrink", "pool_voxel", "pool_axis_lat",
        "pool_axis_lon", "pool_per_cell"))
    q_ref = default_q_ref(get_g1_robot_cfg())
    device = torch.device("cuda:0")
    for seed in args.seeds:
        path = reach_pool_path(key, q_ref, seed)
        if path.exists() and not args.force:
            print(f"[reach_pool_export] exists, skipping: {path.name}")
            continue
        pool = build_reach_pool(
            robot_yml=assets.asset(cfg["robot_yml"]), ee_frames=cfg["ee_frames"],
            target=cfg["pool_size"], x_margin=cfg["x_margin"], y_reach=cfg["y_reach"],
            y_margin=cfg["y_margin"], z_max=cfg["z_max"], z_min=cfg["z_min"],
            torso_shrink=cfg["torso_shrink"], voxel=cfg["pool_voxel"],
            n_lat=cfg["pool_axis_lat"], n_lon=cfg["pool_axis_lon"],
            per_cell=cfg["pool_per_cell"], seed=seed, device=device, q_ref=q_ref)
        save_reach_pool(path, pool, key, q_ref, seed, generator="curobo build_reach_pool")
        back = load_reach_pool(path, device)
        for frame, (pos, quat, anchor, names) in pool.items():
            b = back[frame]
            assert torch.equal(pos, b[0]) and torch.equal(quat, b[1]) and torch.equal(anchor, b[2]) and names == b[3]
        print(f"[reach_pool_export] wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
