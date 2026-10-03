"""Headless smoke of the zero-action reference playback path.

Builds the play env for Unitree-G1-MPC-Tracking on one of our wall clips (or a
whole dataset DIRECTORY), steps zero actions, and checks the pieces the viewer
and training flows depend on: the clip-library loader, per-env clip assignment,
the command-managed mocap wall placement, RSI reset-to-reference, and the force
channels. Run:

    uv run scripts/test/smoke_zero_play.py [--motion-file clip.npz|dataset_dir/]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import rl.mpc_offline_tracking as mpc_offline_tracking  # noqa: F401  (registers the task)
from rl.g1_assets.wall import wall_pose_for_clip

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

DEFAULT_CLIP = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "wall_contact_schedule_v2_mjlab_grf"
    / "sched_000300000000.npz"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion-file", type=str, default=str(DEFAULT_CLIP))
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg("Unitree-G1-MPC-Tracking", play=True)
    env_cfg.commands["motion"].motion_file = args.motion_file
    env_cfg.scene.num_envs = 4
    env_cfg.terminations = {}

    # wall entity is injected at cfg-build time now (env_cfgs.py)
    assert "wall" in env_cfg.scene.entities, "wall entity not in env cfg"
    assert "wall_friction" in env_cfg.events, "wall friction DR not injected"
    print("[smoke] wall entity present in cfg (command-managed pose)")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()

    cmd = env.command_manager.get_term("motion")
    motion = cmd.motion
    print(
        f"[smoke] library: {motion.n_clips} clip(s), {motion.time_step_total} "
        f"frames total, fps={motion.fps}, feet_grf="
        f"{'yes' if motion.feet_grf is not None else 'no'}"
    )
    print(
        f"[smoke] clip_ids={cmd.clip_ids.tolist()} time_steps={cmd.time_steps.tolist()}"
    )

    # start-mode RSI must put every env at its clip's first frame (reset()
    # already advances the command once, hence the +1 tolerance)
    diff = cmd.time_steps - motion.clip_offsets[cmd.clip_ids]
    assert ((diff == 0) | (diff == 1)).all(), f"start mode not at clip offset: {diff}"

    # wall: mocap pose must match the assigned clip's metadata (needs a forward)
    import mujoco

    wall_gid = mujoco.mj_name2id(
        env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "wall/wall"
    )
    assert wall_gid >= 0, "no wall geom in compiled model"
    env.sim.forward()
    gx = env.sim.data.geom_xpos.cpu().numpy()  # [nworld, ngeom, 3]
    wall_cfg = env_cfg.commands["motion"].wall_cfg
    origins = env.scene.env_origins.cpu().numpy()  # terrain grid, NOT zero
    for e in range(env_cfg.scene.num_envs):
        wmeta = motion.walls[int(cmd.clip_ids[e])]
        expected = wall_pose_for_clip(wmeta, wall_cfg)
        got = gx[e, wall_gid]
        err = max(abs(float(got[i]) - expected[i] - origins[e][i]) for i in range(3))
        assert err < 1e-4, f"env {e}: wall at {got}, expected {expected[:3]} + origin {origins[e]}"
    print(
        f"[smoke] wall poses match clip metadata for all envs "
        f"(e.g. env0 pos {gx[0, wall_gid]})"
    )

    # RSI: after reset the robot must sit ON the reference (start mode)
    err0 = torch.norm(cmd.robot_anchor_pos_w - cmd.anchor_pos_w, dim=-1).max()
    print(f"[smoke] post-reset anchor error: {float(err0) * 1000:.1f} mm")
    assert err0 < 0.05, "robot not reset onto the reference"

    # force channels: fixed [left, right] columns
    print(
        f"[smoke] frame0 hand_contact={cmd.hand_contact[0].tolist()} "
        f"hand_forces={cmd.hand_forces[0].tolist()}"
    )
    if motion.feet_grf is not None:
        print(
            f"[smoke] frame0 feet_contact={cmd.feet_contact[0].tolist()} "
            f"feet_grf={cmd.feet_grf[0].tolist()}"
        )

    zero = torch.zeros(env.action_space.shape, device=device)  # type: ignore[arg-type]
    for _ in range(args.steps):
        env.step(zero)
    local = (cmd.time_steps - motion.clip_offsets[cmd.clip_ids]).tolist()
    print(
        f"[smoke] after {args.steps} zero steps: local frame={local}, "
        f"anchor err={float(torch.norm(cmd.robot_anchor_pos_w - cmd.anchor_pos_w, dim=-1).max()) * 1000:.0f} mm"
    )
    assert local[0] in (args.steps, args.steps + 1), "command not advancing per step"

    env.close()
    print("[smoke] OK")


if __name__ == "__main__":
    main()
