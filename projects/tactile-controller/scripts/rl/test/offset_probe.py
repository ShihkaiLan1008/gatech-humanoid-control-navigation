"""Offset probe: mid-episode base teleport -> does the hand re-attach?

The decisive test of the planner-relative local-skill property. The EE command
(mdp.motion_ee_pose_b) re-points at the WORLD target after the offset, so a
policy that truly closes the loop on "target in my frame" should walk/reach the
hand back onto the wall and recover the press. A policy locked to reference
playback never does (v2-era offset probes: error never recovers).

Protocol: play cfg (deterministic, start-frame RSI, terminations active) on the
training dataset; envs split into offset groups (0 cm control + magnitudes);
run SETTLE steps, teleport the base horizontally in a random per-env direction,
run RECOVER steps; report hand-vs-world-target error decay, scheduled-contact
attach fraction, masked press force, and kills, per group.

    uv run scripts/rl/test/offset_probe.py logs/rsl_rl/.../model_XXXXX.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

DEFAULT_DATA = (
    Path(__file__).resolve().parents[3] / "data" / "wall_contact_schedule_v2_mjlab_grf"
)
MAGNITUDES = (0.0, 0.05, 0.10, 0.15, 0.20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str)
    ap.add_argument("--motion-file", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--envs-per-group", type=int, default=16)
    ap.add_argument("--settle-steps", type=int, default=150)
    ap.add_argument("--recover-steps", type=int, default=300)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import rl.mpc_offline_tracking as mpc_offline_tracking  # noqa: F401  (registers the task)
    from rl.mpc_offline_tracking.mdp.rewards import _normal_force, _sensor_indexes_for_frames

    num_envs = args.envs_per_group * len(MAGNITUDES)
    task = "Unitree-G1-MPC-Tracking"
    env_cfg = load_env_cfg(task, play=True)
    env_cfg.commands["motion"].motion_file = str(Path(args.motion_file).resolve())
    env_cfg.scene.num_envs = num_envs

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    wenv = RslRlVecEnvWrapper(env, clip_actions=None)
    runner = MjlabOnPolicyRunner(wenv, asdict(load_rl_cfg(task)), None, args.device)
    runner.load(str(Path(args.checkpoint).resolve()))
    policy = runner.get_inference_policy(device=args.device)

    cmd = env.command_manager.get_term("motion")
    sensor = env.scene["hand_wall"]
    hand_idx = _sensor_indexes_for_frames(sensor, cmd.motion.hand_frames)
    hand_cols = [cmd.cfg.body_names.index(f) for f in cmd.motion.hand_frames]
    robot = env.scene["robot"]

    torch.manual_seed(args.seed)
    group = torch.arange(num_envs, device=args.device) // args.envs_per_group
    mags = torch.tensor(MAGNITUDES, device=args.device)[group]  # [N]
    theta = torch.rand(num_envs, device=args.device) * 2 * torch.pi
    offset_xy = mags[:, None] * torch.stack(
        [torch.cos(theta), torch.sin(theta)], dim=-1
    )  # [N, 2]

    def hand_err_cm() -> torch.Tensor:
        e = torch.norm(
            cmd.body_pos_w[:, hand_cols] - cmd.robot_body_pos_w[:, hand_cols], dim=-1
        )
        return e.mean(-1) * 100.0  # [N], mean over both hands

    def contact_stats() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-env (attach, force_meas, force_ref) masked by the schedule; nan
        when no hand is scheduled pressing at this step."""
        assert sensor.data.found is not None
        found = (sensor.data.found[:, hand_idx] > 0).float()
        meas = _normal_force(sensor, hand_idx)
        mask = cmd.hand_contact
        n = mask.sum(-1)
        attach = torch.where(n > 0, (mask * found).sum(-1) / n.clamp(min=1), torch.nan)
        fmeas = torch.where(n > 0, (mask * meas).sum(-1) / n.clamp(min=1), torch.nan)
        fref = torch.where(
            n > 0, (mask * -cmd.hand_forces).sum(-1) / n.clamp(min=1), torch.nan
        )
        return attach, fmeas, fref

    obs = wenv.get_observations()
    E, A, FM, FR = [], [], [], []  # per-step [N] traces (settle ++ recover)
    dead = torch.zeros(num_envs, dtype=torch.bool, device=args.device)

    def step_and_record() -> None:
        nonlocal obs
        with torch.no_grad():
            obs, _, _, _ = wenv.step(policy(obs))
        dead[:] |= env.termination_manager.terminated
        a, fm, fr = contact_stats()
        E.append(hand_err_cm())
        A.append(a)
        FM.append(fm)
        FR.append(fr)

    for _ in range(args.settle_steps):
        step_and_record()

    # --- teleport: horizontal base offset, velocities kept ---
    d = robot.data
    root = torch.cat(
        [
            d.root_link_pos_w,
            d.root_link_quat_w,
            d.root_link_lin_vel_w,
            d.root_link_ang_vel_w,
        ],
        dim=-1,
    ).clone()
    root[:, 0:2] += offset_xy
    robot.write_root_state_to_sim(root)
    env.sim.forward()

    for _ in range(args.recover_steps):
        step_and_record()

    Et = torch.stack(E)  # [T, N]
    At, FMt, FRt = torch.stack(A), torch.stack(FM), torch.stack(FR)
    Et[:, dead] = torch.nan  # a killed env never counts as recovered
    At[:, dead] = torch.nan
    FMt[:, dead] = torch.nan
    s = args.settle_steps

    def win(x: torch.Tensor, t0: int, t1: int, g: int) -> float:
        sel = x[t0:t1, group == g]
        return float(np.nanmedian(sel.cpu().numpy()))

    def nanmean_win(x: torch.Tensor, t0: int, t1: int, g: int) -> float:
        sel = x[t0:t1, group == g].cpu().numpy()
        return float(np.nanmean(sel)) if np.isfinite(sel).any() else float("nan")

    print(f"\n=== offset probe: {args.checkpoint}")
    print(
        f"=== {args.envs_per_group} envs/group, settle {args.settle_steps / 50:.0f} s"
        f" -> horizontal teleport (random dir) -> recover {args.recover_steps / 50:.0f} s,"
        " terminations active"
    )
    print(
        f"\n{'offset':>7s} {'kills':>6s} | hand err cm (median):"
        f" {'pre':>5s} {'peak':>5s} {'+1s':>5s} {'+2s':>5s} {'+4s':>5s} {'+6s':>5s}"
        f" | attach%: {'pre':>4s} {'+2s':>4s} {'+6s':>4s}"
        f" | force N meas/ref: {'pre':>9s} {'+2s':>9s} {'+6s':>9s}"
    )
    for g, m in enumerate(MAGNITUDES):
        k = int(dead[group == g].sum())
        pre = win(Et, s - 25, s, g)
        peak = win(Et, s, s + 10, g)
        e1, e2 = win(Et, s + 45, s + 55, g), win(Et, s + 95, s + 105, g)
        e4, e6 = win(Et, s + 195, s + 205, g), win(Et, s + 290, s + 300, g)
        a_pre, a2, a6 = (
            nanmean_win(At, s - 25, s, g) * 100,
            nanmean_win(At, s + 75, s + 125, g) * 100,
            nanmean_win(At, s + 250, s + 300, g) * 100,
        )
        fp = f"{nanmean_win(FMt, s - 25, s, g):4.1f}/{nanmean_win(FRt, s - 25, s, g):4.1f}"
        f2 = f"{nanmean_win(FMt, s + 75, s + 125, g):4.1f}/{nanmean_win(FRt, s + 75, s + 125, g):4.1f}"
        f6 = f"{nanmean_win(FMt, s + 250, s + 300, g):4.1f}/{nanmean_win(FRt, s + 250, s + 300, g):4.1f}"
        print(
            f"{m * 100:5.0f}cm {k:3d}/{args.envs_per_group:2d} |"
            f" {pre:21.1f} {peak:5.1f} {e1:5.1f} {e2:5.1f} {e4:5.1f} {e6:5.1f}"
            f" | {a_pre:12.0f} {a2:4.0f} {a6:4.0f}"
            f" | {fp:>27s} {f2:>9s} {f6:>9s}"
        )

    env.close()


if __name__ == "__main__":
    main()
