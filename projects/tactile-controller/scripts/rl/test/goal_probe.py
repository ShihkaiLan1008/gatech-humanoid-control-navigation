"""Headless probe for the SE(2) pose-goal command layer (CURRENT GOAL step 1).

Drives a trained vel-follower locomotion checkpoint under
``Mjlab-MPC-Guided-Locomotion-Goal-G1`` — the "twist" slot is fed by the
goal-attractor :class:`Se2PoseVelocityCommand` instead of sampled velocities,
obs/action unchanged — and reports whether the command layer actually brings
the robot to its goals: arrival rate, time-to-arrive, settle errors, approach
speed near the goal, and episode resets.

Usage:
    uv run python scripts/rl/test/goal_probe.py \
        --checkpoint /home/zhw/LidarLab/g1_mpc_rl/logs/rsl_rl/g1_mpc_velocity/locomotion/model_9000.pt \
        --num-envs 8 --duration 60
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="vel-follower locomotion .pt")
    p.add_argument("--task", default="Mjlab-MPC-Guided-Locomotion-Goal-G1")
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--duration", type=float, default=60.0, help="sim seconds")
    p.add_argument("--device", default=None)
    p.add_argument("--verbose", action="store_true", help="print every finalized goal")
    args = p.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg = load_rl_cfg(args.task)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(wenv, asdict(agent_cfg), device=device)
    runner.load(
        str(Path(args.checkpoint).resolve()),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    term = env.command_manager.get_term("twist")
    assert term is not None and hasattr(term, "pose_target_w"), (
        "The 'twist' term is not Se2PoseVelocityCommand — wrong task?"
    )
    B = args.num_envs
    dt = env.step_dt
    steps = int(args.duration / dt)

    def _dist_yaw() -> tuple[torch.Tensor, torch.Tensor]:
        pos = term.robot.data.root_link_pos_w[:, :2]
        heading = term.robot.data.heading_w
        d = torch.linalg.norm(term.pose_target_w[:, :2] - pos, dim=-1)
        from mjlab.utils.lab_api.math import wrap_to_pi

        ey = wrap_to_pi(term.pose_target_w[:, 2] - heading).abs()
        return d, ey

    # Per-active-goal trackers.
    counter_prev = term.command_counter.clone()
    d0 = torch.zeros(B, device=device)
    t0 = torch.zeros(B, dtype=torch.long, device=device)
    arrive_t = torch.full((B,), -1, dtype=torch.long, device=device)
    min_d = torch.full((B,), float("inf"), device=device)
    last_d = torch.zeros(B, device=device)
    last_ey = torch.zeros(B, device=device)

    records: list[dict] = []  # finalized goals
    resets = 0
    near_goal_speed_max = 0.0  # max |v_cmd| observed while dist < 0.3 m

    obs = wenv.get_observations()
    # Initialize trackers for the first goals (already resampled at env reset).
    d, ey = _dist_yaw()
    d0.copy_(d)
    last_d.copy_(d)
    last_ey.copy_(ey)

    print(f"[goal_probe] {B} envs, {steps} steps ({args.duration:.0f} s sim), device {device}")
    print("[goal_probe] first MPC solve may take a while (solver JIT/AOT compile)...")

    with torch.no_grad():
        for t in range(steps):
            obs, _, dones, _ = wenv.step(policy(obs))

            d, ey = _dist_yaw()
            arrived = term.arrived
            cmd = term.vel_command_b

            # Near-goal approach-speed check (should follow the decel profile).
            near = d < 0.3
            if near.any():
                spd = torch.linalg.norm(cmd[near, :2], dim=-1).max().item()
                near_goal_speed_max = max(near_goal_speed_max, spd)

            newly_arrived = arrived & (arrive_t < 0)
            arrive_t[newly_arrived] = t
            min_d = torch.minimum(min_d, d)

            # A counter change means a NEW goal was latched for that env
            # (mid-episode resample or episode reset) — finalize the old one.
            switched = term.command_counter != counter_prev
            if dones.any():
                resets += int(dones.sum().item())
            if switched.any():
                for i in switched.nonzero(as_tuple=False).flatten().tolist():
                    dur = (t - int(t0[i])) * dt
                    rec = {
                        "env": i,
                        "d0": float(d0[i]),
                        "dur_s": dur,
                        "arrived": bool(arrive_t[i] >= 0),
                        "t_arrive_s": (int(arrive_t[i]) - int(t0[i])) * dt
                        if arrive_t[i] >= 0
                        else float("nan"),
                        "settle_d": float(last_d[i]),
                        "settle_yaw": float(last_ey[i]),
                        "min_d": float(min_d[i]),
                        "reset": bool(dones[i]) if dones.numel() == B else False,
                    }
                    records.append(rec)
                    if args.verbose:
                        print(
                            f"  goal env{i}: d0={rec['d0']:.2f}m "
                            f"{'ARRIVED %.1fs' % rec['t_arrive_s'] if rec['arrived'] else 'NOT arrived'} "
                            f"settle=({rec['settle_d']*100:.0f}cm, {rec['settle_yaw']*57.3:.0f}deg) "
                            f"min_d={rec['min_d']*100:.0f}cm dur={rec['dur_s']:.1f}s"
                        )
                    # Start tracking the new goal.
                    t0[i] = t
                    arrive_t[i] = -1
                    min_d[i] = d[i]
                    d0[i] = d[i]
                counter_prev.copy_(term.command_counter)

            last_d.copy_(d)
            last_ey.copy_(ey)

    # ── summary ──────────────────────────────────────────────────────────────
    full = [r for r in records if r["dur_s"] >= 3.0]  # goals with a fair chance
    n = len(full)
    print("\n================ goal_probe summary ================")
    print(f"goals finalized: {len(records)} (>=3 s: {n}); episode resets: {resets}")
    if n:
        arr = [r for r in full if r["arrived"]]
        rate = 100.0 * len(arr) / n
        print(f"arrival rate: {rate:.0f}%  ({len(arr)}/{n})")
        if arr:
            ts = sorted(r["t_arrive_s"] for r in arr)
            ds = [r["settle_d"] for r in arr]
            ys = [r["settle_yaw"] for r in arr]
            speed = [r["d0"] / max(r["t_arrive_s"], 1e-6) for r in arr]
            print(
                f"time-to-arrive: median {ts[len(ts)//2]:.1f} s, p90 {ts[int(len(ts)*0.9)]:.1f} s"
                f"  (mean straight-line speed {sum(speed)/len(speed):.2f} m/s)"
            )
            print(
                f"settle error: pos mean {100*sum(ds)/len(ds):.1f} cm, "
                f"yaw mean {57.3*sum(ys)/len(ys):.1f} deg"
            )
        missed = [r for r in full if not r["arrived"]]
        if missed:
            md = [r["min_d"] for r in missed]
            print(
                f"missed goals: {len(missed)}, closest-approach mean {100*sum(md)/len(md):.0f} cm"
            )
    print(f"max |v_cmd| within 0.3 m of goal: {near_goal_speed_max:.2f} m/s "
          f"(decel-profile bound ~0.4)")
    print("====================================================")

    env.close()


if __name__ == "__main__":
    main()
