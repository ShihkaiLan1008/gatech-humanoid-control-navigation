"""A/B probe for the EE-command transition governor (smoothness fix).

Runs the SAME trained joint checkpoint on `Mjlab-Decoupled-Joint-Vel-G1`
(play cfg, headless, deterministic) twice:

  jump  — `jump_prob=1.0`: every resample teleports the reference (the old
          behaviour that produced the arm slam + ringing);
  glide — the task cfg as declared (min-jerk speed-capped reference).

and reports, per mode: hand smoothness (tactile-tip peak speed / acceleration),
upper action rate, arm joint velocity, reference continuity (max per-step
reference displacement), and whether accuracy survives (settle error vs the
FINAL goal over the last steps before each resample).

The checkpoint was TRAINED on jumps, so this probe understates what a
governor-trained policy will do — it answers "does the moving reference
already smooth this checkpoint out without breaking arrival".

    uv run python scripts/rl/test/ee_governor_probe.py \
        --checkpoint checkpoints/decoupled_vel/model_17000.pt \
        --num-envs 64 --duration 30
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch


def _pct(x: torch.Tensor, q: float) -> float:
    return torch.quantile(x.float().flatten(), q).item()


def run_mode(task: str, checkpoint: str, num_envs: int, duration: float,
             device: str, jump: bool) -> dict[str, float]:
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = int(1e9)
    if jump:
        for name in ("left_ee_cmd", "right_ee_cmd"):
            env_cfg.commands[name].jump_prob = 1.0
    agent_cfg = load_rl_cfg(task)

    runner_cls = load_runner_cls(task)
    assert runner_cls is not None
    env = runner_cls.env_cls(cfg=env_cfg, device=device)
    wenv = runner_cls.vec_wrapper_cls(env, clip_actions=agent_cfg.clip_actions)
    runner = runner_cls(wenv, asdict(agent_cfg), device=device)
    runner.load(str(Path(checkpoint).resolve()), load_cfg={"actor": True},
                strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    terms = [env.command_manager.get_term(n)
             for n in ("left_ee_cmd", "right_ee_cmd")]
    robot = env.scene["robot"]
    body_ids = torch.tensor([t.body_idx for t in terms], device=device)

    # upper action slice from the ActionManager layout (declaration order).
    am = env.action_manager
    start = 0
    upper_sl = None
    for n in am.active_terms:
        dim = am.get_term(n).action_dim
        if n == "upper_body_joint_pos":
            upper_sl = slice(start, start + dim)
            break
        start += dim
    assert upper_sl is not None

    B = num_envs
    dt = env.step_dt
    steps = int(duration / dt)
    settle_window = max(1, int(0.3 / dt))   # steps before a switch that count

    ee_speed, ee_acc, act_rate, ref_step = [], [], [], []
    settle_pos, settle_ori = [], []
    # ring buffer of recent goal errors so a switch can look back 0.3 s
    hist_pos = torch.zeros(settle_window, B, 2, device=device)
    hist_ori = torch.zeros(settle_window, B, 2, device=device)
    counter_prev = torch.stack([t.command_counter.clone() for t in terms], dim=-1)
    prev_vel = robot.data.body_link_lin_vel_w[:, body_ids].clone()
    prev_ref = torch.stack([t.pose_command_b[:, :3].clone() for t in terms], dim=1)

    obs = wenv.get_observations().to(device)
    with torch.inference_mode():
        for t_i in range(steps):
            obs, _, _, _ = wenv.step(policy(obs))

            vel = robot.data.body_link_lin_vel_w[:, body_ids]
            acc = (vel - prev_vel).norm(dim=-1) / dt
            prev_vel = vel.clone()

            ref = torch.stack([t.pose_command_b[:, :3] for t in terms], dim=1)
            dref = (ref - prev_ref).norm(dim=-1)
            prev_ref = ref.clone()

            da = am.action[:, upper_sl] - am.prev_action[:, upper_sl]

            if t_i > 0:   # first frame: stale prev_vel across reset
                ee_speed.append(vel.norm(dim=-1).cpu())
                ee_acc.append(acc.cpu())
                act_rate.append(da.abs().mean(dim=-1).cpu())

            gpos = torch.stack(
                [t.metrics["goal_position_error"] for t in terms], dim=-1)
            gori = torch.stack(
                [t.metrics["goal_orientation_error"] for t in terms], dim=-1)
            counter = torch.stack([t.command_counter for t in terms], dim=-1)
            switched = counter != counter_prev
            if t_i > settle_window and switched.any():
                # metrics this step were computed BEFORE the resample — they
                # and the ring buffer belong to the OLD goal's hold phase.
                look = torch.cat([hist_pos, gpos.unsqueeze(0)], dim=0)
                looko = torch.cat([hist_ori, gori.unsqueeze(0)], dim=0)
                settle_pos.append(look.mean(dim=0)[switched].cpu())
                settle_ori.append(looko.mean(dim=0)[switched].cpu())
            if t_i > 0:
                ref_step.append(dref.cpu())
            counter_prev = counter.clone()
            hist_pos = torch.cat([hist_pos[1:], gpos.unsqueeze(0)], dim=0)
            hist_ori = torch.cat([hist_ori[1:], gori.unsqueeze(0)], dim=0)

    env.close()

    speed = torch.cat([x.flatten() for x in ee_speed])
    accs = torch.cat([x.flatten() for x in ee_acc])
    rate = torch.cat([x.flatten() for x in act_rate])
    refs = torch.cat([x.flatten() for x in ref_step])
    spos = torch.cat(settle_pos) if settle_pos else torch.zeros(1)
    sori = torch.cat(settle_ori) if settle_ori else torch.zeros(1)
    return {
        "ee_speed_p95": _pct(speed, 0.95), "ee_speed_max": speed.max().item(),
        "ee_acc_p95": _pct(accs, 0.95), "ee_acc_p99": _pct(accs, 0.99),
        "act_rate_mean": rate.mean().item(), "act_rate_p95": _pct(rate, 0.95),
        "ref_step_p99": _pct(refs, 0.99), "ref_step_max": refs.max().item(),
        "settle_pos_p50": _pct(spos, 0.5), "settle_pos_p90": _pct(spos, 0.9),
        "settle_ori_p50": _pct(sori, 0.5), "settle_ori_p90": _pct(sori, 0.9),
        "n_settles": float(spos.numel()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",
                   default="checkpoints/decoupled_vel/model_17000.pt")
    p.add_argument("--task", default="Mjlab-Decoupled-Joint-Vel-G1")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--duration", type=float, default=30.0, help="sim seconds")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    from mjlab.utils.torch import configure_torch_backends

    import rl.decoupled.tasks  # noqa: F401  (registers the decoupled tasks)

    configure_torch_backends()

    results = {}
    for mode, jump in (("jump", True), ("glide", False)):
        print(f"\n[ee_governor_probe] mode={mode} ...")
        results[mode] = run_mode(args.task, args.checkpoint, args.num_envs,
                                 args.duration, device, jump)

    j, g = results["jump"], results["glide"]
    print("\n================= ee_governor_probe =================")
    print(f"{'':28s}{'jump':>10s}{'glide':>10s}")
    rows = [
        ("hand speed p95 [m/s]", "ee_speed_p95", "{:.2f}"),
        ("hand speed max [m/s]", "ee_speed_max", "{:.2f}"),
        ("hand acc p95 [m/s^2]", "ee_acc_p95", "{:.1f}"),
        ("hand acc p99 [m/s^2]", "ee_acc_p99", "{:.1f}"),
        ("upper act-rate mean", "act_rate_mean", "{:.3f}"),
        ("upper act-rate p95", "act_rate_p95", "{:.3f}"),
        ("ref step p99 [m]", "ref_step_p99", "{:.4f}"),
        ("ref step max [m]", "ref_step_max", "{:.3f}"),
        ("settle pos p50 [m]", "settle_pos_p50", "{:.3f}"),
        ("settle pos p90 [m]", "settle_pos_p90", "{:.3f}"),
        ("settle ori p50 [rad]", "settle_ori_p50", "{:.3f}"),
        ("settle ori p90 [rad]", "settle_ori_p90", "{:.3f}"),
        ("settles counted", "n_settles", "{:.0f}"),
    ]
    for label, key, fmt in rows:
        print(f"{label:28s}{fmt.format(j[key]):>10s}{fmt.format(g[key]):>10s}")
    print("=====================================================")
    print("glide ref_step_p99 should be ~peak_speed*dt (<~0.02 m); the max "
          "rows still contain the jump_prob DR teleports. Settle errors say "
          "whether accuracy survived.")


if __name__ == "__main__":
    main()
