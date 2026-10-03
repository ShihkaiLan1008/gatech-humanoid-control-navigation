"""Press probe: run a checkpoint in the hand_force env and quantify contact.

Loads a (stage-1 hand_pose or stage-2 hand_force) checkpoint actor-only into
the play env with TRAINING contact sampling restored, and reports per-window:
contact-achieved ratio, measured force, force-tracking error, and gait
survival. For a stage-1 checkpoint this is the pre-training baseline (it
tracks the pose commands but has never seen a force command); after stage-2
training the same probe measures what the run bought.

Run:
  uv run python scripts/rl/test/press_probe.py \
      --checkpoint logs/rsl_rl/g1_mpc_hand_pose/final_hand_pose/model_24500.pt
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="logs/rsl_rl/g1_mpc_hand_pose/final_hand_pose/model_24500.pt",
    )
    p.add_argument("--task", default="Mjlab-MPC-Guided-Hand-Force-G1")
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--duration", type=float, default=40.0, help="sim seconds")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    # Play cfg slows the cadence for the eye; the probe wants the TRAINING
    # contact distribution (and moderate play twists are kept).
    for name in ("left_ee_pose", "right_ee_pose"):
        env_cfg.commands[name].resampling_time_range = (1.0, 3.0)
        env_cfg.commands[name].contact_resampling_time_range = (3.0, 6.0)
        env_cfg.commands[name].debug_vis = False
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

    B = args.num_envs
    dt = env.step_dt
    steps = int(args.duration / dt)
    terms = {
        "L": env.command_manager.get_term("left_ee_pose"),
        "R": env.command_manager.get_term("right_ee_pose"),
    }

    twist = env.command_manager.get_term("twist")
    robot = env.scene["robot"]

    obs, _ = env.reset()
    masked_steps = 0
    contact_steps = 0
    force_err_sum = 0.0
    force_err_n = 0
    f_meas_sum = 0.0
    fell = torch.zeros(B, dtype=torch.bool, device=device)
    pos_err_free_sum = 0.0
    pos_err_free_n = 0
    # Locomotion-during-window buckets: does the policy still follow the twist
    # while a contact window is open? (in = any hand settled-masked, out =
    # both free; "walk cmd" restricts to |v_cmd_xy| > 0.3.)
    vel_stats = {
        k: {"cmd": 0.0, "meas": 0.0, "err": 0.0, "along": 0.0, "n": 0}
        for k in ("window", "free", "window_walk_cmd", "free_walk_cmd")
    }
    # Force-ripple proxy: per-step |Δf_meas| within persisting contact, split
    # by base speed (tests the riding-cube discretization sawtooth).
    ripple = {
        "still": {"sum": 0.0, "n": 0},
        "moving": {"sum": 0.0, "n": 0},
    }
    prev_f = {k: None for k in terms}
    prev_touch = {k: None for k in terms}
    fsm_hist = [0, 0, 0, 0]  # stand / random / approach / follow (wall task)

    with torch.no_grad():
        for _ in range(steps):
            action = policy(obs)
            obs, _, _, _ = wenv.step(action)

            root_z = env.scene["robot"].data.root_link_pos_w[:, 2]
            fell |= root_z < 0.45  # latch early: mid-fall flailing pollutes stats
            alive = ~fell

            if hasattr(twist, "fsm_state"):
                for s in range(4):
                    fsm_hist[s] += int(((twist.fsm_state == s) & alive).sum())

            any_settled = torch.zeros(B, dtype=torch.bool, device=device)
            for term in terms.values():
                any_settled |= (term.contact_mask > 0.5) & (term._contact_age > 1.0)
            any_settled &= alive
            v_cmd = twist.command[:, :2]
            v_meas = robot.data.root_link_lin_vel_b[:, :2]
            speed_cmd = v_cmd.norm(dim=-1)
            walk_cmd = speed_cmd > 0.3
            along = (v_cmd * v_meas).sum(-1) / speed_cmd.clamp(min=1e-6)
            for key, sel in (
                ("window", any_settled),
                ("free", ~any_settled & alive),
                ("window_walk_cmd", any_settled & walk_cmd),
                ("free_walk_cmd", ~any_settled & alive & walk_cmd),
            ):
                if sel.any():
                    s = vel_stats[key]
                    s["cmd"] += float(speed_cmd[sel].sum())
                    s["meas"] += float(v_meas.norm(dim=-1)[sel].sum())
                    s["err"] += float((v_cmd - v_meas).norm(dim=-1)[sel].sum())
                    s["along"] += float(along[sel].sum())
                    s["n"] += int(sel.sum())

            base_speed = robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
            for key, term in terms.items():
                masked = term.contact_mask > 0.5
                # Skip the first second of each window (reach transit).
                settled = masked & (term._contact_age > 1.0) & alive
                f_meas = term.measured_force_w()
                touching = f_meas.norm(dim=-1) > 0.5
                if prev_f[key] is not None:
                    persist = settled & touching & prev_touch[key]
                    if persist.any():
                        df = (f_meas - prev_f[key]).norm(dim=-1)
                        for bucket, sel in (
                            ("still", persist & (base_speed < 0.15)),
                            ("moving", persist & (base_speed > 0.35)),
                        ):
                            if sel.any():
                                ripple[bucket]["sum"] += float(df[sel].sum())
                                ripple[bucket]["n"] += int(sel.sum())
                prev_f[key] = f_meas.clone()
                prev_touch[key] = (settled & touching).clone()
                masked_steps += int(settled.sum())
                contact_steps += int((settled & touching).sum())
                if settled.any():
                    err = (f_meas - term.force_cmd_w).norm(dim=-1)[settled]
                    force_err_sum += float(err.sum())
                    force_err_n += int(settled.sum())
                    f_meas_sum += float(f_meas.norm(dim=-1)[settled].sum())
                free = ~masked & alive
                if free.any():
                    tgt, _q = term.target_pose_w()
                    ee, _q2 = term.measured_ee_pose_w()
                    pos_err_free_sum += float(
                        (tgt - ee).norm(dim=-1)[free].sum()
                    )
                    pos_err_free_n += int(free.sum())

    print(f"checkpoint : {args.checkpoint}")
    print(f"envs x sec : {B} x {args.duration:.0f}  (dt {dt:.3f})")
    print(f"fell       : {int(fell.sum())}/{B}")
    if fell.any() and hasattr(twist, "wall_map"):
        ids = fell.nonzero().flatten()
        clr = twist.wall_map.clearance(
            env.scene["robot"].data.root_link_pos_w[ids, :2], ids
        )
        print(f"  fallen-env dist to nearest wall: {[f'{c:.2f}' for c in clr.tolist()]}")
    if pos_err_free_n:
        print(f"free-hand pos err   : {pos_err_free_sum / pos_err_free_n:.3f} m")
    if masked_steps:
        print(
            f"window contact ratio: {contact_steps / masked_steps:.2%} "
            f"(settled window steps: {masked_steps})"
        )
    if force_err_n:
        print(f"mean |f_meas| in-window : {f_meas_sum / force_err_n:.2f} N")
        print(f"mean |f_meas - f_cmd|   : {force_err_sum / force_err_n:.2f} N")
    print("locomotion during contact windows (xy, base frame):")
    for k in ("window", "free", "window_walk_cmd", "free_walk_cmd"):
        s = vel_stats[k]
        if s["n"]:
            n = s["n"]
            print(
                f"  {k:16s} n={n:7d}  |v_cmd| {s['cmd'] / n:.2f}  "
                f"|v_meas| {s['meas'] / n:.2f}  |v_err| {s['err'] / n:.2f}  "
                f"progress-along-cmd {s['along'] / n:.2f} m/s"
            )
    print("in-contact force ripple |Δf| per step:")
    for k in ("still", "moving"):
        r = ripple[k]
        if r["n"]:
            print(f"  base {k:6s}: {r['sum'] / r['n']:.2f} N  (n={r['n']})")
    if sum(fsm_hist):
        tot = sum(fsm_hist)
        names = ("stand", "random", "approach", "follow")
        print(
            "FSM state fractions (alive envs): "
            + "  ".join(f"{n} {c / tot:.0%}" for n, c in zip(names, fsm_hist))
        )


if __name__ == "__main__":
    main()
