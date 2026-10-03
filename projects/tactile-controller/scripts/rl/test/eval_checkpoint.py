"""Deterministic checkpoint evaluation for Unitree-G1-MPC-Tracking.

Runs the trained policy (mean action, play cfg: start-frame RSI, no obs noise,
no push unless --with-push) over the whole dataset with GMT termination ACTIVE,
and measures the things TensorBoard cannot show:

- gait quality: stance duty vs the contact schedule, swing-phase foot height
  vs the reference, in-contact foot slip (the skating triple);
- tracking: anchor error, feet global error, per-joint pos/torque error;
- force: masked press force, measured vs reference (deterministic — unlike the
  noisy end-of-episode Metrics/motion/hand_force_meas snapshots in TB);
- survival: GMT/termination kills.

Every row prints next to the v2_good_hand_good_gait baseline so regressions
are visible at a glance. Typical run: ~2 min on one GPU.

    uv run scripts/test/eval_checkpoint.py logs/rsl_rl/g1_mpc_tracking/<run>/model_XXXXX.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

DEFAULT_DATA = (
    Path(__file__).resolve().parents[3] / "data" / "wall_contact_schedule_v2_mjlab_grf"
)

# v2_good_hand_good_gait / model_39000 (2026-07-16), 64 envs x 500 steps.
BASELINE = {
    "kills_per_env": 0.00,
    "stance_duty": 0.77,
    "stance_duty_ref": 0.76,
    "swing_h_cm": 5.5,
    "swing_h_p90_cm": 6.9,
    "swing_h_ref_cm": 6.2,
    "slip_p95_mm": 1.3,
    "feet_err_cm": 3.8,
    "anchor_err_cm": 3.5,
    "anchor_p95_cm": 6.7,
    "force_meas_n": 10.9,
    "force_ref_n": 10.1,
    "joint_pos_med_deg": 2.2,
    "joint_tau_med_nm": 1.4,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str)
    ap.add_argument("--motion-file", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--with-push", action="store_true", help="re-enable training pushes")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import rl.mpc_offline_tracking as mpc_offline_tracking  # noqa: F401  (registers the task)

    task = "Unitree-G1-MPC-Tracking"
    env_cfg = load_env_cfg(task, play=True)
    env_cfg.commands["motion"].motion_file = str(Path(args.motion_file).resolve())
    env_cfg.scene.num_envs = args.num_envs
    if args.with_push:
        from mjlab.managers.event_manager import EventTermCfg

        from rl.mpc_offline_tracking import mdp
        from rl.mpc_offline_tracking.env_configs.event_cfgs import VELOCITY_RANGE

        env_cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={"velocity_range": VELOCITY_RANGE},
        )

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    wenv = RslRlVecEnvWrapper(env, clip_actions=None)
    runner = MjlabOnPolicyRunner(wenv, asdict(load_rl_cfg(task)), None, args.device)
    runner.load(str(Path(args.checkpoint).resolve()))
    policy = runner.get_inference_policy(device=args.device)

    cmd = env.command_manager.get_term("motion")
    sensor = env.scene["feet_ground"]
    obs = wenv.get_observations()
    feet_cols = [
        cmd.cfg.body_names.index(n)
        for n in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    n_joints = cmd.joint_pos.shape[-1]

    kills = torch.zeros(args.num_envs, device=args.device)
    mc, mh, rh, sl, fe, errs, fmeas, fref = [], [], [], [], [], [], [], []
    jp_err = torch.zeros(n_joints, device=args.device)
    jt_err = torch.zeros(n_joints, device=args.device)
    prev_foot = None
    for _ in range(args.steps):
        with torch.no_grad():
            obs, _, _, _ = wenv.step(policy(obs))
        kills += env.termination_manager.terminated.float()
        found = (sensor.data.found > 0).float()
        ref_c = cmd.feet_contact
        foot_w = cmd.robot_body_pos_w[:, feet_cols]
        ref_foot = cmd.body_pos_w[:, feet_cols]
        mc.append(torch.stack([found.cpu(), ref_c.cpu()]))
        sw = (ref_c < 0.5).cpu()
        mh.append(torch.where(sw, foot_w[:, :, 2].cpu(), torch.nan))
        rh.append(torch.where(sw, ref_foot[:, :, 2].cpu(), torch.nan))
        if prev_foot is not None:
            d = torch.norm(foot_w[:, :, :2] - prev_foot[:, :, :2], dim=-1)
            sl.append(torch.where(found > 0.5, d, torch.nan).cpu())
        prev_foot = foot_w.clone()
        fe.append(torch.norm(ref_foot - foot_w, dim=-1).mean().cpu())
        errs.append(torch.norm(cmd.anchor_pos_w - cmd.robot_anchor_pos_w, dim=-1).cpu())
        fmeas.append(cmd.metrics["hand_force_meas"].cpu().clone())
        fref.append(cmd.metrics["hand_force_ref"].cpu().clone())
        jp_err += (cmd.joint_pos - cmd.robot_joint_pos).abs().mean(0)
        jt_err += (cmd.joint_torque - cmd.robot.data.qfrc_actuator).abs().mean(0)

    mcs = torch.stack(mc)
    mhn = torch.stack(mh).numpy()
    rhn = torch.stack(rh).numpy()
    sln = torch.stack(sl).numpy()
    err = torch.stack(errs).numpy()
    fm = torch.stack(fmeas)
    fr = torch.stack(fref)
    press = fr > 0.5
    jp = (jp_err / args.steps).cpu().numpy() * 57.2958
    jt = (jt_err / args.steps).cpu().numpy()
    names = list(cmd.motion.joint_order or [f"j{i}" for i in range(n_joints)])

    rows = [
        ("kills / env", float(kills.mean()), BASELINE["kills_per_env"], ""),
        ("stance duty (meas)", float(mcs[:, 0].mean()), BASELINE["stance_duty"],
         f"ref {float(mcs[:, 1].mean()):.2f}"),
        ("swing foot height cm", float(np.nanmean(mhn)) * 100, BASELINE["swing_h_cm"],
         f"ref {float(np.nanmean(rhn)) * 100:.1f}"),
        ("swing height p90 cm", float(np.nanpercentile(mhn, 90)) * 100,
         BASELINE["swing_h_p90_cm"], f"ref {float(np.nanpercentile(rhn, 90)) * 100:.1f}"),
        ("in-contact slip p95 mm/step", float(np.nanpercentile(sln, 95)) * 1000,
         BASELINE["slip_p95_mm"], "1 mm/step = 5 cm/s"),
        ("feet global err cm", float(torch.stack(fe).mean()) * 100,
         BASELINE["feet_err_cm"], ""),
        ("anchor err cm", float(err.mean()) * 100, BASELINE["anchor_err_cm"],
         f"p95 {float(np.percentile(err, 95)) * 100:.1f} (base {BASELINE['anchor_p95_cm']})"),
        ("press force N", float(fm[press].mean()), BASELINE["force_meas_n"],
         f"ref {float(fr[press].mean()):.1f} (base ref {BASELINE['force_ref_n']})"),
        ("joint pos err median deg", float(np.median(jp)),
         BASELINE["joint_pos_med_deg"], ""),
        ("joint torque err median N*m", float(np.median(jt)),
         BASELINE["joint_tau_med_nm"], ""),
    ]

    push = "WITH training pushes" if args.with_push else "no push"
    print(f"\n=== eval: {args.checkpoint}")
    print(f"=== {args.num_envs} envs x {args.steps} steps ({args.steps / 50:.0f} s), "
          f"play cfg ({push}), terminations active\n")
    print(f"{'metric':32s} {'this ckpt':>10s} {'v2 base':>9s}  notes")
    for name, val, base, note in rows:
        print(f"{name:32s} {val:10.2f} {base:9.2f}  {note}")

    worst = np.argsort(jp)[::-1][:5]
    print("\nworst 5 joints by pos err (deg / N*m):")
    for i in worst:
        print(f"  {names[int(i)]:28s} {jp[int(i)]:5.1f}   {jt[int(i)]:5.1f}")

    env.close()


if __name__ == "__main__":
    main()
