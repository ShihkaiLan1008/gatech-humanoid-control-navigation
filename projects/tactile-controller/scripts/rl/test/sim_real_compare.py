"""Sim-side twin of the real deploy log: run model_17500 in the planner play
env under the same forward-velocity command and record the SAME channels the
deploy stack logs (newest-frame obs slices + action), so the real-vs-sim gap
can be read off directly.

  uv run python scripts/rl/test/sim_real_compare.py            # rollout + stats
  uv run python scripts/rl/test/sim_real_compare.py --skip-rollout   # stats only
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

CKPT = Path("logs/rsl_rl/g1_decoupled_joint/2026-08-06_18-29-40/model_17500.pt")
OUT = Path("data/debug_real_0805/sim_twin.npz")
REAL = Path("data/debug_real_0805/debug_0806_0.extract.npz")

NUM_ENVS = 8
WALK_S = 40.0
STAND_S = 10.0

# newest-frame slices inside the history-5 lower actor obs [480]
SL_ANGVEL = slice(12, 15)
SL_GRAV = slice(27, 30)
SL_JPOS = slice(146, 175)
SL_JVEL = slice(291, 320)
SL_STEPCMD = slice(465, 472)


def rollout() -> None:
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    task = "Mjlab-Decoupled-Joint-G1-Planner"
    agent_cfg = load_rl_cfg(task)
    runner_cls = load_runner_cls(task)
    env_cls = runner_cls.env_cls
    wrapper_cls = runner_cls.vec_wrapper_cls

    records = {}
    for phase, twist, dur in (("walk", (0.2, 0.0, 0.0), WALK_S),
                              ("stand", (0.0, 0.0, 0.0), STAND_S)):
        env_cfg = load_env_cfg(task, play=True)
        env_cfg.scene.num_envs = NUM_ENVS
        env_cfg.episode_length_s = dur + 5.0
        env_cfg.commands["step_cmd"].fixed_twist = twist
        env = env_cls(cfg=env_cfg, device=device)
        wrapped = wrapper_cls(env, clip_actions=agent_cfg.clip_actions)
        runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        runner.load(str(CKPT), load_cfg={"actor": True}, strict=True,
                    map_location=device)
        policy = runner.get_inference_policy(device=device)

        obs = wrapped.get_observations()
        n_steps = int(dur / env.step_dt)
        buf: dict[str, list[np.ndarray]] = {k: [] for k in
            ("ang_vel", "grav", "jpos", "jvel", "step_cmd", "action", "root_pitch")}
        with torch.no_grad():
            for _ in range(n_steps):
                low = obs["lower_body_actor"]
                buf["ang_vel"].append(low[:, SL_ANGVEL].cpu().numpy())
                buf["grav"].append(low[:, SL_GRAV].cpu().numpy())
                buf["jpos"].append(low[:, SL_JPOS].cpu().numpy())
                buf["jvel"].append(low[:, SL_JVEL].cpu().numpy())
                buf["step_cmd"].append(low[:, SL_STEPCMD].cpu().numpy())
                act = policy(obs)
                buf["action"].append(act.cpu().numpy())
                q = env.scene["robot"].data.root_link_quat_w  # wxyz
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                pitch = torch.asin((2 * (w * y - z * x)).clamp(-1, 1))
                buf["root_pitch"].append(pitch.cpu().numpy())
                obs, _, _, _ = wrapped.step(act)
        for k, v in buf.items():
            records[f"{phase}_{k}"] = np.stack(v)
        env.close()

    with open(OUT, "wb") as f:
        np.savez_compressed(f, **records)
    print(f"saved sim twin rollout -> {OUT}")


def stats() -> None:
    import re
    from rl.g1_assets.g1_constants import G1_ACTION_SCALE
    from rl.decoupled.mdp import LOWER_JOINT_NAMES, UPPER_JOINT_NAMES

    joints = list(LOWER_JOINT_NAMES) + list(UPPER_JOINT_NAMES)
    scale = np.array([next(s for p, s in G1_ACTION_SCALE.items()
                           if re.fullmatch(p, j)) for j in joints])

    sim = dict(np.load(OUT))
    real = dict(np.load(REAL))
    t = (real["t_ns"] - real["t_ns"][0]) / 1e9

    def pd_stats(jpos, action, label):
        err = action * scale - jpos  # both relative to default
        print(f"\n== {label} (n={len(jpos)})")
        for k, j in enumerate(joints):
            if any(s in j for s in ("hip_pitch", "knee", "ankle_pitch",
                                    "waist_pitch", "hip_roll", "ankle_roll")):
                print(f"  {j:26s} target-meas mean {err[:, k].mean():+.4f} "
                      f"p95|.| {np.percentile(abs(err[:, k]), 95):.4f}")

    # sim arrays are [T, B, ...] -> flatten env axis; skip the first 2 s settle
    skip = 100
    for phase in ("stand", "walk"):
        jp = sim[f"{phase}_jpos"][skip:].reshape(-1, 29)
        ac = sim[f"{phase}_action"][skip:].reshape(-1, 29)
        pd_stats(jp, ac, f"SIM {phase}")
        g = sim[f"{phase}_grav"][skip:].reshape(-1, 3)
        rp = sim[f"{phase}_root_pitch"][skip:].reshape(-1)
        print(f"  grav_x mean {g[:, 0].mean():+.4f} p1 {np.percentile(g[:, 0], 1):+.4f} "
              f"p99 {np.percentile(g[:, 0], 99):+.4f} | root_pitch mean {rp.mean():+.4f} rad")

    stand_m = (t > 5) & (t < 20)
    walk_m = (t > 22) & (t < 41)
    for label, m in (("REAL stand(5-20s)", stand_m), ("REAL walk_fwd(22-41s)", walk_m)):
        pd_stats(real["dof_pos"][m], real["action"][m], label)
        g = real["projected_gravity"][m]
        print(f"  grav_x mean {g[:, 0].mean():+.4f} p1 {np.percentile(g[:, 0], 1):+.4f} "
              f"p99 {np.percentile(g[:, 0], 99):+.4f}")

    # command stream comparison during walking
    sc_sim = sim["walk_step_cmd"][skip:].reshape(-1, 7)
    cv = real["command_vector"][walk_m]
    act_sim = sc_sim[sc_sim[:, 2] > 0]
    act_real = cv[cv[:, 2] > 0]
    print("\n== step command (active frames): target x/y")
    print(f"  sim : x {act_sim[:, 3].mean():+.3f}±{act_sim[:, 3].std():.3f}  "
          f"y {act_sim[:, 4].mean():+.3f}±{act_sim[:, 4].std():.3f}  "
          f"duty {len(act_sim)/len(sc_sim):.2f}")
    print(f"  real: x {act_real[:, 3].mean():+.3f}±{act_real[:, 3].std():.3f}  "
          f"y {act_real[:, 4].mean():+.3f}±{act_real[:, 4].std():.3f}  "
          f"duty {len(act_real)/len(cv):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rollout", action="store_true")
    args = ap.parse_args()
    if not args.skip_rollout:
        rollout()
    stats()


if __name__ == "__main__":
    main()
