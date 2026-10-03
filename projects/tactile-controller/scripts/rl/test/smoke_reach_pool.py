"""Smoke-check the reachable-command pool WITHOUT touching the hand_pose task.

Builds the pool the way `ReachPoolEePoseIkCommand` would, then answers the
questions that decide whether it is fit to sample training commands from:

1. Is every emitted pose actually reachable?  Re-solves each one with the SAME
   online DLS IK the RL env runs (torso-rooted tactile-TIP chain on
   `g1_29dof_tactile.urdf` — all hand tasks since 2026-08-01) and
   reports the residual -- this is the number the box sampler loses on.
2. Do the cuts hold?  x / y-outboard / z, checked on the emitted commands.
3. How is coverage distributed, and how much does balancing change it?

Run:
    uv run python -m scripts.rl.test.smoke_reach_pool
    uv run python scripts/rl/test/smoke_reach_pool.py --size 100000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import assets  # noqa: E402
from rl.mpc_online_tracking.tasks.hand_pose.mdp.commands import (  # noqa: E402
    build_arm_chain,
    fused_fk_jac,
)
from rl.mpc_online_tracking.tasks.hand_pose.mdp.reach_pool import (  # noqa: E402
    build_reach_pool,
)

EE_FRAMES = ("left_tactile", "right_tactile")


def default_joint_pos() -> dict[str, float]:
    """Full-name default arm map (both sides, shoulder_roll sign-mirrored)."""
    out = {}
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        for short, val in KEYFRAME.items():
            out[f"{side}_{short}_joint"] = val * (sgn if short == "shoulder_roll" else 1.0)
    return out

# The shipped online-IK weights (UniformEePoseIkCommandCfg defaults).
IK_POS_W, IK_ORI_W, IK_DAMP, IK_MAX_DQ, IK_MAX_DDQ = 1.0, 0.5, 0.05, 0.3, 0.02
IK_POST_GAIN = 0.1  # shipped null-space posture pull (fraction/step)
IK_POST_W_OLD = 0.05  # pre-2026-08-01 penalty weight, kept as an A/B row
# mjlab default keyframe arm pose = the IK posture target.
KEYFRAME = {
    "shoulder_pitch": -0.2094, "shoulder_roll": 0.33165, "shoulder_yaw": 0.0,
    "elbow": 0.5236, "wrist_roll": 0.0, "wrist_pitch": 0.0, "wrist_yaw": 0.0,
}


def dls_residual(
    frame: str,
    pos: torch.Tensor,
    quat: torch.Tensor,
    mode: str = "null",
    strength: float = IK_POST_GAIN,
    q_start: torch.Tensor | None = None,
    start_names: list[str] | None = None,
    pull_q: torch.Tensor | None = None,
    iters: int = 400,
):
    """Converged residual of the env's own IK on these commands.

    `mode="null"` replicates the SHIPPED solver (posture pull strictly in the
    task null space + trapezoid step shaping); `mode="penalty"` replicates the
    pre-2026-08-01 objective-penalty formulation for the A/B rows. `strength`
    is the null-space gain resp. the penalty weight.

    `q_start` seeds the solve from the configuration that generated each
    command instead of from the keyframe; comparing the two separates "the
    solver landed in another basin" from "the objective chose this".

    Returns (pos_err_m, ori_err_deg, posture_dist_rad, max_step, max_accel) —
    the last two are the per-iteration emitted-step extremes, i.e. the
    reference's velocity/accel profile the policy has to track.
    """
    import pytorch_kinematics as pk

    side = frame.split("_")[0]
    dev = pos.device
    chain = pk.build_chain_from_urdf(Path(assets.asset(assets.TACTILE_URDF)).read_text())
    arm = pk.SerialChain(chain, frame, "torso_link").to(device=dev, dtype=torch.float32)
    base = pk.SerialChain(chain, "torso_link", "pelvis").to(device=dev, dtype=torch.float32)
    n_base = len(base.get_joint_parameter_names())
    M0 = base.forward_kinematics(torch.zeros(1, n_base, device=dev)).get_matrix()[0]
    R_pt, t_pt = M0[:3, :3], M0[:3, 3]

    names = arm.get_joint_parameter_names()
    sgn = 1.0 if side == "left" else -1.0
    q_post = torch.tensor(
        [
            KEYFRAME[j.replace(f"{side}_", "").replace("_joint", "")]
            * (sgn if "shoulder_roll" in j else 1.0)
            for j in names
        ],
        device=dev,
    )
    lo, hi = arm.get_joint_limits()
    q_lo = torch.tensor(lo, device=dev)
    q_hi = torch.tensor(hi, device=dev)

    # pelvis-frame command -> torso-rooted chain frame
    p_goal = (pos - t_pt) @ R_pt
    R_cmd = pk.quaternion_to_matrix(quat)
    R_goal = R_pt.transpose(0, 1).unsqueeze(0) @ R_cmd

    q_default = q_post.clone()  # |q-q0| metric always vs the default arms
    if pull_q is not None:  # per-sample posture anchor instead of the keyframe
        q_post = pull_q[:, [start_names.index(j) for j in names]].clone()
    if q_start is None:
        q = q_default.repeat(pos.shape[0], 1).clone()
    else:
        q = q_start[:, [start_names.index(j) for j in names]].clone()
    dq_prev = torch.zeros_like(q)
    null_v = torch.full_like(q, q.shape[1] ** -0.5)
    max_step = 0.0
    max_accel = 0.0
    for _ in range(iters):
        T = arm.forward_kinematics(q).get_matrix()
        J = arm.jacobian(q)
        e_p = p_goal - T[:, :3, 3]
        e_r = pk.matrix_to_axis_angle(R_goal @ T[:, :3, :3].transpose(-1, -2))
        Jp, Jr = J[:, :3], J[:, 3:]
        JTJ = IK_POS_W**2 * Jp.transpose(-1, -2) @ Jp + IK_ORI_W**2 * Jr.transpose(-1, -2) @ Jr
        JTe = IK_POS_W**2 * (Jp.transpose(-1, -2) @ e_p.unsqueeze(-1)).squeeze(-1) + IK_ORI_W**2 * (
            Jr.transpose(-1, -2) @ e_r.unsqueeze(-1)
        ).squeeze(-1)
        if mode == "penalty":
            JTJ.diagonal(dim1=-2, dim2=-1).add_(IK_DAMP**2 + strength**2)
            dq = torch.linalg.solve(JTJ, JTe + strength**2 * (q_post - q)).clamp(
                -IK_MAX_DQ, IK_MAX_DQ
            )
        else:
            JTJ.diagonal(dim1=-2, dim2=-1).add_(IK_DAMP**2)
            sol = torch.linalg.solve(JTJ, torch.stack([JTe, null_v], dim=-1))
            dq, v = sol[..., 0], sol[..., 1]
            null_v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            dq = dq + strength * null_v * ((q_post - q) * null_v).sum(-1, keepdim=True)
            v_cap = torch.minimum(
                torch.full_like(dq, IK_MAX_DQ), (2.0 * IK_MAX_DDQ * dq.abs()).sqrt()
            )
            dq = dq.clamp(-v_cap, v_cap).clamp(dq_prev - IK_MAX_DDQ, dq_prev + IK_MAX_DDQ)
            dq = torch.minimum(dq, (2.0 * IK_MAX_DDQ * (q_hi - q).clamp_min(0.0)).sqrt())
            dq = torch.maximum(dq, -((2.0 * IK_MAX_DDQ * (q - q_lo).clamp_min(0.0)).sqrt()))
        q_new = (q + dq).clamp(q_lo, q_hi)
        step = q_new - q
        max_step = max(max_step, float(step.abs().max()))
        max_accel = max(max_accel, float((step - dq_prev).abs().max()))
        dq_prev = step
        q = q_new

    T = arm.forward_kinematics(q).get_matrix()
    e = (p_goal - T[:, :3, 3]).norm(dim=-1)
    ang = torch.rad2deg(
        pk.matrix_to_axis_angle(R_goal @ T[:, :3, :3].transpose(-1, -2)).norm(dim=-1)
    )
    posture = (q - q_default).norm(dim=-1)
    return e.cpu().numpy(), ang.cpu().numpy(), posture.cpu().numpy(), max_step, max_accel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=300_000)
    ap.add_argument("--check", type=int, default=3000, help="commands to re-solve with IK")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cuda")

    urdf_text = Path(assets.asset(assets.TACTILE_URDF)).read_text()
    print("Building pool (with anchors) ...")
    t0 = time.perf_counter()
    pool = build_reach_pool(
        robot_yml=assets.asset(assets.TACTILE_CUROBO_FIXED_YML),
        ee_frames=EE_FRAMES,
        target=args.size,
        x_margin=0.0,
        y_reach=0.4,
        z_max=None,
        torso_shrink=-0.012,
        voxel=0.05,
        n_lat=9,
        n_lon=18,
        per_cell=8,
        seed=args.seed,
        device=dev,
        return_configs=True,
        anchor_cfg={
            "urdf_text": urdf_text,
            "root_link": "torso_link",
            "default_joint_pos": default_joint_pos(),
        },
    )
    print(f"  built in {time.perf_counter() - t0:.2f} s\n")

    from data_gen.ee_target_curobo.reach_boundary import geometry_bounds, load_kinematics_cfg

    bounds = geometry_bounds(
        load_kinematics_cfg(
            assets.asset(assets.TACTILE_CUROBO_FIXED_YML), ee_frames=EE_FRAMES
        )
    )

    cc_names = None
    ok = True
    for frame, (pos, quat, cfgs, anchor) in pool.items():
        side = frame.split("_")[0]
        p = pos.cpu().numpy()
        mem = pos.element_size() * pos.numel() + quat.element_size() * quat.numel()
        print(f"{frame}: {pos.shape[0]} commands, {mem / 1e6:.1f} MB")
        print(
            f"   extent  x [{p[:, 0].min():+.3f},{p[:, 0].max():+.3f}]  "
            f"y [{p[:, 1].min():+.3f},{p[:, 1].max():+.3f}]  "
            f"z [{p[:, 2].min():+.3f},{p[:, 2].max():+.3f}]"
        )

        # --- cuts hold?
        y_lim = bounds[f"shoulder_y_{side}"] + (0.4 if side == "left" else -0.4)
        viol = {
            "x > 0": int((p[:, 0] <= 0.0).sum()),
            "y outboard": int((p[:, 1] > y_lim).sum() if side == "left" else (p[:, 1] < y_lim).sum()),
            "z <= head": int((p[:, 2] > bounds["head_top_z"] + 1e-6).sum()),
        }
        bad = {k: v for k, v in viol.items() if v}
        print(f"   cuts    {'OK' if not bad else f'VIOLATED {bad}'}")
        ok &= not bad

        # --- quaternions normalized?
        qn = quat.norm(dim=-1)
        print(f"   quat    norm in [{qn.min():.6f}, {qn.max():.6f}]")
        ok &= bool(((qn - 1.0).abs() < 1e-3).all())

        # --- can the env's IK actually consume these commands?
        # The pool's contract is that every command is EXACTLY reachable. The
        # DLS is a separate question: its posture term is a PENALTY, so it
        # trades pose accuracy for staying near the keyframe. Sweeping the
        # weight, and seeding from the generating q, separates the two.
        if cc_names is None:
            from data_gen.ee_target_curobo.reach_boundary import build_collision_checker

            cc_names = build_collision_checker(
                load_kinematics_cfg(
                    assets.asset(assets.TACTILE_CUROBO_FIXED_YML), ee_frames=EE_FRAMES
                )
            ).kinematics.joint_names
        k = min(args.check, pos.shape[0])
        sel = torch.randperm(pos.shape[0], device=dev)[:k]
        print(f"   env DLS IK on {k} of them:")
        print(
            f"      {'start':>14s} {'posture':>14s} {'p50':>9s} {'p90':>9s} "
            f"{'>10mm':>7s} {'|q-q0|p50':>10s}"
        )
        rows = {}
        shipped_stats = None
        for tag, q0 in (("keyframe", None), ("generating q", cfgs[sel])):
            for mode, s in (
                ("penalty", IK_POST_W_OLD),
                ("null", IK_POST_GAIN),
                ("null", 0.0),
            ):
                e, _, post, mstep, macc = dls_residual(
                    frame, pos[sel], quat[sel], mode=mode, strength=s,
                    q_start=q0, start_names=cc_names,
                )
                rows[(tag, mode, s)] = e
                shipped = tag == "keyframe" and mode == "null" and s == IK_POST_GAIN
                if shipped:
                    shipped_stats = (mstep, macc)
                note = (
                    "  <- shipped"
                    if shipped
                    else ("  <- old penalty" if tag == "keyframe" and mode == "penalty" else "")
                )
                print(
                    f"      {tag:>14s} {mode + f' {s:.2f}':>14s} {np.median(e) * 1000:8.2f}m "
                    f"{np.percentile(e, 90) * 1000:8.2f}m {(e > 0.01).mean() * 100:6.1f}% "
                    f"{np.median(post):9.3f}r{note}"
                )
        # ANCHOR-PULL probe: keyframe start, but the posture pull aims at the
        # GENERATING config (what the runtime would do with per-pose anchors
        # stored in the pool). Decides whether pull alone steers the warm
        # start into the right basin, or whether the solver state must be
        # re-seeded at resample.
        e, _, post, _, _ = dls_residual(
            frame, pos[sel], quat[sel], mode="null", strength=IK_POST_GAIN,
            q_start=None, start_names=cc_names, pull_q=cfgs[sel],
        )
        print(
            f"      {'keyframe':>14s} {'anchor 0.10':>14s} {np.median(e) * 1000:8.2f}m "
            f"{np.percentile(e, 90) * 1000:8.2f}m {(e > 0.01).mean() * 100:6.1f}% "
            f"{np.median(post):9.3f}r  <- pull toward stored anchor"
        )
        # POOL CONTRACT: from the generating q with the posture pull off, the
        # DLS must land on the command exactly. Anything else means the pool is
        # emitting poses that are not actually reachable.
        contract = rows[("generating q", "null", 0.0)]
        bad = int((contract > 1e-3).sum())
        print(
            f"   pool contract (generating q, pull off): max {contract.max() * 1000:.4f} mm, "
            f"{bad} of {k} above 1 mm -> {'OK' if bad == 0 else 'BROKEN'}"
        )
        ok &= bad == 0
        # ANCHOR CONTRACT: the stored anchors are exact solutions of their
        # pose (the runtime SEEDS the solver with them at each draw, so their
        # FK error IS the converged reference error) and must be no less
        # natural than the raw generating configs they replace.
        chain = build_arm_chain(urdf_text, frame, "torso_link", dev)
        T_a, _ = fused_fk_jac(chain, anchor)
        e_anchor = (T_a[:, :3, 3] - (pos - chain.root_t) @ chain.root_R).norm(dim=-1)
        q_def_chain = torch.tensor(
            [default_joint_pos()[j] for j in chain.joint_names], device=dev
        )
        cols = [cc_names.index(j) for j in chain.joint_names]
        nat_anchor = (anchor - q_def_chain).norm(dim=-1).median()
        nat_gen = (cfgs[:, cols] - q_def_chain).norm(dim=-1).median()
        n_bad = int((e_anchor > 1.5e-3).sum())
        print(
            f"   anchor contract: FK err max {e_anchor.max() * 1000:.3f} mm "
            f"({n_bad} above 1.5 mm), |q-q0| p50 {nat_anchor:.3f}r vs raw "
            f"{nat_gen:.3f}r -> {'OK' if n_bad == 0 and nat_anchor <= nat_gen else 'BROKEN'}"
        )
        ok &= n_bad == 0 and bool(nat_anchor <= nat_gen)
        # SHIPPED SOLVER: must consume the pool exactly (null-space pull cannot
        # trade accuracy) with a bounded-accel reference profile.
        e_ship = rows[("keyframe", "null", IK_POST_GAIN)]
        mstep, macc = shipped_stats
        n_off = int((e_ship > 5e-3).sum())
        # Accel bound is 2*ddq: a joint-limit graze may decelerate one step
        # faster than the clamp (stopping-distance cap into the limit).
        acc_cap = 2.0 * IK_MAX_DDQ
        print(
            f"   shipped solver: p50 {np.median(e_ship) * 1000:.2f} mm, "
            f"{n_off} of {k} above 5 mm ({n_off / k * 100:.1f}%, basin misses); "
            f"step max {mstep:.3f} rad (cap {IK_MAX_DQ}), "
            f"accel max {macc:.4f} rad/step^2 (cap {acc_cap}) -> "
            f"{'OK' if macc <= acc_cap + 1e-6 and mstep <= IK_MAX_DQ + 1e-6 else 'BROKEN'}"
        )
        ok &= macc <= acc_cap + 1e-6 and mstep <= IK_MAX_DQ + 1e-6
        print()

    print(
        "NOTE: the pool is reachable by construction (pool contract line). The\n"
        "      solver's posture pull lives in the task NULL SPACE, so it converges\n"
        "      exactly where its warm start finds the right basin; the 'old\n"
        "      penalty' row shows the pre-2026-08-01 formulation (~13-28 mm p50)\n"
        "      and the shipped/anchor rows show basin misses are 16-26% from ANY\n"
        "      single start -- which is why the runtime SEEDS the solver with the\n"
        "      stored anchor at each draw (anchor contract line = the resulting\n"
        "      converged error) and the emitted reference glides over under the\n"
        "      trapezoid tracker.\n"
    )
    print("SMOKE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
