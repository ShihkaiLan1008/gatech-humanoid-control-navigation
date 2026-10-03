"""Measure the centroidal MPC's frozen-moment-arm linearization error.

The QP linearizes the bilinear moment arm ``(r_i - c) x f_i`` by freezing the
CoM at ``c_bar`` (whatever ``MPCInput.c_bar`` carries; when None the solver
falls back to ``x_ref[:, :N, :3]``, an open-loop rollout of the commanded
twist from the measured pelvis).  The dropped term is

    tau_err = (c - c_bar) x F_total

which lands ENTIRELY on the angular-momentum row and is invisible to every
existing diagnostic: the dynamics residual is zero by construction and PiMPC
always reports ``feasible=True``.

This probe rolls a trained policy and compares, at every solve, the CoM the QP
actually planned (``x_pred``) against the CoM it assumed while building B_k
(``c_bar``).  The resulting torque error is reported against the ankle CoP
budget, which is what it competes with:

    roll  budget ~ foot_y_half * fz  (+-0.03 m -> the TIGHT direction)
    pitch budget ~ foot_x_toe  * fz  (+-0.09 m)

If p95 |tau_err| is well under the budget the frozen-arm approximation is fine
and the c_bar work (SQP carry-over, LIP sway) is not worth doing.

Run:
  uv run python scripts/rl/test/mpc_linearization_probe.py \
      --checkpoint logs/rsl_rl/g1_decoupled_joint_vel_mpc/2026-08-10_09-01-29/model_XXXX.pt
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

# Node-index alignment (see cd.py / centroidal_mpc.py):
#   c_bar[:, k, :]  = linearization point for node k (x_ref[:, k, :3] fallback)
#   x_pred[:, j, :] = x_{j+1}              the solver strips x_0
# so the CoM the QP planned for node k (k >= 1) is x_pred[:, k-1, :3].
# Node 0 is exact by construction (c_bar[0] == measured pelvis) and is skipped.


def _pct(t: torch.Tensor, q: float) -> float:
    if t.numel() == 0:
        return float("nan")
    return float(torch.quantile(t.flatten().float(), q))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", default="Mjlab-Decoupled-Joint-Vel-MPC-G1")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--duration", type=float, default=20.0, help="sim seconds")
    p.add_argument("--device", default=None)
    p.add_argument("--warmup", type=float, default=2.0, help="sim seconds discarded")
    args = p.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    from rl.decoupled.cd_mpc import cd as cdm
    from rl.decoupled.cd_mpc import centroidal_mpc as cm
    from rl.envs.joint_manager_based_rl_env import JointManagerBasedRlEnv

    configure_torch_backends()

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.commands["loco_mpc"].debug_vis = False
    agent_cfg = load_rl_cfg(args.task)

    env = JointManagerBasedRlEnv(cfg=env_cfg, device=device)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    # The joint (decoupled) runner needs its own wrapper -- same dispatch play.py uses.
    wrapper_cls = getattr(runner_cls, "vec_wrapper_cls", RslRlVecEnvWrapper)
    wenv = wrapper_cls(env, clip_actions=agent_cfg.clip_actions)
    runner = runner_cls(wenv, asdict(agent_cfg), device=device)
    runner.load(
        str(Path(args.checkpoint).resolve()),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    # ---- instrument CentroidalMPC.solve (no production code touched) --------
    # Two DIFFERENT errors are measured, and the distinction matters:
    #   (A) self-consistency:  x_pred - c_bar.  How far the QP's OWN plan moved
    #       off the point it linearized about.  This is the part an SQP/carry-
    #       over c_bar could remove.  It is a LOWER BOUND -- the QP is
    #       optimistic about its own tracking.
    #   (B) realized:  measured pelvis at t+k*mpc_dt  -  c_bar[k].  The error
    #       the robot actually incurs.  This is the number that matters.
    buf: dict[str, list[torch.Tensor]] = {"e": [], "tau": [], "fz": [], "tau_cmd": []}
    cbar_log: list[tuple[int, torch.Tensor, torch.Tensor]] = []  # (step, c_bar, F)
    recording = {"on": False, "step": 0}

    def _make_patched(orig_solve):
        def _patched_solve(self, mpc_in):
            out = orig_solve(self, mpc_in)
            if not recording["on"]:
                return out
            cb_full = (mpc_in.c_bar if mpc_in.c_bar is not None
                       else mpc_in.x_ref[:, :, :3])
            cbar_log.append((
                recording["step"],
                cb_full[:, : self.cfg.N, :].detach().cpu().clone(),
                (out.u_pred[:, :, 0:3] + out.u_pred[:, :, 6:9]).detach().cpu().clone(),
            ))
            N = self.cfg.N
            c_bar = cb_full[:, :N, :]                          # [B, N, 3]
            c_sol = out.x_pred[:, :, :3]                       # [B, N, 3] = x_1..x_N
            e = c_sol[:, : N - 1, :] - c_bar[:, 1:N, :]        # [B, N-1, 3] nodes 1..N-1

            u = out.u_pred[:, 1:N, :]                          # forces at nodes 1..N-1
            f_lf, f_rf = u[:, :, 0:3], u[:, :, 6:9]
            F = f_lf + f_rf                                    # [B, N-1, 3]

            tau_err = torch.cross(e, F, dim=-1)                # the DROPPED term

            # For scale: the moment the QP *does* model, about c_bar.
            r_lf = mpc_in.schedule.r_LF[:, 1:N, :] - c_bar[:, 1:N, :]
            r_rf = mpc_in.schedule.r_RF[:, 1:N, :] - c_bar[:, 1:N, :]
            tau_cmd = torch.cross(r_lf, f_lf, dim=-1) + torch.cross(r_rf, f_rf, dim=-1)

            buf["e"].append(e.detach().cpu())
            buf["tau"].append(tau_err.detach().cpu())
            buf["fz"].append(F[..., 2].detach().cpu())
            buf["tau_cmd"].append(tau_cmd.detach().cpu())
            return out
        return _patched_solve

    # Patch BOTH implementations -- the task cfg picks one via mpc_impl.
    _orig_cm, _orig_cd = cm.CentroidalMPC.solve, cdm.CentroidalMPC.solve
    cm.CentroidalMPC.solve = _make_patched(_orig_cm)
    cdm.CentroidalMPC.solve = _make_patched(_orig_cd)
    mpc_obj = env.command_manager.get_term("loco_mpc")._mpc
    print(f"MPC impl in the loop: {type(mpc_obj).__module__}")

    # ---- rollout -------------------------------------------------------------
    dt = env.step_dt
    steps = int(args.duration / dt)
    warm = int(args.warmup / dt)
    twist_term = env.command_manager.get_term("twist")
    v_cmd: list[torch.Tensor] = []

    robot = env.scene["robot"]
    pel_hist: list[torch.Tensor] = []
    done_hist: list[torch.Tensor] = []

    env.reset()
    with torch.inference_mode():
        for i in range(steps):
            recording["on"] = i >= warm
            recording["step"] = i
            act = policy(wenv.get_observations().to(device))
            _, _, dones, _ = wenv.step(act.to(device))
            if recording["on"]:
                v_cmd.append(twist_term.command[:, :3].detach().cpu().clone())
            pel_hist.append(robot.data.root_link_pos_w.detach().cpu().clone())
            done_hist.append(dones.detach().cpu().clone().bool())

    cm.CentroidalMPC.solve = _orig_cm
    cdm.CentroidalMPC.solve = _orig_cd

    # ---- report --------------------------------------------------------------
    e = torch.cat(buf["e"])          # [S, N-1, 3]
    tau = torch.cat(buf["tau"])
    fz = torch.cat(buf["fz"])
    tau_cmd = torch.cat(buf["tau_cmd"])
    vc = torch.cat(v_cmd)
    cfg = env_cfg.commands["loco_mpc"]

    fy, fx = 0.03, 0.09  # foot_y_half, foot_x_toe  -> CoP torque budgets
    roll_budget = fy * fz.clamp(min=0.0)
    pitch_budget = fx * fz.clamp(min=0.0)

    print(f"\n{'='*72}")
    print(f"MPC linearization probe -- {args.task}")
    print(f"  ckpt      {args.checkpoint}")
    print(f"  envs {args.num_envs}  rollout {args.duration:.0f}s (warmup {args.warmup:.0f}s)")
    print(f"  solves recorded: {e.shape[0] // args.num_envs} per env, N={cfg.mpc_horizon}, mpc_dt={cfg.mpc_dt}")
    print(f"  commanded twist |vx| p50/p95 = {_pct(vc[:,0].abs(),.5):.3f}/{_pct(vc[:,0].abs(),.95):.3f}"
          f"   |vy| = {_pct(vc[:,1].abs(),.5):.3f}/{_pct(vc[:,1].abs(),.95):.3f}"
          f"   |wz| = {_pct(vc[:,2].abs(),.5):.3f}/{_pct(vc[:,2].abs(),.95):.3f}")
    print(f"  total vertical force fz p50 = {_pct(fz,.5):.1f} N")
    print(f"{'='*72}")

    print("\n-- |c - c_bar| per axis  [m]  (how far the QP moved off its own linearization point)")
    for i, ax in enumerate("xyz"):
        a = e[..., i].abs()
        print(f"   {ax}:  p50 {_pct(a,.5):.4f}   p95 {_pct(a,.95):.4f}   max {a.max():.4f}")

    print("\n-- |tau_err| = |(c - c_bar) x F|  [N*m]  (the DROPPED term)")
    names = {0: "roll (x)", 1: "pitch(y)", 2: "yaw  (z)"}
    for i in range(3):
        a = tau[..., i].abs()
        print(f"   {names[i]}:  p50 {_pct(a,.5):.3f}   p95 {_pct(a,.95):.3f}   max {a.max():.3f}")

    print("\n-- tau_err vs the ankle CoP budget it competes with  [ratio, >1 = error exceeds authority]")
    rr = (tau[..., 0].abs() / roll_budget.clamp(min=1e-3))
    rp = (tau[..., 1].abs() / pitch_budget.clamp(min=1e-3))
    print(f"   roll  |tau_err| / (0.03*fz):  p50 {_pct(rr,.5):.3f}   p95 {_pct(rr,.95):.3f}")
    print(f"   pitch |tau_err| / (0.09*fz):  p50 {_pct(rp,.5):.3f}   p95 {_pct(rp,.95):.3f}")

    print("\n-- tau_err vs the modelled moment (relative model error)")
    rel = tau.norm(dim=-1) / tau_cmd.norm(dim=-1).clamp(min=1e-3)
    print(f"   |tau_err| / |tau_modelled|:   p50 {_pct(rel,.5):.3f}   p95 {_pct(rel,.95):.3f}")

    print("\n-- growth along the horizon (node 1 .. N-1): p95 |tau_err| [N*m]")
    for k in range(e.shape[1]):
        print(f"   node {k+1:2d} (t={cfg.mpc_dt*(k+1):.2f}s):  "
              f"roll {_pct(tau[:,k,0].abs(),.95):6.3f}   "
              f"pitch {_pct(tau[:,k,1].abs(),.95):6.3f}   "
              f"|e| {_pct(e[:,k,:].norm(dim=-1),.95):.4f} m")

    # ---- (B) REALIZED error: measured pelvis vs the c_bar used for it --------
    pel = torch.stack(pel_hist)      # [T, B, 3]
    dn = torch.stack(done_hist)      # [T, B]
    T = pel.shape[0]
    ratio = cfg.mpc_dt / dt          # policy steps per MPC node
    er_all, tr_all = [], []
    for k in range(1, cfg.mpc_horizon):
        off = int(round(k * ratio))
        eks, tks = [], []
        for s, cb, F in cbar_log:
            t_hit = s + off
            if t_hit >= T:
                continue
            # drop any env that reset between the solve and the target time
            alive = ~dn[s:t_hit + 1].any(dim=0)
            if not alive.any():
                continue
            ek = pel[t_hit][alive] - cb[:, k, :][alive]
            eks.append(ek)
            tks.append(torch.cross(ek, F[:, k, :][alive], dim=-1))
        if eks:
            er_all.append(torch.cat(eks))
            tr_all.append(torch.cat(tks))

    print("\n-- (B) REALIZED error: MEASURED pelvis at t+k*dt  vs  c_bar[k]")
    print("   (this is the error the robot actually incurs; (A) above is only")
    print("    the QP's self-consistency error, a lower bound)")
    print("   |e| per axis [m, p50]        roll tau [N*m]     vs 0.03*fz")
    for k, (er, tr) in enumerate(zip(er_all, tr_all), start=1):
        rb = (0.03 * fz.median()).item()
        print(f"   node {k:2d} (t={cfg.mpc_dt*k:.2f}s):  "
              f"ex {_pct(er[:,0].abs(),.5):.4f}  ey {_pct(er[:,1].abs(),.5):.4f}  "
              f"ez {_pct(er[:,2].abs(),.5):.4f}  |  "
              f"p50 {_pct(tr[:,0].abs(),.5):6.2f} p95 {_pct(tr[:,0].abs(),.95):6.2f}  |  "
              f"p95/budget {_pct(tr[:,0].abs(),.95)/rb:.2f}")
    print()


if __name__ == "__main__":
    main()
