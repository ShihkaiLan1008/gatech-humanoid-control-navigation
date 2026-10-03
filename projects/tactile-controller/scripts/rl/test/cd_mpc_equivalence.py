"""Prove rl.decoupled.cd_mpc.cd == centroidal_mpc (jax_pimpc) on identical inputs.

``cd.py`` is the merged rewrite (jax-PiMPC only); ``centroidal_mpc.py`` is the
implementation every shipped checkpoint trained with.  Before any task is
switched onto ``cd`` the two must agree bit-for-bit-ish on the loco (nu=12)
path: same B_k, same box bounds, same solver parameters, and — since the jax
backend keeps no warm-start state — the SAME solution.

Three cases, all G1-realistic (B=64, N=10, dt=0.07, mass=G1_MASS):
  1. standing double support, c_bar=None
  2. walking mid-gait (random phase/state/twist), c_bar=None   (x_ref fallback)
  3. walking mid-gait, c_bar = measured-velocity extrapolation (production path)

Also diffs the shared MPCConfig fields programmatically so a silent default
drift (the R_delta 1e-4 vs 1e-3 kind) fails loudly instead of skewing every
future comparison.

Run:  uv run python scripts/rl/test/cd_mpc_equivalence.py
"""

import math

import torch

from rl.decoupled.cd_mpc import cd as cd_mod
from rl.decoupled.cd_mpc import centroidal_mpc as cm_mod
from rl.decoupled.cd_mpc.contact_schedule import (
    make_double_support_schedule,
    make_walking_schedule,
)
from rl.g1_assets.g1_constants import G1_MASS, HIP_WIDTH, PELVIS_HOME_Z

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, N, DT = 64, 10, 0.07
TOL = 1e-3  # both run the identical jax op sequence; expect ~0

# Fields with intentionally different names/absence between the two configs.
_NAME_MAP = {"max_iter": "admm_max_iter"}
_CD_ONLY = {"locomanipulation", "mu_hand", "f_hand_max", "R_f_hand"}
_CM_ONLY = {"solver_type", "admm_rho", "admm_rho_eq", "admm_eps_abs",
            "admm_eps_rel", "use_condensed", "unconstrained"}


def check_config_parity() -> None:
    cd_cfg = cd_mod.MPCConfig(mass=G1_MASS)
    cm_cfg = cm_mod.MPCConfig(mass=G1_MASS)
    drift = []
    for name, val in vars(cd_cfg).items():
        if name in _CD_ONLY:
            continue
        other = getattr(cm_cfg, _NAME_MAP.get(name, name), None)
        if other is None:
            drift.append(f"  {name}: missing in centroidal_mpc")
        elif other != val:
            drift.append(f"  {name}: cd={val}  centroidal={other}")
    missing_cm = [f for f in vars(cm_cfg) if f not in _CM_ONLY
                  and f not in vars(cd_cfg)
                  and f not in _NAME_MAP.values()]
    for f in missing_cm:
        drift.append(f"  {f}: missing in cd")
    assert not drift, "MPCConfig default drift:\n" + "\n".join(drift)
    print("config parity: shared fields identical")


def build_mpcs():
    cd_mpc = cd_mod.CentroidalMPC(
        cd_mod.MPCConfig(N=N, dt=DT, mass=G1_MASS), device=DEV)
    cm_mpc = cm_mod.CentroidalMPC(
        cm_mod.MPCConfig(N=N, dt=DT, mass=G1_MASS, solver_type="jax_pimpc"),
        device=DEV)
    return cd_mpc, cm_mpc


def standing_case():
    c0 = torch.tensor([0.0, 0.0, 0.699], device=DEV).expand(B, 3).contiguous()
    r_lf = torch.tensor([0.0,  0.11851, 0.0158], device=DEV).expand(B, 3).contiguous()
    r_rf = torch.tensor([0.0, -0.11851, 0.0158], device=DEV).expand(B, 3).contiguous()
    sched = make_double_support_schedule(B=B, N=N, r_LF=r_lf, r_RF=r_rf, device=DEV)

    x0 = torch.zeros(B, 9, device=DEV)
    x0[:, :3] = c0
    x_ref = x0.unsqueeze(1).expand(B, N + 1, 9).contiguous()
    u_ref = torch.zeros(B, N, 12, device=DEV)
    u_prev = torch.zeros(B, 12, device=DEV)
    u_prev[:, 2] = u_prev[:, 8] = G1_MASS * 9.81 / 2
    return sched, x0, x_ref, u_ref, u_prev, None


def walking_case(pass_c_bar: bool):
    g = torch.Generator(device="cpu").manual_seed(42 + int(pass_c_bar))

    def rand(*shape, lo, hi):
        return (lo + (hi - lo) * torch.rand(*shape, generator=g)).to(DEV)

    c = torch.stack([rand(B, lo=-0.5, hi=0.5), rand(B, lo=-0.5, hi=0.5),
                     rand(B, lo=PELVIS_HOME_Z - 0.03, hi=PELVIS_HOME_Z + 0.01)], -1)
    lv = torch.stack([rand(B, lo=-0.6, hi=1.0), rand(B, lo=-0.5, hi=0.5),
                      rand(B, lo=-0.1, hi=0.1)], -1)
    av_z = rand(B, lo=-0.5, hi=0.5)
    yaw = rand(B, lo=-math.pi, hi=math.pi)
    vx_b, vy_b = rand(B, lo=-0.5, hi=1.0), rand(B, lo=-0.5, hi=0.5)
    wz = rand(B, lo=-0.5, hi=0.5)
    phase = rand(B, lo=0.0, hi=2 * math.pi)

    cos_y, sin_y = yaw.cos(), yaw.sin()
    # feet roughly under the hips, mid-stride scatter
    hip = torch.stack([-sin_y, cos_y], -1) * HIP_WIDTH
    r_lf = torch.cat([c[:, :2] + hip + rand(B, 2, lo=-0.15, hi=0.15),
                      torch.full((B, 1), 0.0158, device=DEV)], -1)
    r_rf = torch.cat([c[:, :2] - hip + rand(B, 2, lo=-0.15, hi=0.15),
                      torch.full((B, 1), 0.0158, device=DEV)], -1)

    v_cmd_w = torch.stack([cos_y * vx_b - sin_y * vy_b,
                           sin_y * vx_b + cos_y * vy_b,
                           torch.zeros(B, device=DEV)], -1)
    sched = make_walking_schedule(
        B=B, N=N, r_LF=r_lf, r_RF=r_rf, gait_phase=phase, period=0.8, dt=DT,
        duty_factor=0.5, com_pos=c, v_cmd=v_cmd_w, yaw=yaw, yaw_rate=wz,
        hip_width=HIP_WIDTH, device=DEV)

    x0 = torch.cat([c, lv * G1_MASS,
                    torch.stack([torch.zeros(B, device=DEV),
                                 torch.zeros(B, device=DEV),
                                 av_z * 0.470], -1)], -1)

    # x_ref: commanded-velocity integration, nominal height (as the command builds it)
    k = torch.arange(N + 1, device=DEV, dtype=torch.float32)
    yaw_k = yaw.unsqueeze(1) + wz.unsqueeze(1) * k * DT
    vx_w = yaw_k.cos() * vx_b.unsqueeze(1) - yaw_k.sin() * vy_b.unsqueeze(1)
    vy_w = yaw_k.sin() * vx_b.unsqueeze(1) + yaw_k.cos() * vy_b.unsqueeze(1)
    x_ref = torch.zeros(B, N + 1, 9, device=DEV)
    x_ref[:, 0, 0], x_ref[:, 0, 1] = c[:, 0], c[:, 1]
    x_ref[:, 1:, 0] = c[:, 0:1] + torch.cumsum(vx_w[:, :-1] * DT, dim=1)
    x_ref[:, 1:, 1] = c[:, 1:2] + torch.cumsum(vy_w[:, :-1] * DT, dim=1)
    x_ref[:, :, 2] = PELVIS_HOME_Z
    x_ref[:, :, 3] = vx_w * G1_MASS
    x_ref[:, :, 4] = vy_w * G1_MASS
    x_ref[:, :, 8] = (0.470 * wz).unsqueeze(1)

    u_ref = torch.zeros(B, N, 12, device=DEV)
    u_prev = torch.zeros(B, 12, device=DEV)
    u_prev[:, 2] = u_prev[:, 8] = G1_MASS * 9.81 / 2

    c_bar = None
    if pass_c_bar:
        c_bar = c.unsqueeze(1).expand(B, N + 1, 3).clone()
        c_bar[:, :, 0] += lv[:, 0:1] * (k * DT).unsqueeze(0)
        c_bar[:, :, 1] += lv[:, 1:2] * (k * DT).unsqueeze(0)
    return sched, x0, x_ref, u_ref, u_prev, c_bar


def solve_both(cd_mpc, cm_mpc, case) -> dict[str, float]:
    sched, x0, x_ref, u_ref, u_prev, c_bar = case
    zero3 = torch.zeros(B, 3, device=DEV)
    cd_out = cd_mpc.solve(cd_mod.MPCInput(
        x0=x0, schedule=sched, x_ref=x_ref, u_ref=u_ref, u_prev=u_prev,
        other_body_force=zero3, other_body_torque=zero3, c_bar=c_bar))
    cm_out = cm_mpc.solve(cm_mod.MPCInput(
        x0=x0, schedule=sched, x_ref=x_ref, u_ref=u_ref, u_prev=u_prev,
        c_bar=c_bar))
    assert bool(cd_out.feasible.all()) and bool(cm_out.feasible.all())
    return {
        "u_star": float((cd_out.u_star - cm_out.u_star).abs().max()),
        "u_pred": float((cd_out.u_pred - cm_out.u_pred).abs().max()),
        "x_pred": float((cd_out.x_pred - cm_out.x_pred).abs().max()),
        "|u_star| scale": float(cm_out.u_star.abs().max()),
    }


def main() -> None:
    check_config_parity()
    cd_mpc, cm_mpc = build_mpcs()
    failures = []
    for name, case in [
        ("standing DS, c_bar=None", standing_case()),
        ("walking,     c_bar=None", walking_case(pass_c_bar=False)),
        ("walking,     c_bar=measured-extrapolation", walking_case(pass_c_bar=True)),
    ]:
        d = solve_both(cd_mpc, cm_mpc, case)
        line = "  ".join(f"{k} {v:.3e}" for k, v in d.items())
        bad = any(v > TOL for k, v in d.items() if not k.startswith("|"))
        print(f"[{'FAIL' if bad else ' ok '}] {name}:  {line}")
        if bad:
            failures.append(name)
    assert not failures, f"cd != centroidal on: {failures}"
    print("\nEQUIVALENT: cd.py matches centroidal_mpc (jax_pimpc) on all cases")


if __name__ == "__main__":
    main()
